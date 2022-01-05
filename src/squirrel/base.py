# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

import sys
import os

import math
import logging
import threading
import queue
from collections import defaultdict

from pyrocko.guts import Object, Int, List, Tuple, String, Timestamp, Dict
from pyrocko import util, trace
from pyrocko.progress import progress

from . import model, io, cache, dataset

from .model import to_kind_id, separator, WaveformOrder
from .client import fdsn, catalog
from .selection import Selection, filldocs
from .database import abspath
from . import client, environment, error

logger = logging.getLogger('psq.base')

guts_prefix = 'squirrel'


def make_task(*args):
    return progress.task(*args, logger=logger)


def lpick(condition, seq):
    ft = [], []
    for ele in seq:
        ft[int(bool(condition(ele)))].append(ele)

    return ft


def codes_fill(n, codes):
    return codes[:n] + ('*',) * (n-len(codes))


c_kind_to_ncodes = {
    'station': 4,
    'channel': 6,
    'response': 6,
    'waveform': 6,
    'event': 1,
    'waveform_promise': 6,
    'undefined': 1}


c_inflated = ['', '*', '*', '*', '*', '*']
c_offsets = [0, 2, 1, 1, 1, 1, 0]


def codes_inflate(codes):
    codes = codes[:6]
    inflated = list(c_inflated)
    ncodes = len(codes)
    offset = c_offsets[ncodes]
    inflated[offset:offset+ncodes] = codes
    return inflated


def codes_inflate2(codes):
    inflated = list(c_inflated)
    ncodes = len(codes)
    inflated[:ncodes] = codes
    return tuple(inflated)


def codes_patterns_for_kind(kind, codes):
    if not codes:
        return []

    if not isinstance(codes[0], str):
        out = []
        for subcodes in codes:
            out.extend(codes_patterns_for_kind(kind, subcodes))
        return out

    if kind in ('event', 'undefined'):
        return [codes]

    cfill = codes_inflate(codes)[:c_kind_to_ncodes[kind]]

    if kind == 'station':
        cfill2 = list(cfill)
        cfill2[3] = '[*]'
        return [cfill, cfill2]

    return [cfill]


def group_channels(channels):
    groups = defaultdict(list)
    for channel in channels:
        codes = channel.codes
        gcodes = codes[:-1] + (codes[-1][:-1],)
        groups[gcodes].append(channel)

    return groups


def pyrocko_station_from_channel_group(group, extra_args):
    list_of_args = [channel._get_pyrocko_station_args() for channel in group]
    args = util.consistency_merge(list_of_args + extra_args)
    from pyrocko import model as pmodel
    return pmodel.Station(
        network=args[0],
        station=args[1],
        location=args[2],
        lat=args[3],
        lon=args[4],
        elevation=args[5],
        depth=args[6],
        channels=[ch.get_pyrocko_channel() for ch in group])


def blocks(tmin, tmax, deltat, nsamples_block=100000):
    tblock = deltat * nsamples_block
    iblock_min = int(math.floor(tmin / tblock))
    iblock_max = int(math.ceil(tmax / tblock))
    for iblock in range(iblock_min, iblock_max):
        yield iblock * tblock, (iblock+1) * tblock


def gaps(avail, tmin, tmax):
    assert tmin < tmax

    data = [(tmax, 1), (tmin, -1)]
    for (tmin_a, tmax_a) in avail:
        assert tmin_a < tmax_a
        data.append((tmin_a, 1))
        data.append((tmax_a, -1))

    data.sort()
    s = 1
    gaps = []
    tmin_g = None
    for t, x in data:
        if s == 1 and x == -1:
            tmin_g = t
        elif s == 0 and x == 1 and tmin_g is not None:
            tmax_g = t
            if tmin_g != tmax_g:
                gaps.append((tmin_g, tmax_g))

        s += x

    return gaps


def order_key(order):
    return (order.codes, order.tmin, order.tmax)


class Batch(object):
    '''
    Batch of waveforms from window-wise data extraction.

    Encapsulates state and results yielded for each window in window-wise
    waveform extraction with the :py:meth:`Squirrel.chopper_waveforms` method.

    *Attributes:*

    .. py:attribute:: tmin

        Start of this time window.

    .. py:attribute:: tmax

        End of this time window.

    .. py:attribute:: i

        Index of this time window in sequence.

    .. py:attribute:: n

        Total number of time windows in sequence.

    .. py:attribute:: traces

        Extracted waveforms for this time window.
    '''

    def __init__(self, tmin, tmax, i, n, traces):
        self.tmin = tmin
        self.tmax = tmax
        self.i = i
        self.n = n
        self.traces = traces


class Squirrel(Selection):
    '''
    Prompt, lazy, indexing, caching, dynamic seismological dataset access.

    :param env:
        Squirrel environment instance or directory path to use as starting
        point for its detection. By default, the current directory is used as
        starting point. When searching for a usable environment the directory
        ``'.squirrel'`` or ``'squirrel'`` in the current (or starting point)
        directory is used if it exists, otherwise the parent directories are
        search upwards for the existence of such a directory. If no such
        directory is found, the user's global Squirrel environment
        ``'$HOME/.pyrocko/squirrel'`` is used.
    :type env:
        :py:class:`~pyrocko.squirrel.environment.Environment` or
        :py:class:`str`

    :param database:
        Database instance or path to database. By default the
        database found in the detected Squirrel environment is used.
    :type database:
        :py:class:`~pyrocko.squirrel.database.Database` or :py:class:`str`

    :param cache_path:
        Directory path to use for data caching. By default, the ``'cache'``
        directory in the detected Squirrel environment is used.
    :type cache_path:
        :py:class:`str`

    :param persistent:
        If given a name, create a persistent selection.
    :type persistent:
        :py:class:`str`

    This is the central class of the Squirrel framework. It provides a unified
    interface to query and access seismic waveforms, station meta-data and
    event information from local file collections and remote data sources. For
    prompt responses, a profound database setup is used under the hood. To
    speed up assemblage of ad-hoc data selections, files are indexed on first
    use and the extracted meta-data is remembered in the database for
    subsequent accesses. Bulk data is lazily loaded from disk and remote
    sources, just when requested. Once loaded, data is cached in memory to
    expedite typical access patterns. Files and data sources can be dynamically
    added to and removed from the Squirrel selection at runtime.

    Queries are restricted to the contents of the files currently added to the
    Squirrel selection (usually a subset of the file meta-information
    collection in the database). This list of files is referred to here as the
    "selection". By default, temporary tables are created in the attached
    database to hold the names of the files in the selection as well as various
    indices and counters. These tables are only visible inside the application
    which created them and are deleted when the database connection is closed
    or the application exits. To create a selection which is not deleted at
    exit, supply a name to the ``persistent`` argument of the Squirrel
    constructor. Persistent selections are shared among applications using the
    same database.

    **Method summary**

    Some of the methods are implemented in :py:class:`Squirrel`'s base class
    :py:class:`~pyrocko.squirrel.selection.Selection`.

    .. autosummary::

        ~Squirrel.add
        ~Squirrel.add_source
        ~Squirrel.add_fdsn
        ~Squirrel.add_catalog
        ~Squirrel.add_dataset
        ~Squirrel.add_virtual
        ~Squirrel.update
        ~Squirrel.update_waveform_promises
        ~Squirrel.advance_accessor
        ~Squirrel.clear_accessor
        ~Squirrel.reload
        ~pyrocko.squirrel.selection.Selection.iter_paths
        ~Squirrel.iter_nuts
        ~Squirrel.iter_kinds
        ~Squirrel.iter_deltats
        ~Squirrel.iter_codes
        ~Squirrel.iter_counts
        ~pyrocko.squirrel.selection.Selection.get_paths
        ~Squirrel.get_nuts
        ~Squirrel.get_kinds
        ~Squirrel.get_deltats
        ~Squirrel.get_codes
        ~Squirrel.get_counts
        ~Squirrel.get_time_span
        ~Squirrel.get_deltat_span
        ~Squirrel.get_nfiles
        ~Squirrel.get_nnuts
        ~Squirrel.get_total_size
        ~Squirrel.get_stats
        ~Squirrel.get_content
        ~Squirrel.get_stations
        ~Squirrel.get_channels
        ~Squirrel.get_responses
        ~Squirrel.get_events
        ~Squirrel.get_waveform_nuts
        ~Squirrel.get_waveforms
        ~Squirrel.chopper_waveforms
        ~Squirrel.get_coverage
        ~Squirrel.pile
        ~Squirrel.snuffle
        ~Squirrel.glob_codes
        ~pyrocko.squirrel.selection.Selection.get_database
        ~Squirrel.print_tables
    '''

    def __init__(
            self, env=None, database=None, cache_path=None, persistent=None):

        if not isinstance(env, environment.Environment):
            env = environment.get_environment(env)

        if database is None:
            database = env.expand_path(env.database_path)

        if cache_path is None:
            cache_path = env.expand_path(env.cache_path)

        if persistent is None:
            persistent = env.persistent

        Selection.__init__(
            self, database=database, persistent=persistent)

        self.get_database().set_basepath(os.path.dirname(env.get_basepath()))

        self._content_caches = {
            'waveform': cache.ContentCache(),
            'default': cache.ContentCache()}

        self._cache_path = cache_path

        self._sources = []
        self._operators = []
        self._operator_registry = {}

        self._pile = None
        self._n_choppers_active = 0

        self._names.update({
            'nuts': self.name + '_nuts',
            'kind_codes_count': self.name + '_kind_codes_count',
            'coverage': self.name + '_coverage'})

        with self.transaction() as cursor:
            self._create_tables_squirrel(cursor)

    def _create_tables_squirrel(self, cursor):

        cursor.execute(self._register_table(self._sql(
            '''
                CREATE TABLE IF NOT EXISTS %(db)s.%(nuts)s (
                    nut_id integer PRIMARY KEY,
                    file_id integer,
                    file_segment integer,
                    file_element integer,
                    kind_id integer,
                    kind_codes_id integer,
                    tmin_seconds integer,
                    tmin_offset integer,
                    tmax_seconds integer,
                    tmax_offset integer,
                    kscale integer)
            ''')))

        cursor.execute(self._register_table(self._sql(
            '''
                CREATE TABLE IF NOT EXISTS %(db)s.%(kind_codes_count)s (
                    kind_codes_id integer PRIMARY KEY,
                    count integer)
            ''')))

        cursor.execute(self._sql(
            '''
                CREATE UNIQUE INDEX IF NOT EXISTS %(db)s.%(nuts)s_file_element
                    ON %(nuts)s (file_id, file_segment, file_element)
            '''))

        cursor.execute(self._sql(
            '''
                CREATE INDEX IF NOT EXISTS %(db)s.%(nuts)s_index_file_id
                ON %(nuts)s (file_id)
            '''))

        cursor.execute(self._sql(
            '''
                CREATE INDEX IF NOT EXISTS %(db)s.%(nuts)s_index_tmin_seconds
                ON %(nuts)s (kind_id, tmin_seconds)
            '''))

        cursor.execute(self._sql(
            '''
                CREATE INDEX IF NOT EXISTS %(db)s.%(nuts)s_index_tmax_seconds
                ON %(nuts)s (kind_id, tmax_seconds)
            '''))

        cursor.execute(self._sql(
            '''
                CREATE INDEX IF NOT EXISTS %(db)s.%(nuts)s_index_kscale
                ON %(nuts)s (kind_id, kscale, tmin_seconds)
            '''))

        cursor.execute(self._sql(
            '''
                CREATE TRIGGER IF NOT EXISTS %(db)s.%(nuts)s_delete_nuts
                BEFORE DELETE ON main.files FOR EACH ROW
                BEGIN
                  DELETE FROM %(nuts)s WHERE file_id == old.file_id;
                END
            '''))

        # trigger only on size to make silent update of mtime possible
        cursor.execute(self._sql(
            '''
                CREATE TRIGGER IF NOT EXISTS %(db)s.%(nuts)s_delete_nuts2
                BEFORE UPDATE OF size ON main.files FOR EACH ROW
                BEGIN
                  DELETE FROM %(nuts)s WHERE file_id == old.file_id;
                END
            '''))

        cursor.execute(self._sql(
            '''
                CREATE TRIGGER IF NOT EXISTS
                    %(db)s.%(file_states)s_delete_files
                BEFORE DELETE ON %(db)s.%(file_states)s FOR EACH ROW
                BEGIN
                    DELETE FROM %(nuts)s WHERE file_id == old.file_id;
                END
            '''))

        cursor.execute(self._sql(
            '''
                CREATE TRIGGER IF NOT EXISTS %(db)s.%(nuts)s_inc_kind_codes
                BEFORE INSERT ON %(nuts)s FOR EACH ROW
                BEGIN
                    INSERT OR IGNORE INTO %(kind_codes_count)s VALUES
                    (new.kind_codes_id, 0);
                    UPDATE %(kind_codes_count)s
                    SET count = count + 1
                    WHERE new.kind_codes_id
                        == %(kind_codes_count)s.kind_codes_id;
                END
            '''))

        cursor.execute(self._sql(
            '''
                CREATE TRIGGER IF NOT EXISTS %(db)s.%(nuts)s_dec_kind_codes
                BEFORE DELETE ON %(nuts)s FOR EACH ROW
                BEGIN
                    UPDATE %(kind_codes_count)s
                    SET count = count - 1
                    WHERE old.kind_codes_id
                        == %(kind_codes_count)s.kind_codes_id;
                END
            '''))

        cursor.execute(self._register_table(self._sql(
            '''
                CREATE TABLE IF NOT EXISTS %(db)s.%(coverage)s (
                    kind_codes_id integer,
                    time_seconds integer,
                    time_offset integer,
                    step integer)
            ''')))

        cursor.execute(self._sql(
            '''
                CREATE UNIQUE INDEX IF NOT EXISTS %(db)s.%(coverage)s_time
                    ON %(coverage)s (kind_codes_id, time_seconds, time_offset)
            '''))

        cursor.execute(self._sql(
            '''
                CREATE TRIGGER IF NOT EXISTS %(db)s.%(nuts)s_add_coverage
                AFTER INSERT ON %(nuts)s FOR EACH ROW
                BEGIN
                    INSERT OR IGNORE INTO %(coverage)s VALUES
                    (new.kind_codes_id, new.tmin_seconds, new.tmin_offset, 0)
                    ;
                    UPDATE %(coverage)s
                    SET step = step + 1
                    WHERE new.kind_codes_id == %(coverage)s.kind_codes_id
                        AND new.tmin_seconds == %(coverage)s.time_seconds
                        AND new.tmin_offset == %(coverage)s.time_offset
                    ;
                    INSERT OR IGNORE INTO %(coverage)s VALUES
                    (new.kind_codes_id, new.tmax_seconds, new.tmax_offset, 0)
                    ;
                    UPDATE %(coverage)s
                    SET step = step - 1
                    WHERE new.kind_codes_id == %(coverage)s.kind_codes_id
                        AND new.tmax_seconds == %(coverage)s.time_seconds
                        AND new.tmax_offset == %(coverage)s.time_offset
                    ;
                    DELETE FROM %(coverage)s
                        WHERE new.kind_codes_id == %(coverage)s.kind_codes_id
                            AND new.tmin_seconds == %(coverage)s.time_seconds
                            AND new.tmin_offset == %(coverage)s.time_offset
                            AND step == 0
                    ;
                    DELETE FROM %(coverage)s
                        WHERE new.kind_codes_id == %(coverage)s.kind_codes_id
                            AND new.tmax_seconds == %(coverage)s.time_seconds
                            AND new.tmax_offset == %(coverage)s.time_offset
                            AND step == 0
                    ;
                END
            '''))

        cursor.execute(self._sql(
            '''
                CREATE TRIGGER IF NOT EXISTS %(db)s.%(nuts)s_remove_coverage
                BEFORE DELETE ON %(nuts)s FOR EACH ROW
                BEGIN
                    INSERT OR IGNORE INTO %(coverage)s VALUES
                    (old.kind_codes_id, old.tmin_seconds, old.tmin_offset, 0)
                    ;
                    UPDATE %(coverage)s
                    SET step = step - 1
                    WHERE old.kind_codes_id == %(coverage)s.kind_codes_id
                        AND old.tmin_seconds == %(coverage)s.time_seconds
                        AND old.tmin_offset == %(coverage)s.time_offset
                    ;
                    INSERT OR IGNORE INTO %(coverage)s VALUES
                    (old.kind_codes_id, old.tmax_seconds, old.tmax_offset, 0)
                    ;
                    UPDATE %(coverage)s
                    SET step = step + 1
                    WHERE old.kind_codes_id == %(coverage)s.kind_codes_id
                        AND old.tmax_seconds == %(coverage)s.time_seconds
                        AND old.tmax_offset == %(coverage)s.time_offset
                    ;
                    DELETE FROM %(coverage)s
                        WHERE old.kind_codes_id == %(coverage)s.kind_codes_id
                            AND old.tmin_seconds == %(coverage)s.time_seconds
                            AND old.tmin_offset == %(coverage)s.time_offset
                            AND step == 0
                    ;
                    DELETE FROM %(coverage)s
                        WHERE old.kind_codes_id == %(coverage)s.kind_codes_id
                            AND old.tmax_seconds == %(coverage)s.time_seconds
                            AND old.tmax_offset == %(coverage)s.time_offset
                            AND step == 0
                    ;
                END
            '''))

    def _delete(self):
        '''Delete database tables associated with this Squirrel.'''

        for s in '''
                DROP TRIGGER %(db)s.%(nuts)s_delete_nuts;
                DROP TRIGGER %(db)s.%(nuts)s_delete_nuts2;
                DROP TRIGGER %(db)s.%(file_states)s_delete_files;
                DROP TRIGGER %(db)s.%(nuts)s_inc_kind_codes;
                DROP TRIGGER %(db)s.%(nuts)s_dec_kind_codes;
                DROP TABLE %(db)s.%(nuts)s;
                DROP TABLE %(db)s.%(kind_codes_count)s;
                DROP TRIGGER IF EXISTS %(db)s.%(nuts)s_add_coverage;
                DROP TRIGGER IF EXISTS %(db)s.%(nuts)s_remove_coverage;
                DROP TABLE IF EXISTS %(db)s.%(coverage)s;
                '''.strip().splitlines():

            self._conn.execute(self._sql(s))

        Selection._delete(self)

    @filldocs
    def add(self,
            paths,
            kinds=None,
            format='detect',
            include=None,
            exclude=None,
            check=True):

        '''
        Add files to the selection.

        :param paths:
            Iterator yielding paths to files or directories to be added to the
            selection. Recurses into directories. If given a ``str``, it
            is treated as a single path to be added.
        :type paths:
            :py:class:`list` of :py:class:`str`

        :param kinds:
            Content types to be made available through the Squirrel selection.
            By default, all known content types are accepted.
        :type kinds:
            :py:class:`list` of :py:class:`str`

        :param format:
            File format identifier or ``'detect'`` to enable auto-detection
            (available: %(file_formats)s).
        :type format:
            str

        :param include:
            If not ``None``, files are only included if their paths match the
            given regular expression pattern.
        :type format:
            str

        :param exclude:
            If not ``None``, files are only included if their paths do not
            match the given regular expression pattern.
        :type format:
            str

        :param check:
            If ``True``, all file modification times are checked to see if
            cached information has to be updated (slow). If ``False``, only
            previously unknown files are indexed and cached information is used
            for known files, regardless of file state (fast, corrresponds to
            Squirrel's ``--optimistic`` mode). File deletions will go
            undetected in the latter case.
        :type check:
            bool

        :Complexity:
            O(log N)
        '''

        if isinstance(kinds, str):
            kinds = (kinds,)

        if isinstance(paths, str):
            paths = [paths]

        kind_mask = model.to_kind_mask(kinds)

        with progress.view():
            Selection.add(
                self, util.iter_select_files(
                    paths,
                    show_progress=False,
                    include=include,
                    exclude=exclude,
                    pass_through=lambda path: path.startswith('virtual:')
                ), kind_mask, format)

            self._load(check)
            self._update_nuts()

    def reload(self):
        '''
        Check for modifications and reindex modified files.

        Based on file modification times.
        '''

        self._set_file_states_force_check()
        self._load(check=True)
        self._update_nuts()

    def add_virtual(self, nuts, virtual_paths=None):
        '''
        Add content which is not backed by files.

        :param nuts:
            Content pieces to be added.
        :type nuts:
            iterator yielding :py:class:`~pyrocko.squirrel.model.Nut` objects

        :param virtual_paths:
            List of virtual paths to prevent creating a temporary list of the
            nuts while aggregating the file paths for the selection.
        :type virtual_paths:
            :py:class:`list` of :py:class:`str`

        Stores to the main database and the selection.
        '''

        if isinstance(virtual_paths, str):
            virtual_paths = [virtual_paths]

        if virtual_paths is None:
            if not isinstance(nuts, list):
                nuts = list(nuts)
            virtual_paths = set(nut.file_path for nut in nuts)

        Selection.add(self, virtual_paths)
        self.get_database().dig(nuts)
        self._update_nuts()

    def add_volatile(self, nuts):
        if not isinstance(nuts, list):
            nuts = list(nuts)

        paths = list(set(nut.file_path for nut in nuts))
        io.backends.virtual.add_nuts(nuts)
        self.add_virtual(nuts, paths)
        self._volatile_paths.extend(paths)

    def add_volatile_waveforms(self, traces):
        '''
        Add in-memory waveforms which will be removed when the app closes.
        '''

        name = model.random_name()

        path = 'virtual:volatile:%s' % name

        nuts = []
        for itr, tr in enumerate(traces):
            assert tr.tmin <= tr.tmax
            tmin_seconds, tmin_offset = model.tsplit(tr.tmin)
            tmax_seconds, tmax_offset = model.tsplit(
                tr.tmin + tr.data_len()*tr.deltat)

            nuts.append(model.Nut(
                file_path=path,
                file_format='virtual',
                file_segment=itr,
                file_element=0,
                codes=separator.join(tr.codes),
                tmin_seconds=tmin_seconds,
                tmin_offset=tmin_offset,
                tmax_seconds=tmax_seconds,
                tmax_offset=tmax_offset,
                deltat=tr.deltat,
                kind_id=to_kind_id('waveform'),
                content=tr))

        self.add_volatile(nuts)
        return path

    def _load(self, check):
        for _ in io.iload(
                self,
                content=[],
                skip_unchanged=True,
                check=check):
            pass

    def _update_nuts(self):
        transaction = self.transaction()
        with make_task('Aggregating selection') as task, \
                transaction as cursor:

            self._conn.set_progress_handler(task.update, 100000)
            nrows = cursor.execute(self._sql(
                '''
                    INSERT INTO %(db)s.%(nuts)s
                    SELECT NULL,
                        nuts.file_id, nuts.file_segment, nuts.file_element,
                        nuts.kind_id, nuts.kind_codes_id,
                        nuts.tmin_seconds, nuts.tmin_offset,
                        nuts.tmax_seconds, nuts.tmax_offset,
                        nuts.kscale
                    FROM %(db)s.%(file_states)s
                    INNER JOIN nuts
                        ON %(db)s.%(file_states)s.file_id == nuts.file_id
                    INNER JOIN kind_codes
                        ON nuts.kind_codes_id ==
                           kind_codes.kind_codes_id
                    WHERE %(db)s.%(file_states)s.file_state != 2
                        AND (((1 << kind_codes.kind_id)
                            & %(db)s.%(file_states)s.kind_mask) != 0)
                ''')).rowcount

            task.update(nrows)
            self._set_file_states_known(transaction)
            self._conn.set_progress_handler(None, 0)

    def add_source(self, source, check=True):
        '''
        Add remote resource.

        :param source:
            Remote data access client instance.
        :type source:
           subclass of :py:class:`~pyrocko.squirrel.client.base.Source`
        '''

        self._sources.append(source)
        source.setup(self, check=check)

    def add_fdsn(self, *args, **kwargs):
        '''
        Add FDSN site for transparent remote data access.

        Arguments are passed to
        :py:class:`~pyrocko.squirrel.client.fdsn.FDSNSource`.
        '''

        self.add_source(fdsn.FDSNSource(*args, **kwargs))

    def add_catalog(self, *args, **kwargs):
        '''
        Add online catalog for transparent event data access.

        Arguments are passed to
        :py:class:`~pyrocko.squirrel.client.catalog.CatalogSource`.
        '''

        self.add_source(catalog.CatalogSource(*args, **kwargs))

    def add_dataset(self, ds, check=True, warn_persistent=True):
        '''
        Read dataset description from file and add its contents.

        :param ds:
            Path to dataset description file or dataset description object
            . See :py:mod:`~pyrocko.squirrel.dataset`.
        :type ds:
            :py:class:`str` or :py:class:`~pyrocko.squirrel.dataset.Dataset`

        :param check:
            If ``True``, all file modification times are checked to see if
            cached information has to be updated (slow). If ``False``, only
            previously unknown files are indexed and cached information is used
            for known files, regardless of file state (fast, corrresponds to
            Squirrel's ``--optimistic`` mode). File deletions will go
            undetected in the latter case.
        :type check:
            bool
        '''
        if isinstance(ds, str):
            ds = dataset.read_dataset(ds)
            path = ds
        else:
            path = None

        if warn_persistent and ds.persistent and (
                not self._persistent or (self._persistent != ds.persistent)):

            logger.warning(
                'Dataset `persistent` flag ignored. Can not be set on already '
                'existing Squirrel instance.%s' % (
                    ' Dataset: %s' % path if path else ''))

        ds.setup(self, check=check)

    def _get_selection_args(
            self, obj=None, tmin=None, tmax=None, time=None, codes=None):

        if time is not None:
            tmin = time
            tmax = time

        if obj is not None:
            tmin = tmin if tmin is not None else obj.tmin
            tmax = tmax if tmax is not None else obj.tmax
            codes = codes if codes is not None else codes_inflate2(obj.codes)

        if isinstance(codes, str):
            codes = tuple(codes.split('.'))

        return tmin, tmax, codes

    def _selection_args_to_kwargs(
            self, obj=None, tmin=None, tmax=None, time=None, codes=None):

        return dict(obj=obj, tmin=tmin, tmax=tmax, time=time, codes=codes)

    def _timerange_sql(self, tmin, tmax, kind, cond, args, naiv):

        tmin_seconds, tmin_offset = model.tsplit(tmin)
        tmax_seconds, tmax_offset = model.tsplit(tmax)
        if naiv:
            cond.append('%(db)s.%(nuts)s.tmin_seconds <= ?')
            args.append(tmax_seconds)
        else:
            tscale_edges = model.tscale_edges
            tmin_cond = []
            for kscale in range(tscale_edges.size + 1):
                if kscale != tscale_edges.size:
                    tscale = int(tscale_edges[kscale])
                    tmin_cond.append('''
                        (%(db)s.%(nuts)s.kind_id = ?
                         AND %(db)s.%(nuts)s.kscale == ?
                         AND %(db)s.%(nuts)s.tmin_seconds BETWEEN ? AND ?)
                    ''')
                    args.extend(
                        (to_kind_id(kind), kscale,
                         tmin_seconds - tscale - 1, tmax_seconds + 1))

                else:
                    tmin_cond.append('''
                        (%(db)s.%(nuts)s.kind_id == ?
                         AND %(db)s.%(nuts)s.kscale == ?
                         AND %(db)s.%(nuts)s.tmin_seconds <= ?)
                    ''')

                    args.extend(
                        (to_kind_id(kind), kscale, tmax_seconds + 1))
            if tmin_cond:
                cond.append(' ( ' + ' OR '.join(tmin_cond) + ' ) ')

        cond.append('%(db)s.%(nuts)s.tmax_seconds >= ?')
        args.append(tmin_seconds)

    def iter_nuts(
            self, kind=None, tmin=None, tmax=None, codes=None, naiv=False,
            kind_codes_ids=None, path=None):

        '''
        Iterate over content entities matching given constraints.

        :param kind:
            Content kind (or kinds) to extract.
        :type kind:
            :py:class:`str`, :py:class:`list` of :py:class:`str`

        :param tmin:
            Start time of query interval.
        :type tmin:
            timestamp

        :param tmax:
            End time of query interval.
        :type tmax:
            timestamp

        :param codes:
            Pattern of content codes to query.
        :type codes:
            :py:class:`tuple` of :py:class:`str`

        :param naiv:
            Bypass time span lookup through indices (slow, for testing).
        :type naiv:
            :py:class:`bool`

        :param kind_codes_ids:
            Kind-codes IDs of contents to be retrieved (internal use).
        :type kind_codes_ids:
            :py:class:`list` of :py:class:`str`

        :yields:
            :py:class:`~pyrocko.squirrel.model.Nut` objects representing the
            intersecting content.

        :complexity:
            O(log N) for the time selection part due to heavy use of database
            indices.

        Query time span is treated as a half-open interval ``[tmin, tmax)``.
        However, if ``tmin`` equals ``tmax``, the edge logics are modified to
        closed-interval so that content intersecting with the time instant ``t
        = tmin = tmax`` is returned (otherwise nothing would be returned as
        ``[t, t)`` never matches anything).

        Time spans of content entities to be matched are also treated as half
        open intervals, e.g. content span ``[0, 1)`` is matched by query span
        ``[0, 1)`` but not by ``[-1, 0)`` or ``[1, 2)``. Also here, logics are
        modified to closed-interval when the content time span is an empty
        interval, i.e. to indicate a time instant. E.g. time instant 0 is
        matched by ``[0, 1)`` but not by ``[-1, 0)`` or ``[1, 2)``.
        '''

        if not isinstance(kind, str):
            if kind is None:
                kind = model.g_content_kinds
            for kind_ in kind:
                for nut in self.iter_nuts(kind_, tmin, tmax, codes):
                    yield nut

            return

        cond = []
        args = []
        if tmin is not None or tmax is not None:
            assert kind is not None
            if tmin is None:
                tmin = self.get_time_span()[0]
            if tmax is None:
                tmax = self.get_time_span()[1] + 1.0

            self._timerange_sql(tmin, tmax, kind, cond, args, naiv)

        elif kind is not None:
            cond.append('kind_codes.kind_id == ?')
            args.append(to_kind_id(kind))

        if codes is not None:
            pats = codes_patterns_for_kind(kind, codes)
            if pats:
                cond.append(
                    ' ( %s ) ' % ' OR '.join(
                        ('kind_codes.codes GLOB ?',) * len(pats)))
                args.extend(separator.join(pat) for pat in pats)

        if kind_codes_ids is not None:
            cond.append(
                ' ( kind_codes.kind_codes_id IN ( %s ) ) ' % ', '.join(
                    '?'*len(kind_codes_ids)))

            args.extend(kind_codes_ids)

        db = self.get_database()
        if path is not None:
            cond.append('files.path == ?')
            args.append(db.relpath(abspath(path)))

        sql = ('''
            SELECT
                files.path,
                files.format,
                files.mtime,
                files.size,
                %(db)s.%(nuts)s.file_segment,
                %(db)s.%(nuts)s.file_element,
                kind_codes.kind_id,
                kind_codes.codes,
                %(db)s.%(nuts)s.tmin_seconds,
                %(db)s.%(nuts)s.tmin_offset,
                %(db)s.%(nuts)s.tmax_seconds,
                %(db)s.%(nuts)s.tmax_offset,
                kind_codes.deltat
            FROM files
            INNER JOIN %(db)s.%(nuts)s
                ON files.file_id == %(db)s.%(nuts)s.file_id
            INNER JOIN kind_codes
                ON %(db)s.%(nuts)s.kind_codes_id == kind_codes.kind_codes_id
            ''')

        if cond:
            sql += ''' WHERE ''' + ' AND '.join(cond)

        sql = self._sql(sql)
        if tmin is None and tmax is None:
            for row in self._conn.execute(sql, args):
                row = (db.abspath(row[0]),) + row[1:]
                nut = model.Nut(values_nocheck=row)
                yield nut
        else:
            assert tmin is not None and tmax is not None
            if tmin == tmax:
                for row in self._conn.execute(sql, args):
                    row = (db.abspath(row[0]),) + row[1:]
                    nut = model.Nut(values_nocheck=row)
                    if (nut.tmin <= tmin < nut.tmax) \
                            or (nut.tmin == nut.tmax and tmin == nut.tmin):

                        yield nut
            else:
                for row in self._conn.execute(sql, args):
                    row = (db.abspath(row[0]),) + row[1:]
                    nut = model.Nut(values_nocheck=row)
                    if (tmin < nut.tmax and nut.tmin < tmax) \
                            or (nut.tmin == nut.tmax
                                and tmin <= nut.tmin < tmax):

                        yield nut

    def get_nuts(self, *args, **kwargs):
        '''
        Get content entities matching given constraints.

        Like :py:meth:`iter_nuts` but returns results as a list.
        '''

        return list(self.iter_nuts(*args, **kwargs))

    def _split_nuts(
            self, kind, tmin=None, tmax=None, codes=None, path=None):

        tmin_seconds, tmin_offset = model.tsplit(tmin)
        tmax_seconds, tmax_offset = model.tsplit(tmax)

        names_main_nuts = dict(self._names)
        names_main_nuts.update(db='main', nuts='nuts')

        db = self.get_database()

        def main_nuts(s):
            return s % names_main_nuts

        with self.transaction() as cursor:
            # modify selection and main
            for sql_subst in [
                    self._sql, main_nuts]:

                cond = []
                args = []

                self._timerange_sql(tmin, tmax, kind, cond, args, False)

                if codes is not None:
                    pats = codes_patterns_for_kind(kind, codes)
                    if pats:
                        cond.append(
                            ' ( %s ) ' % ' OR '.join(
                                ('kind_codes.codes GLOB ?',) * len(pats)))
                        args.extend(separator.join(pat) for pat in pats)

                if path is not None:
                    cond.append('files.path == ?')
                    args.append(db.relpath(abspath(path)))

                sql = sql_subst('''
                    SELECT
                        %(db)s.%(nuts)s.nut_id,
                        %(db)s.%(nuts)s.tmin_seconds,
                        %(db)s.%(nuts)s.tmin_offset,
                        %(db)s.%(nuts)s.tmax_seconds,
                        %(db)s.%(nuts)s.tmax_offset,
                        kind_codes.deltat
                    FROM files
                    INNER JOIN %(db)s.%(nuts)s
                        ON files.file_id == %(db)s.%(nuts)s.file_id
                    INNER JOIN kind_codes
                        ON %(db)s.%(nuts)s.kind_codes_id == kind_codes.kind_codes_id
                    WHERE ''' + ' AND '.join(cond))  # noqa

                insert = []
                delete = []
                for row in cursor.execute(sql, args):
                    nut_id, nut_tmin_seconds, nut_tmin_offset, \
                        nut_tmax_seconds, nut_tmax_offset, nut_deltat = row

                    nut_tmin = model.tjoin(
                        nut_tmin_seconds, nut_tmin_offset)
                    nut_tmax = model.tjoin(
                        nut_tmax_seconds, nut_tmax_offset)

                    if nut_tmin < tmax and tmin < nut_tmax:
                        if nut_tmin < tmin:
                            insert.append((
                                nut_tmin_seconds, nut_tmin_offset,
                                tmin_seconds, tmin_offset,
                                model.tscale_to_kscale(
                                    tmin_seconds - nut_tmin_seconds),
                                nut_id))

                        if tmax < nut_tmax:
                            insert.append((
                                tmax_seconds, tmax_offset,
                                nut_tmax_seconds, nut_tmax_offset,
                                model.tscale_to_kscale(
                                    nut_tmax_seconds - tmax_seconds),
                                nut_id))

                        delete.append((nut_id,))

                sql_add = '''
                    INSERT INTO %(db)s.%(nuts)s (
                            file_id, file_segment, file_element, kind_id,
                            kind_codes_id, tmin_seconds, tmin_offset,
                            tmax_seconds, tmax_offset, kscale )
                        SELECT
                            file_id, file_segment, file_element,
                            kind_id, kind_codes_id, ?, ?, ?, ?, ?
                        FROM %(db)s.%(nuts)s
                        WHERE nut_id == ?
                '''
                cursor.executemany(sql_subst(sql_add), insert)

                sql_delete = '''
                    DELETE FROM %(db)s.%(nuts)s WHERE nut_id == ?
                '''
                cursor.executemany(sql_subst(sql_delete), delete)

    def get_time_span(self, kinds=None):
        '''
        Get time interval over all content in selection.

        :complexity:
            O(1), independent of the number of nuts.

        :returns: (tmin, tmax)
        '''

        sql_min = self._sql('''
            SELECT MIN(tmin_seconds), MIN(tmin_offset)
            FROM %(db)s.%(nuts)s
            WHERE kind_id == ?
                AND tmin_seconds == (
                    SELECT MIN(tmin_seconds)
                    FROM %(db)s.%(nuts)s
                    WHERE kind_id == ?)
        ''')

        sql_max = self._sql('''
            SELECT MAX(tmax_seconds), MAX(tmax_offset)
            FROM %(db)s.%(nuts)s
            WHERE kind_id == ?
                AND tmax_seconds == (
                    SELECT MAX(tmax_seconds)
                    FROM %(db)s.%(nuts)s
                    WHERE kind_id == ?)
        ''')

        gtmin = None
        gtmax = None

        if isinstance(kinds, str):
            kinds = [kinds]

        if kinds is None:
            kind_ids = model.g_content_kind_ids
        else:
            kind_ids = model.to_kind_ids(kinds)

        for kind_id in kind_ids:
            for tmin_seconds, tmin_offset in self._conn.execute(
                    sql_min, (kind_id, kind_id)):
                tmin = model.tjoin(tmin_seconds, tmin_offset)
                if tmin is not None and (gtmin is None or tmin < gtmin):
                    gtmin = tmin

            for (tmax_seconds, tmax_offset) in self._conn.execute(
                    sql_max, (kind_id, kind_id)):
                tmax = model.tjoin(tmax_seconds, tmax_offset)
                if tmax is not None and (gtmax is None or tmax > gtmax):
                    gtmax = tmax

        return gtmin, gtmax

    def has(self, kinds):
        '''
        Check availability of given content kinds.

        :param kinds:
            Content kinds to query.
        :type kind:
            list of str

        :returns:
            ``True`` if any of the queried content kinds is available
            in the selection.
        '''
        self_tmin, self_tmax = self.get_time_span(kinds)

        return None not in (self_tmin, self_tmax)

    def get_deltat_span(self, kind):
        '''
        Get min and max sampling interval of all content of given kind.

        :param kind:
            Content kind
        :type kind:
            str

        :returns: (deltat_min, deltat_max)
        '''

        deltats = [
            deltat for deltat in self.get_deltats(kind)
            if deltat is not None]

        if deltats:
            return min(deltats), max(deltats)
        else:
            return None, None

    def iter_kinds(self, codes=None):
        '''
        Iterate over content types available in selection.

        :param codes:
            If given, get kinds only for selected codes identifier.
        :type codes:
            :py:class:`tuple` of :py:class:`str`

        :yields:
            Available content kinds as :py:class:`str`.

        :complexity:
            O(1), independent of number of nuts.
        '''

        return self._database._iter_kinds(
            codes=codes,
            kind_codes_count='%(db)s.%(kind_codes_count)s' % self._names)

    def iter_deltats(self, kind=None):
        '''
        Iterate over sampling intervals available in selection.

        :param kind:
            If given, get sampling intervals only for a given content type.
        :type kind:
            str

        :yields:
            :py:class:`float` values.

        :complexity:
            O(1), independent of number of nuts.
        '''
        return self._database._iter_deltats(
            kind=kind,
            kind_codes_count='%(db)s.%(kind_codes_count)s' % self._names)

    def iter_codes(self, kind=None):
        '''
        Iterate over content identifier code sequences available in selection.

        :param kind:
            If given, get codes only for a given content type.
        :type kind:
            str

        :yields:
            :py:class:`tuple` of :py:class:`str`

        :complexity:
            O(1), independent of number of nuts.
        '''
        return self._database._iter_codes(
            kind=kind,
            kind_codes_count='%(db)s.%(kind_codes_count)s' % self._names)

    def iter_counts(self, kind=None):
        '''
        Iterate over number of occurrences of any (kind, codes) combination.

        :param kind:
            If given, get counts only for selected content type.
        :type kind:
            str

        :yields:
            Tuples of the form ``((kind, codes), count)``.

        :complexity:
            O(1), independent of number of nuts.
        '''
        return self._database._iter_counts(
            kind=kind,
            kind_codes_count='%(db)s.%(kind_codes_count)s' % self._names)

    def get_kinds(self, codes=None):
        '''
        Get content types available in selection.

        :param codes:
            If given, get kinds only for selected codes identifier.
        :type codes:
            :py:class:`tuple` of :py:class:`str`

        :returns:
            Sorted list of available content types.

        :complexity:
            O(1), independent of number of nuts.

        '''
        return sorted(list(self.iter_kinds(codes=codes)))

    def get_deltats(self, kind=None):
        '''
        Get sampling intervals available in selection.

        :param kind:
            If given, get codes only for selected content type.
        :type kind:
            str

        :complexity:
            O(1), independent of number of nuts.

        :returns: sorted list of available sampling intervals
        '''
        return sorted(list(self.iter_deltats(kind=kind)))

    def get_codes(self, kind=None):
        '''
        Get identifier code sequences available in selection.

        :param kind:
            If given, get codes only for selected content type.
        :type kind:
            str

        :complexity:
            O(1), independent of number of nuts.

        :returns: sorted list of available codes as tuples of strings
        '''
        return sorted(list(self.iter_codes(kind=kind)))

    def get_counts(self, kind=None):
        '''
        Get number of occurrences of any (kind, codes) combination.

        :param kind:
            If given, get codes only for selected content type.
        :type kind:
            str

        :complexity:
            O(1), independent of number of nuts.

        :returns: ``dict`` with ``counts[kind][codes]`` or ``counts[codes]``
            if kind is not ``None``
        '''
        d = {}
        for (k, codes, deltat), count in self.iter_counts():
            if k not in d:
                v = d[k] = {}
            else:
                v = d[k]

            if codes not in v:
                v[codes] = 0

            v[codes] += count

        if kind is not None:
            return d[kind]
        else:
            return d

    def glob_codes(self, kind, codes_list):
        '''
        Find codes matching given patterns.

        :param kind:
            Content kind to be queried.
        :type kind:
            str

        :param codes_list:
            List of code patterns to query. If not given or empty, an empty
            list is returned.
        :type codes_list:
            :py:class:`list` of :py:class:`tuple` of :py:class:`str`

        :returns:
            List of matches of the form ``[kind_codes_id, codes, deltat]``.
        '''

        args = [to_kind_id(kind)]
        pats = []
        for codes in codes_list:
            pats.extend(codes_patterns_for_kind(kind, codes))

        codes_cond = ' ( %s ) ' % ' OR '.join(
                ('kind_codes.codes GLOB ?',) * len(pats))

        args.extend(separator.join(pat) for pat in pats)

        sql = self._sql('''
            SELECT kind_codes_id, codes, deltat FROM kind_codes
            WHERE
                kind_id == ?
                AND ''' + codes_cond)

        return list(map(list, self._conn.execute(sql, args)))

    def update(self, constraint=None, **kwargs):
        '''
        Update or partially update channel and event inventories.

        :param constraint:
            Selection of times or areas to be brought up to date.
        :type constraint:
            :py:class:`~pyrocko.squirrel.client.Constraint`

        :param \\*\\*kwargs:
            Shortcut for setting ``constraint=Constraint(**kwargs)``.

        This function triggers all attached remote sources, to check for
        updates in the meta-data. The sources will only submit queries when
        their expiration date has passed, or if the selection spans into
        previously unseen times or areas.
        '''

        if constraint is None:
            constraint = client.Constraint(**kwargs)

        for source in self._sources:
            source.update_channel_inventory(self, constraint)
            source.update_event_inventory(self, constraint)

    def update_waveform_promises(self, constraint=None, **kwargs):
        '''
        Permit downloading of remote waveforms.

        :param constraint:
            Remote waveforms compatible with the given constraint are enabled
            for download.
        :type constraint:
            :py:class:`~pyrocko.squirrel.client.Constraint`

        :param \\*\\*kwargs:
            Shortcut for setting ``constraint=Constraint(**kwargs)``.

        Calling this method permits Squirrel to download waveforms from remote
        sources when processing subsequent waveform requests. This works by
        inserting so called waveform promises into the database. It will look
        into the available channels for each remote source and create a promise
        for each channel compatible with the given constraint. If the promise
        then matches in a waveform request, Squirrel tries to download the
        waveform. If the download is successful, the downloaded waveform is
        added to the Squirrel and the promise is deleted. If the download
        fails, the promise is kept if the reason of failure looks like being
        temporary, e.g. because of a network failure. If the cause of failure
        however seems to be permanent, the promise is deleted so that no
        further attempts are made to download a waveform which might not be
        available from that server at all. To force re-scheduling after a
        permanent failure, call :py:meth:`update_waveform_promises`
        yet another time.
        '''

        if constraint is None:
            constraint = client.Constraint(**kwargs)

        # TODO
        print('contraint ignored atm')

        for source in self._sources:
            source.update_waveform_promises(self)

    def update_responses(self, constraint=None, **kwargs):
        # TODO
        if constraint is None:
            constraint = client.Constraint(**kwargs)

        print('contraint ignored atm')
        for source in self._sources:
            source.update_response_inventory(self, constraint)

    def get_nfiles(self):
        '''
        Get number of files in selection.
        '''

        sql = self._sql('''SELECT COUNT(*) FROM %(db)s.%(file_states)s''')
        for row in self._conn.execute(sql):
            return row[0]

    def get_nnuts(self):
        '''
        Get number of nuts in selection.
        '''

        sql = self._sql('''SELECT COUNT(*) FROM %(db)s.%(nuts)s''')
        for row in self._conn.execute(sql):
            return row[0]

    def get_total_size(self):
        '''
        Get aggregated file size available in selection.
        '''

        sql = self._sql('''
            SELECT SUM(files.size) FROM %(db)s.%(file_states)s
            INNER JOIN files
                ON %(db)s.%(file_states)s.file_id = files.file_id
        ''')

        for row in self._conn.execute(sql):
            return row[0] or 0

    def get_stats(self):
        '''
        Get statistics on contents available through this selection.
        '''

        kinds = self.get_kinds()
        time_spans = {}
        for kind in kinds:
            time_spans[kind] = self.get_time_span([kind])

        return SquirrelStats(
            nfiles=self.get_nfiles(),
            nnuts=self.get_nnuts(),
            kinds=kinds,
            codes=self.get_codes(),
            total_size=self.get_total_size(),
            counts=self.get_counts(),
            time_spans=time_spans,
            sources=[s.describe() for s in self._sources],
            operators=[op.describe() for op in self._operators])

    def get_content(
            self,
            nut,
            cache_id='default',
            accessor_id='default',
            show_progress=False):

        '''
        Get and possibly load full content for a given index entry from file.

        Loads the actual content objects (channel, station, waveform, ...) from
        file. For efficiency sibling content (all stuff in the same file
        segment) will also be loaded as a side effect. The loaded contents are
        cached in the Squirrel object.
        '''

        content_cache = self._content_caches[cache_id]
        if not content_cache.has(nut):

            for nut_loaded in io.iload(
                    nut.file_path,
                    segment=nut.file_segment,
                    format=nut.file_format,
                    database=self._database,
                    show_progress=show_progress):

                content_cache.put(nut_loaded)

        try:
            return content_cache.get(nut, accessor_id)
        except KeyError:
            raise error.NotAvailable(
                'Unable to retrieve content: %s, %s, %s, %s' % nut.key)

    def advance_accessor(self, accessor_id, cache_id=None):
        '''
        Notify memory caches about consumer moving to a new data batch.

        :param accessor_id:
            Name of accessing consumer to be advanced.
        :type accessor_id:
            str

        :param cache_id:
            Name of cache to for which the accessor should be advanced. By
            default the named accessor is advanced in all registered caches.
            By default, two caches named ``'default'`` and ``'waveforms'`` are
            available.
        :type cache_id:
            str

        See :py:class:`~pyrocko.squirrel.cache.ContentCache` for details on how
        Squirrel's memory caching works and can be tuned. Default behaviour is
        to release data when it has not been used in the latest data
        window/batch. If the accessor is never advanced, data is cached
        indefinitely - which is often desired e.g. for station meta-data.
        Methods for consecutive data traversal, like
        :py:meth:`chopper_waveforms` automatically advance and clear
        their accessor.
        '''
        for cache_ in (
                self._content_caches.keys()
                if cache_id is None
                else [cache_id]):

            self._content_caches[cache_].advance_accessor(accessor_id)

    def clear_accessor(self, accessor_id, cache_id=None):
        '''
        Notify memory caches about a consumer having finished.

        :param accessor_id:
            Name of accessor to be cleared.
        :type accessor_id:
            str

        :param cache_id:
            Name of cache to for which the accessor should be cleared. By
            default the named accessor is cleared from all registered caches.
            By default, two caches named ``'default'`` and ``'waveforms'`` are
            available.
        :type cache_id:
            str

        Calling this method clears all references to cache entries held by the
        named accessor. Cache entries are then freed if not referenced by any
        other accessor.
        '''

        for cache_ in (
                self._content_caches.keys()
                if cache_id is None
                else [cache_id]):

            self._content_caches[cache_].clear_accessor(accessor_id)

    def _check_duplicates(self, nuts):
        d = defaultdict(list)
        for nut in nuts:
            d[nut.codes].append(nut)

        for codes, group in d.items():
            if len(group) > 1:
                logger.warning(
                    'Multiple entries matching codes: %s'
                    % '.'.join(codes.split(separator)))

    @filldocs
    def get_stations(
            self, obj=None, tmin=None, tmax=None, time=None, codes=None,
            model='squirrel'):

        '''
        Get stations matching given constraints.

        %(query_args)s

        :param model:
            Select object model for returned values: ``'squirrel'`` to get
            Squirrel station objects or ``'pyrocko'`` to get Pyrocko station
            objects with channel information attached.
        :type model:
            str

        :returns:
            List of :py:class:`pyrocko.squirrel.Station
            <pyrocko.squirrel.model.Station>` objects by default or list of
            :py:class:`pyrocko.model.Station <pyrocko.model.station.Station>`
            objects if ``model='pyrocko'`` is requested.

        See :py:meth:`iter_nuts` for details on time span matching.
        '''

        if model == 'pyrocko':
            return self._get_pyrocko_stations(obj, tmin, tmax, time, codes)
        elif model == 'squirrel':
            args = self._get_selection_args(obj, tmin, tmax, time, codes)
            nuts = sorted(
                self.iter_nuts('station', *args), key=lambda nut: nut.dkey)
            self._check_duplicates(nuts)
            return [self.get_content(nut) for nut in nuts]
        else:
            raise ValueError('Invalid station model: %s' % model)

    @filldocs
    def get_channels(
            self, obj=None, tmin=None, tmax=None, time=None, codes=None):

        '''
        Get channels matching given constraints.

        %(query_args)s

        :returns:
            List of :py:class:`~pyrocko.squirrel.model.Channel` objects.

        See :py:meth:`iter_nuts` for details on time span matching.
        '''

        args = self._get_selection_args(obj, tmin, tmax, time, codes)
        nuts = sorted(
            self.iter_nuts('channel', *args), key=lambda nut: nut.dkey)
        self._check_duplicates(nuts)
        return [self.get_content(nut) for nut in nuts]

    @filldocs
    def get_sensors(
            self, obj=None, tmin=None, tmax=None, time=None, codes=None):

        '''
        Get sensors matching given constraints.

        %(query_args)s

        :returns:
            List of :py:class:`~pyrocko.squirrel.model.Sensor` objects.

        See :py:meth:`iter_nuts` for details on time span matching.
        '''

        tmin, tmax, codes = self._get_selection_args(
            obj, tmin, tmax, time, codes)

        if codes is not None:
            if isinstance(codes, str):
                codes = codes.split('.')
            codes = tuple(codes_inflate(codes))
            if codes[4] != '*':
                codes = codes[:4] + (codes[4][:-1] + '?',) + codes[5:]

        nuts = sorted(
            self.iter_nuts(
                'channel', tmin, tmax, codes), key=lambda nut: nut.dkey)
        self._check_duplicates(nuts)
        return model.Sensor.from_channels(
            self.get_content(nut) for nut in nuts)

    @filldocs
    def get_responses(
            self, obj=None, tmin=None, tmax=None, time=None, codes=None):

        '''
        Get instrument responses matching given constraints.

        %(query_args)s

        :returns:
            List of :py:class:`~pyrocko.squirrel.model.Response` objects.

        See :py:meth:`iter_nuts` for details on time span matching.
        '''

        args = self._get_selection_args(obj, tmin, tmax, time, codes)
        nuts = sorted(
            self.iter_nuts('response', *args), key=lambda nut: nut.dkey)
        self._check_duplicates(nuts)
        return [self.get_content(nut) for nut in nuts]

    @filldocs
    def get_response(
            self, obj=None, tmin=None, tmax=None, time=None, codes=None):

        '''
        Get instrument response matching given constraints.

        %(query_args)s

        :returns:
            :py:class:`~pyrocko.squirrel.model.Response` object.

        Same as :py:meth:`get_responses` but returning exactly one response.
        Raises :py:exc:`~pyrocko.squirrel.error.NotAvailable` if zero or more
        than one is available.

        See :py:meth:`iter_nuts` for details on time span matching.
        '''

        responses = self.get_responses(obj, tmin, tmax, time, codes)
        if len(responses) == 0:
            raise error.NotAvailable(
                'No instrument response available.')
        elif len(responses) > 1:
            raise error.NotAvailable(
                'Multiple instrument responses matching given constraints.')

        return responses[0]

    @filldocs
    def get_events(
            self, obj=None, tmin=None, tmax=None, time=None, codes=None):

        '''
        Get events matching given constraints.

        %(query_args)s

        :returns:
            List of :py:class:`~pyrocko.model.event.Event` objects.

        See :py:meth:`iter_nuts` for details on time span matching.
        '''

        args = self._get_selection_args(obj, tmin, tmax, time, codes)
        nuts = sorted(
            self.iter_nuts('event', *args), key=lambda nut: nut.dkey)
        self._check_duplicates(nuts)
        return [self.get_content(nut) for nut in nuts]

    def _redeem_promises(self, *args):

        tmin, tmax, _ = args

        waveforms = list(self.iter_nuts('waveform', *args))
        promises = list(self.iter_nuts('waveform_promise', *args))

        codes_to_avail = defaultdict(list)
        for nut in waveforms:
            codes_to_avail[nut.codes].append((nut.tmin, nut.tmax+nut.deltat))

        def tts(x):
            if isinstance(x, tuple):
                return tuple(tts(e) for e in x)
            elif isinstance(x, list):
                return list(tts(e) for e in x)
            else:
                return util.time_to_str(x)

        orders = []
        for promise in promises:
            waveforms_avail = codes_to_avail[promise.codes]
            for block_tmin, block_tmax in blocks(
                    max(tmin, promise.tmin),
                    min(tmax, promise.tmax),
                    promise.deltat):

                orders.append(
                    WaveformOrder(
                        source_id=promise.file_path,
                        codes=tuple(promise.codes.split(separator)),
                        tmin=block_tmin,
                        tmax=block_tmax,
                        deltat=promise.deltat,
                        gaps=gaps(waveforms_avail, block_tmin, block_tmax)))

        orders_noop, orders = lpick(lambda order: order.gaps, orders)

        order_keys_noop = set(order_key(order) for order in orders_noop)
        if len(order_keys_noop) != 0 or len(orders_noop) != 0:
            logger.info(
                'Waveform orders already satisified with cached/local data: '
                '%i (%i)' % (len(order_keys_noop), len(orders_noop)))

        source_ids = []
        sources = {}
        for source in self._sources:
            if isinstance(source, fdsn.FDSNSource):
                source_ids.append(source._source_id)
                sources[source._source_id] = source

        source_priority = dict(
            (source_id, i) for (i, source_id) in enumerate(source_ids))

        order_groups = defaultdict(list)
        for order in orders:
            order_groups[order_key(order)].append(order)

        for k, order_group in order_groups.items():
            order_group.sort(
                key=lambda order: source_priority[order.source_id])

        n_order_groups = len(order_groups)

        if len(order_groups) != 0 or len(orders) != 0:
            logger.info(
                'Waveform orders standing for download: %i (%i)'
                % (len(order_groups), len(orders)))

            task = make_task('Waveform orders processed', n_order_groups)
        else:
            task = None

        def split_promise(order):
            self._split_nuts(
                'waveform_promise',
                order.tmin, order.tmax,
                codes=order.codes,
                path=order.source_id)

        def release_order_group(order):
            okey = order_key(order)
            for followup in order_groups[okey]:
                split_promise(followup)

            del order_groups[okey]

            if task:
                task.update(n_order_groups - len(order_groups))

        def noop(order):
            pass

        def success(order):
            release_order_group(order)
            split_promise(order)

        def batch_add(paths):
            self.add(paths)

        calls = queue.Queue()

        def enqueue(f):
            def wrapper(*args):
                calls.put((f, args))

            return wrapper

        for order in orders_noop:
            split_promise(order)

        while order_groups:

            orders_now = []
            empty = []
            for k, order_group in order_groups.items():
                try:
                    orders_now.append(order_group.pop(0))
                except IndexError:
                    empty.append(k)

            for k in empty:
                del order_groups[k]

            by_source_id = defaultdict(list)
            for order in orders_now:
                by_source_id[order.source_id].append(order)

            threads = []
            for source_id in by_source_id:
                def download():
                    try:
                        sources[source_id].download_waveforms(
                            by_source_id[source_id],
                            success=enqueue(success),
                            error_permanent=enqueue(split_promise),
                            error_temporary=noop,
                            batch_add=enqueue(batch_add))

                    finally:
                        calls.put(None)

                thread = threading.Thread(target=download)
                thread.start()
                threads.append(thread)

            ndone = 0
            while ndone < len(threads):
                ret = calls.get()
                if ret is None:
                    ndone += 1
                else:
                    ret[0](*ret[1])

            for thread in threads:
                thread.join()

            if task:
                task.update(n_order_groups - len(order_groups))

        if task:
            task.done()

    @filldocs
    def get_waveform_nuts(
            self, obj=None, tmin=None, tmax=None, time=None, codes=None):

        '''
        Get waveform content entities matching given constraints.

        %(query_args)s

        Like :py:meth:`get_nuts` with ``kind='waveform'`` but additionally
        resolves matching waveform promises (downloads waveforms from remote
        sources).

        See :py:meth:`iter_nuts` for details on time span matching.
        '''

        args = self._get_selection_args(obj, tmin, tmax, time, codes)
        self._redeem_promises(*args)
        return sorted(
            self.iter_nuts('waveform', *args), key=lambda nut: nut.dkey)

    @filldocs
    def get_waveforms(
            self, obj=None, tmin=None, tmax=None, time=None, codes=None,
            uncut=False, want_incomplete=True, degap=True, maxgap=5,
            maxlap=None, snap=None, include_last=False, load_data=True,
            accessor_id='default', operator_params=None):

        '''
        Get waveforms matching given constraints.

        %(query_args)s

        :param uncut:
            Set to ``True``, to disable cutting traces to [``tmin``, ``tmax``]
            and to disable degapping/deoverlapping. Returns untouched traces as
            they are read from file segment. File segments are always read in
            their entirety.
        :type uncut:
            bool

        :param want_incomplete:
            If ``True``, gappy/incomplete traces are included in the result.
        :type want_incomplete:
            bool

        :param degap:
            If ``True``, connect traces and remove gaps and overlaps.
        :type degap:
            bool

        :param maxgap:
            Maximum gap size in samples which is filled with interpolated
            samples when ``degap`` is ``True``.
        :type maxgap:
            int

        :param maxlap:
            Maximum overlap size in samples which is removed when ``degap`` is
            ``True``
        :type maxlap:
            int

        :param snap:
            Rounding functions used when computing sample index from time
            instance, for trace start and trace end, respectively. By default,
            ``(round, round)`` is used.
        :type snap:
             tuple of 2 callables

        :param include_last:
            If ``True``, add one more sample to the returned traces (the sample
            which would be the first sample of a query with ``tmin`` set to the
            current value of ``tmax``).
        :type include_last:
            bool

        :param load_data:
            If ``True``, waveform data samples are read from files (or cache).
            If ``False``, meta-information-only traces are returned (dummy
            traces with no data samples).
        :type load_data:
            bool

        :param accessor_id:
            Name of consumer on who's behalf data is accessed. Used in cache
            management (see :py:mod:`~pyrocko.squirrel.cache`). Used as a key
            to distinguish different points of extraction for the decision of
            when to release cached waveform data. Should be used when data is
            alternately extracted from more than one region / selection.
        :type accessor_id:
            str

        See :py:meth:`iter_nuts` for details on time span matching.

        Loaded data is kept in memory (at least) until
        :py:meth:`clear_accessor` has been called or
        :py:meth:`advance_accessor` has been called two consecutive times
        without data being accessed between the two calls (by this accessor).
        Data may still be further kept in the memory cache if held alive by
        consumers with a different ``accessor_id``.
        '''

        tmin, tmax, codes = self._get_selection_args(
            obj, tmin, tmax, time, codes)

        self_tmin, self_tmax = self.get_time_span(
            ['waveform', 'waveform_promise'])

        if None in (self_tmin, self_tmax):
            logger.warning(
                'No waveforms available.')
            return []

        tmin = tmin if tmin is not None else self_tmin
        tmax = tmax if tmax is not None else self_tmax

        if codes is not None:
            operator = self.get_operator(codes)
            if operator is not None:
                return operator.get_waveforms(
                    self, codes,
                    tmin=tmin, tmax=tmax,
                    uncut=uncut, want_incomplete=want_incomplete, degap=degap,
                    maxgap=maxgap, maxlap=maxlap, snap=snap,
                    include_last=include_last, load_data=load_data,
                    accessor_id=accessor_id, params=operator_params)

        nuts = self.get_waveform_nuts(obj, tmin, tmax, time, codes)

        if load_data:
            traces = [
                self.get_content(nut, 'waveform', accessor_id) for nut in nuts]

        else:
            traces = [
                trace.Trace(**nut.trace_kwargs) for nut in nuts]

        if uncut:
            return traces

        if snap is None:
            snap = (round, round)

        chopped = []
        for tr in traces:
            if not load_data and tr.ydata is not None:
                tr = tr.copy(data=False)
                tr.ydata = None

            try:
                chopped.append(tr.chop(
                    tmin, tmax,
                    inplace=False,
                    snap=snap,
                    include_last=include_last))

            except trace.NoData:
                pass

        processed = self._process_chopped(
            chopped, degap, maxgap, maxlap, want_incomplete, tmin, tmax)

        return processed

    @filldocs
    def chopper_waveforms(
            self, obj=None, tmin=None, tmax=None, time=None, codes=None,
            tinc=None, tpad=0.,
            want_incomplete=True, snap_window=False,
            degap=True, maxgap=5, maxlap=None,
            snap=None, include_last=False, load_data=True,
            accessor_id=None, clear_accessor=True, operator_params=None):

        '''
        Iterate window-wise over waveform archive.

        %(query_args)s

        :param tinc:
            Time increment (window shift time) (default uses ``tmax-tmin``)
        :type tinc:
            timestamp

        :param tpad:
            Padding time appended on either side of the data window (window
            overlap is ``2*tpad``).
        :type tpad:
            timestamp

        :param want_incomplete:
            If ``True``, gappy/incomplete traces are included in the result.
        :type want_incomplete:
            bool

        :param snap_window:
            If ``True``, start time windows at multiples of tinc with respect
            to system time zero.

        :param degap:
            If ``True``, connect traces and remove gaps and overlaps.
        :type degap:
            bool

        :param maxgap:
            Maximum gap size in samples which is filled with interpolated
            samples when ``degap`` is ``True``.
        :type maxgap:
            int

        :param maxlap:
            Maximum overlap size in samples which is removed when ``degap`` is
            ``True``
        :type maxlap:
            int

        :param snap:
            Rounding functions used when computing sample index from time
            instance, for trace start and trace end, respectively. By default,
            ``(round, round)`` is used.
        :type snap:
             tuple of 2 callables

        :param include_last:
            If ``True``, add one more sample to the returned traces (the sample
            which would be the first sample of a query with ``tmin`` set to the
            current value of ``tmax``).
        :type include_last:
            bool

        :param load_data:
            If ``True``, waveform data samples are read from files (or cache).
            If ``False``, meta-information-only traces are returned (dummy
            traces with no data samples).
        :type load_data:
            bool

        :param accessor_id:
            Name of consumer on who's behalf data is accessed. Used in cache
            management (see :py:mod:`~pyrocko.squirrel.cache`). Used as a key
            to distinguish different points of extraction for the decision of
            when to release cached waveform data. Should be used when data is
            alternately extracted from more than one region / selection.
        :type accessor_id:
            str

        :param clear_accessor:
            If ``True`` (default), :py:meth:`clear_accessor` is called when the
            chopper finishes. Set to ``False`` to keep loaded waveforms in
            memory when the generator returns.

        :yields:
            A list of :py:class:`~pyrocko.trace.Trace` objects for every
            extracted time window.

        See :py:meth:`iter_nuts` for details on time span matching.
        '''

        tmin, tmax, codes = self._get_selection_args(
            obj, tmin, tmax, time, codes)

        self_tmin, self_tmax = self.get_time_span(
            ['waveform', 'waveform_promise'])

        if None in (self_tmin, self_tmax):
            logger.warning(
                'Content has undefined time span. No waveforms and no '
                'waveform promises?')
            return

        if snap_window and tinc is not None:
            tmin = tmin if tmin is not None else self_tmin
            tmax = tmax if tmax is not None else self_tmax
            tmin = math.floor(tmin / tinc) * tinc
            tmax = math.ceil(tmax / tinc) * tinc
        else:
            tmin = tmin if tmin is not None else self_tmin + tpad
            tmax = tmax if tmax is not None else self_tmax - tpad

        tinc = tinc if tinc is not None else tmax - tmin

        try:
            if accessor_id is None:
                accessor_id = 'chopper%i' % self._n_choppers_active

            self._n_choppers_active += 1

            eps = tinc * 1e-6
            if tinc != 0.0:
                nwin = int(((tmax - eps) - tmin) / tinc) + 1
            else:
                nwin = 1

            for iwin in range(nwin):
                wmin, wmax = tmin+iwin*tinc, min(tmin+(iwin+1)*tinc, tmax)
                chopped = []
                wmin, wmax = tmin+iwin*tinc, min(tmin+(iwin+1)*tinc, tmax)
                eps = tinc*1e-6
                if wmin >= tmax-eps:
                    break

                chopped = self.get_waveforms(
                    tmin=wmin-tpad,
                    tmax=wmax+tpad,
                    codes=codes,
                    snap=snap,
                    include_last=include_last,
                    load_data=load_data,
                    want_incomplete=want_incomplete,
                    degap=degap,
                    maxgap=maxgap,
                    maxlap=maxlap,
                    accessor_id=accessor_id,
                    operator_params=operator_params)

                self.advance_accessor(accessor_id)

                yield Batch(
                    tmin=wmin,
                    tmax=wmax,
                    i=iwin,
                    n=nwin,
                    traces=chopped)

                iwin += 1

        finally:
            self._n_choppers_active -= 1
            if clear_accessor:
                self.clear_accessor(accessor_id, 'waveform')

    def _process_chopped(
            self, chopped, degap, maxgap, maxlap, want_incomplete, tmin, tmax):

        chopped.sort(key=lambda a: a.full_id)
        if degap:
            chopped = trace.degapper(chopped, maxgap=maxgap, maxlap=maxlap)

        if not want_incomplete:
            chopped_weeded = []
            for tr in chopped:
                emin = tr.tmin - tmin
                emax = tr.tmax + tr.deltat - tmax
                if (abs(emin) <= 0.5*tr.deltat and abs(emax) <= 0.5*tr.deltat):
                    chopped_weeded.append(tr)

                elif degap:
                    if (0. < emin <= 5. * tr.deltat
                            and -5. * tr.deltat <= emax < 0.):

                        tr.extend(tmin, tmax-tr.deltat, fillmethod='repeat')
                        chopped_weeded.append(tr)

            chopped = chopped_weeded

        return chopped

    def _get_pyrocko_stations(
            self, obj=None, tmin=None, tmax=None, time=None, codes=None):

        from pyrocko import model as pmodel

        by_nsl = defaultdict(lambda: (list(), list()))
        for station in self.get_stations(obj, tmin, tmax, time, codes):
            sargs = station._get_pyrocko_station_args()
            nsl = sargs[1:4]
            by_nsl[nsl][0].append(sargs)

        for channel in self.get_channels(obj, tmin, tmax, time, codes):
            sargs = channel._get_pyrocko_station_args()
            nsl = sargs[1:4]
            sargs_list, channels_list = by_nsl[nsl]
            sargs_list.append(sargs)
            channels_list.append(channel)

        pstations = []
        nsls = list(by_nsl.keys())
        nsls.sort()
        for nsl in nsls:
            sargs_list, channels_list = by_nsl[nsl]
            sargs = util.consistency_merge(sargs_list)

            by_c = defaultdict(list)
            for ch in channels_list:
                by_c[ch.channel].append(ch._get_pyrocko_channel_args())

            chas = list(by_c.keys())
            chas.sort()
            pchannels = []
            for cha in chas:
                list_of_cargs = by_c[cha]
                cargs = util.consistency_merge(list_of_cargs)
                pchannels.append(pmodel.Channel(
                    name=cargs[0],
                    azimuth=cargs[1],
                    dip=cargs[2]))

            pstations.append(pmodel.Station(
                network=sargs[0],
                station=sargs[1],
                location=sargs[2],
                lat=sargs[3],
                lon=sargs[4],
                elevation=sargs[5],
                depth=sargs[6] or 0.0,
                channels=pchannels))

        return pstations

    @property
    def pile(self):

        '''
        Emulates the older :py:class:`pyrocko.pile.Pile` interface.

        This property exposes a :py:class:`pyrocko.squirrel.pile.Pile` object,
        which emulates most of the older :py:class:`pyrocko.pile.Pile` methods
        but uses the fluffy power of the Squirrel under the hood.

        This interface can be used as a drop-in replacement for piles which are
        used in existing scripts and programs for efficient waveform data
        access. The Squirrel-based pile scales better for large datasets. Newer
        scripts should use Squirrel's native methods to avoid the emulation
        overhead.
        '''
        from . import pile

        if self._pile is None:
            self._pile = pile.Pile(self)

        return self._pile

    def snuffle(self):
        '''
        Look at dataset in Snuffler.
        '''
        self.pile.snuffle()

    def _gather_codes_keys(self, kind, gather, selector):
        return set(
            gather(codes)
            for codes in self.iter_codes(kind)
            if selector is None or selector(codes))

    def __str__(self):
        return str(self.get_stats())

    def get_coverage(
            self, kind, tmin=None, tmax=None, codes_list=None, limit=None,
            return_raw=True):

        '''
        Get coverage information.

        Get information about strips of gapless data coverage.

        :param kind:
            Content kind to be queried.
        :type kind:
            str

        :param tmin:
            Start time of query interval.
        :type tmin:
            timestamp

        :param tmax:
            End time of query interval.
        :type tmax:
            timestamp

        :param codes_list:
            List of code patterns to query. If not given or empty, an empty
            list is returned.
        :type codes_list:
            :py:class:`list` of :py:class:`tuple` of :py:class:`str`

        :param limit:
            Limit query to return only up to a given maximum number of entries
            per matching channel (without setting this option, very gappy data
            could cause the query to execute for a very long time).
        :type limit:
            int

        :returns:
            List of entries of the form ``(pattern, codes, deltat, tmin, tmax,
            data)`` where ``pattern`` is the request code pattern which
            yielded this entry, ``codes`` are the matching channel codes,
            ``tmin`` and ``tmax`` are the global min and max times for which
            data for this channel is available, regardless of any time
            restrictions in the query. ``data`` is a list with (up to
            ``limit``) change-points of the form ``(time, count)`` where a
            ``count`` of zero indicates a data gap, a value of 1 normal data
            coverage and higher values indicate duplicate/redundant data.
        '''

        tmin_seconds, tmin_offset = model.tsplit(tmin)
        tmax_seconds, tmax_offset = model.tsplit(tmax)
        kind_id = to_kind_id(kind)

        if codes_list is None:
            codes_list = self.get_codes(kind=kind)

        kdata_all = []
        for pattern in codes_list:
            kdata = self.glob_codes(kind, [pattern])
            for row in kdata:
                row[0:0] = [pattern]

            kdata_all.extend(kdata)

        kind_codes_ids = [x[1] for x in kdata_all]

        counts_at_tmin = {}
        if tmin is not None:
            for nut in self.iter_nuts(
                    kind, tmin, tmin, kind_codes_ids=kind_codes_ids):

                k = nut.codes, nut.deltat
                if k not in counts_at_tmin:
                    counts_at_tmin[k] = 0

                counts_at_tmin[k] += 1

        coverage = []
        for pattern, kind_codes_id, codes, deltat in kdata_all:
            entry = [pattern, codes, deltat, None, None, []]
            for i, order in [(0, 'ASC'), (1, 'DESC')]:
                sql = self._sql('''
                    SELECT
                        time_seconds,
                        time_offset
                    FROM %(db)s.%(coverage)s
                    WHERE
                        kind_codes_id == ?
                    ORDER BY
                        kind_codes_id ''' + order + ''',
                        time_seconds ''' + order + ''',
                        time_offset ''' + order + '''
                    LIMIT 1
                ''')

                for row in self._conn.execute(sql, [kind_codes_id]):
                    entry[3+i] = model.tjoin(row[0], row[1])

            if None in entry[3:5]:
                continue

            args = [kind_codes_id]

            sql_time = ''
            if tmin is not None:
                # intentionally < because (== tmin) is queried from nuts
                sql_time += ' AND ( ? < time_seconds ' \
                    'OR ( ? == time_seconds AND ? < time_offset ) ) '
                args.extend([tmin_seconds, tmin_seconds, tmin_offset])

            if tmax is not None:
                sql_time += ' AND ( time_seconds < ? ' \
                    'OR ( ? == time_seconds AND time_offset <= ? ) ) '
                args.extend([tmax_seconds, tmax_seconds, tmax_offset])

            sql_limit = ''
            if limit is not None:
                sql_limit = ' LIMIT ?'
                args.append(limit)

            sql = self._sql('''
                SELECT
                    time_seconds,
                    time_offset,
                    step
                FROM %(db)s.%(coverage)s
                WHERE
                    kind_codes_id == ?
                    ''' + sql_time + '''
                ORDER BY
                    kind_codes_id,
                    time_seconds,
                    time_offset
            ''' + sql_limit)

            rows = list(self._conn.execute(sql, args))

            if limit is not None and len(rows) == limit:
                entry[-1] = None
            else:
                counts = counts_at_tmin.get((codes, deltat), 0)
                tlast = None
                if tmin is not None:
                    entry[-1].append((tmin, counts))
                    tlast = tmin

                for row in rows:
                    t = model.tjoin(row[0], row[1])
                    counts += row[2]
                    entry[-1].append((t, counts))
                    tlast = t

                if tmax is not None and (tlast is None or tlast != tmax):
                    entry[-1].append((tmax, counts))

            coverage.append(entry)

        if return_raw:
            return coverage
        else:
            return [model.Coverage.from_values(
                entry + [kind_id]) for entry in coverage]

    def add_operator(self, op):
        self._operators.append(op)

    def update_operator_mappings(self):
        available = [
            separator.join(codes)
            for codes in self.get_codes(kind=('channel'))]

        for operator in self._operators:
            operator.update_mappings(available, self._operator_registry)

    def iter_operator_mappings(self):
        for operator in self._operators:
            for in_codes, out_codes in operator.iter_mappings():
                yield operator, in_codes, out_codes

    def get_operator_mappings(self):
        return list(self.iter_operator_mappings())

    def get_operator(self, codes):
        if isinstance(codes, tuple):
            codes = separator.join(codes)
        try:
            return self._operator_registry[codes][0]
        except KeyError:
            return None

    def get_operator_group(self, codes):
        if isinstance(codes, tuple):
            codes = separator.join(codes)
        try:
            return self._operator_registry[codes]
        except KeyError:
            return None, (None, None, None)

    def iter_operator_codes(self):
        for _, _, out_codes in self.iter_operator_mappings():
            for codes in out_codes:
                yield tuple(codes.split(separator))

    def get_operator_codes(self):
        return list(self.iter_operator_codes())

    def print_tables(self, table_names=None, stream=None):
        '''
        Dump raw database tables in textual form (for debugging purposes).

        :param table_names:
            Names of tables to be dumped or ``None`` to dump all.
        :type table_names:
            :py:class:`list` of :py:class:`str`

        :param stream:
            Open file or ``None`` to dump to standard output.
        '''

        if stream is None:
            stream = sys.stdout

        if isinstance(table_names, str):
            table_names = [table_names]

        if table_names is None:
            table_names = [
                'selection_file_states',
                'selection_nuts',
                'selection_kind_codes_count',
                'files', 'nuts', 'kind_codes', 'kind_codes_count']

        m = {
            'selection_file_states': '%(db)s.%(file_states)s',
            'selection_nuts': '%(db)s.%(nuts)s',
            'selection_kind_codes_count': '%(db)s.%(kind_codes_count)s',
            'files': 'files',
            'nuts': 'nuts',
            'kind_codes': 'kind_codes',
            'kind_codes_count': 'kind_codes_count'}

        for table_name in table_names:
            self._database.print_table(
                m[table_name] % self._names, stream=stream)


class SquirrelStats(Object):
    '''
    Container to hold statistics about contents available from a Squirrel.

    See also :py:meth:`Squirrel.get_stats`.
    '''

    nfiles = Int.T(
        help='Number of files in selection.')
    nnuts = Int.T(
        help='Number of index nuts in selection.')
    codes = List.T(
        Tuple.T(content_t=String.T()),
        help='Available code sequences in selection, e.g. '
             '(agency, network, station, location) for stations nuts.')
    kinds = List.T(
        String.T(),
        help='Available content types in selection.')
    total_size = Int.T(
        help='Aggregated file size of files is selection.')
    counts = Dict.T(
        String.T(), Dict.T(Tuple.T(content_t=String.T()), Int.T()),
        help='Breakdown of how many nuts of any content type and code '
             'sequence are available in selection, ``counts[kind][codes]``.')
    time_spans = Dict.T(
        String.T(), Tuple.T(content_t=Timestamp.T()),
        help='Time spans by content type.')
    sources = List.T(
        String.T(),
        help='Descriptions of attached sources.')
    operators = List.T(
        String.T(),
        help='Descriptions of attached operators.')

    def __str__(self):
        kind_counts = dict(
            (kind, sum(self.counts[kind].values())) for kind in self.kinds)

        scodes = model.codes_to_str_abbreviated(self.codes)

        ssources = '<none>' if not self.sources else '\n' + '\n'.join(
            '  ' + s for s in self.sources)

        soperators = '<none>' if not self.operators else '\n' + '\n'.join(
            '  ' + s for s in self.operators)

        def stime(t):
            return util.tts(t) if t is not None and t not in (
                model.g_tmin, model.g_tmax) else '<none>'

        def stable(rows):
            ns = [max(len(w) for w in col) for col in zip(*rows)]
            return '\n'.join(
                ' '.join(w.ljust(n) for n, w in zip(ns, row))
                for row in rows)

        def indent(s):
            return '\n'.join('  '+line for line in s.splitlines())

        stspans = '<none>' if not self.kinds else '\n' + indent(stable([(
            kind + ':',
            str(kind_counts[kind]),
            stime(self.time_spans[kind][0]),
            '-',
            stime(self.time_spans[kind][1])) for kind in sorted(self.kinds)]))

        s = '''
Number of files:               %i
Total size of known files:     %s
Number of index nuts:          %i
Available content kinds:       %s
Available codes:               %s
Sources:                       %s
Operators:                     %s''' % (
            self.nfiles,
            util.human_bytesize(self.total_size),
            self.nnuts,
            stspans, scodes, ssources, soperators)

        return s.lstrip()


__all__ = [
    'Squirrel',
    'SquirrelStats',
]
