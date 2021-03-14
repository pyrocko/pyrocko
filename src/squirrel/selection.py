#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

import os
import re
import threading
import logging

from pyrocko import util
from pyrocko.io_common import FileLoadError
from pyrocko.progress import progress

from . import error, io, model
from .database import Database, get_database, execute_get1

logger = logging.getLogger('psq.selection')

g_icount = 0
g_lock = threading.Lock()

re_persistent_name = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]{0,64}$')


def make_unique_name():
    with g_lock:
        global g_icount
        name = '%i_%i' % (os.getpid(), g_icount)
        g_icount += 1

    return name


def make_task(*args, **kwargs):
    kwargs['logger'] = logger
    return progress.task(*args, **kwargs)


doc_snippets = dict(
    query_args='''
        :param obj:
            Object providing ``tmin``, ``tmax`` and ``codes`` to be used to
            constrain the query. Direct arguments override those from ``obj``.
        :type obj:
            any object with attributes ``tmin``, ``tmax`` and ``codes``

        :param tmin:
            Start time of query interval.
        :type tmin:
            timestamp

        :param tmax:
            End time of query interval.
        :type tmax:
            timestamp

        :param time:
            Time instant to query. Equivalent to setting ``tmin`` and ``tmax``
            to the same value.
        :type time:
            timestamp

        :param codes:
            Pattern of content codes to query.
        :type codes:
            :py:class:`tuple` of :py:class:`str`
''',
    file_formats=', '.join(
        "``'%s'``" % fmt for fmt in io.supported_formats()))


def filldocs(meth):
    meth.__doc__ %= doc_snippets
    return meth


class GeneratorWithLen(object):

    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen


class Selection(object):

    '''
    Database backed file selection (base class for
    :py:class:`~pyrocko.squirrel.base.Squirrel`).

    :param database:
        Database instance or file path to database.
    :type database:
        :py:class:`~pyrocko.squirrel.database.Database` or :py:class:`str`

    :param persistent:
        If given a name, create a persistent selection.
    :type persistent:
        :py:class:`str`

    A selection in this context represents the list of files available to the
    application. Instead of using :py:class:`Selection` directly, user
    applications should usually use its subclass
    :py:class:`~pyrocko.squirrel.base.Squirrel` which adds content indices to
    the selection and provides high level data querying.

    By default, a temporary table in the database is created to hold the names
    of the files in the selection. This table is only visible inside the
    application which created it. If a name is given to ``persistent``, a named
    selection is created, which is visible also in other applications using the
    same database.

    Besides the filename references, desired content kind masks and file format
    indications are stored in the selection's database table to make the user
    choice regarding these options persistent on a per-file basis. Book-keeping
    on whether files are unknown, known or if modification checks are forced is
    handled in the selection's file-state table.

    Paths of files can be added to the selection using the :py:meth:`add`
    method and removed with :py:meth:`remove`. :py:meth:`undig_grouped` can be
    used to iterate over all content known to the selection.
    '''

    def __init__(self, database, persistent=None):
        self._conn = None

        if not isinstance(database, Database):
            database = get_database(database)

        if persistent is not None:
            assert isinstance(persistent, str)
            if not re_persistent_name.match(persistent):
                raise error.SquirrelError(
                    'invalid persistent selection name: %s' % persistent)

            self.name = 'psel_' + persistent
        else:
            self.name = 'sel_' + make_unique_name()

        self._persistent = persistent
        self._database = database
        self._conn = self._database.get_connection()
        self._sources = []
        self._is_new = True

        if persistent is not None:
            self._is_new = 1 == self._conn.execute(
                '''
                    INSERT OR IGNORE INTO persistent VALUES (?)
                ''', (persistent,)).rowcount

        self._names = {
            'db': 'main' if self._persistent else 'temp',
            'file_states': self.name + '_file_states',
            'bulkinsert': self.name + '_bulkinsert'}

        self._conn.execute(self._register_table(self._sql(
            '''
                CREATE TABLE IF NOT EXISTS %(db)s.%(file_states)s (
                    file_id integer PRIMARY KEY,
                    file_state integer,
                    kind_mask integer,
                    format text)
            ''')))

        self._conn.execute(self._sql(
            '''
                CREATE INDEX
                IF NOT EXISTS %(db)s.%(file_states)s_index_file_state
                ON %(file_states)s (file_state)
            '''))

    def __del__(self):
        if hasattr(self, '_conn') and self._conn:
            if not self._persistent:
                self._delete()
            else:
                self._conn.commit()

    def _register_table(self, s):
        return self._database._register_table(s)

    def _sql(self, s):
        return s % self._names

    def is_new(self):
        '''
        Is this a new selection?

        Always ``True`` for non-persistent selections. Only ``False`` for
        a persistent selection which already existed in the database when the
        it was initialized.
        '''
        return self._is_new

    def get_database(self):
        '''
        Get the database to which this selection belongs.

        :returns: :py:class:`~pyrocko.squirrel.database.Database` object
        '''
        return self._database

    def _delete(self):
        '''
        Destroy the tables assoctiated with this selection.
        '''
        self._conn.execute(self._sql(
            'DROP TABLE %(db)s.%(file_states)s'))

        if self._persistent:
            self._conn.execute(
                '''
                    DELETE FROM persistent WHERE name == ?
                ''', (self.name[5:],))

        self._conn.commit()
        self._conn = None

    def delete(self):
        self._delete()

    @filldocs
    def add(
            self,
            paths,
            kind_mask=model.g_kind_mask_all,
            format='detect'):

        '''
        Add files to the selection.

        :param paths:
            Paths to files to be added to the selection.
        :type paths:
            iterator yielding :py:class:`str` objects

        :param kind_mask:
            Content kinds to be added to the selection.
        :type kind_mask:
            :py:class:`int` (bit mask)

        :param format:
            File format identifier or ``'detect'`` to enable auto-detection
            (available: %(file_formats)s).
        :type format:
            str
        '''

        if isinstance(paths, str):
            paths = [paths]

        task = make_task('Gathering file names')
        paths = task(paths)
        paths = util.short_to_list(200, paths)

        if isinstance(paths, list) and len(paths) <= 200:

            # short non-iterator paths: can do without temp table

            self._conn.executemany(
                '''
                    INSERT OR IGNORE INTO files
                    VALUES (NULL, ?, NULL, NULL, NULL)
                ''', ((x,) for x in paths))

            task = make_task('Preparing database', 3)
            task.update(0, condition='pruning stale information')
            self._conn.executemany(self._sql(
                '''
                    DELETE FROM %(db)s.%(file_states)s
                    WHERE file_id IN (
                        SELECT files.file_id
                            FROM files
                            WHERE files.path == ? )
                        AND ( kind_mask != ? OR format != ? )
                '''), (
                    (path, kind_mask, format) for path in paths))

            task.update(1, condition='adding file names to selection')
            self._conn.executemany(self._sql(
                '''
                    INSERT OR IGNORE INTO %(db)s.%(file_states)s
                    SELECT files.file_id, 0, ?, ?
                    FROM files
                    WHERE files.path = ?
                '''), ((kind_mask, format, path) for path in paths))

            task.update(2, condition='updating file states')
            self._conn.executemany(self._sql(
                '''
                    UPDATE %(db)s.%(file_states)s
                    SET file_state = 1
                    WHERE file_id IN (
                        SELECT files.file_id
                            FROM files
                            WHERE files.path == ? )
                        AND file_state != 0
                '''), ((path,) for path in paths))

            task.update(3)
            task.done()

        else:

            self._conn.execute(self._sql(
                '''
                    CREATE TEMP TABLE temp.%(bulkinsert)s
                    (path text)
                '''))

            self._conn.executemany(self._sql(
                'INSERT INTO temp.%(bulkinsert)s VALUES (?)'),
                ((x,) for x in paths))

            task = make_task('Preparing database', 5)
            task.update(0, condition='adding file names to database')
            self._conn.execute(self._sql(
                '''
                    INSERT OR IGNORE INTO files
                    SELECT NULL, path, NULL, NULL, NULL
                    FROM temp.%(bulkinsert)s
                '''))

            task.update(1, condition='pruning stale information')
            self._conn.execute(self._sql(
                '''
                    DELETE FROM %(db)s.%(file_states)s
                    WHERE file_id IN (
                        SELECT files.file_id
                            FROM temp.%(bulkinsert)s
                            INNER JOIN files
                            ON temp.%(bulkinsert)s.path == files.path)
                        AND ( kind_mask != ? OR format != ? )
                '''), (kind_mask, format))

            task.update(2, condition='adding file names to selection')
            self._conn.execute(self._sql(
                '''
                    INSERT OR IGNORE INTO %(db)s.%(file_states)s
                    SELECT files.file_id, 0, ?, ?
                    FROM temp.%(bulkinsert)s
                    INNER JOIN files
                    ON temp.%(bulkinsert)s.path == files.path
                '''), (kind_mask, format))

            task.update(3, condition='updating file states')
            self._conn.execute(self._sql(
                '''
                    UPDATE %(db)s.%(file_states)s
                    SET file_state = 1
                    WHERE file_id IN (
                        SELECT files.file_id
                            FROM temp.%(bulkinsert)s
                            INNER JOIN files
                            ON temp.%(bulkinsert)s.path == files.path)
                        AND file_state != 0
                '''))

            task.update(4, condition='dropping temporary data')
            self._conn.execute(self._sql(
                'DROP TABLE temp.%(bulkinsert)s'))
            task.update(5)
            task.done()

    def remove(self, paths):
        '''
        Remove files from the selection.

        :param paths:
            Paths to files to be removed from the selection.
        :type paths:
            :py:class:`list` of :py:class:`str`
        '''
        if isinstance(paths, str):
            paths = [paths]

        self._conn.executemany(self._sql(
            '''
                DELETE FROM %(db)s.%(file_states)s
                WHERE %(db)s.%(file_states)s.file_id IN
                    (SELECT files.file_id
                     FROM files
                     WHERE files.path == ?)
            '''), ((path,) for path in paths))

    def iter_paths(self):
        '''
        Iterate over all file paths currently belonging to the selection.

        :yields: File paths.
        '''

        sql = self._sql('''
            SELECT
                files.path
            FROM %(db)s.%(file_states)s
            INNER JOIN files
            ON files.file_id = %(db)s.%(file_states)s.file_id
            ORDER BY %(db)s.%(file_states)s.file_id
        ''')

        for values in self._conn.execute(sql):
            yield values[0]

    def get_paths(self):
        '''
        Get all file paths currently belonging to the selection.

        :returns: List of file paths.
        '''
        return list(self.iter_paths())

    def _set_file_states_known(self):
        '''
        Set file states to "known" (2).
        '''
        self._conn.execute(self._sql(
            '''
                UPDATE %(db)s.%(file_states)s
                SET file_state = 2
                WHERE file_state < 2
            '''))

    def _set_file_states_force_check(self):
        '''
        Set file states to "request force check" (1).
        '''
        self._conn.execute(self._sql(
            '''
                UPDATE %(db)s.%(file_states)s
                SET file_state = 1
            '''))

    def undig_grouped(self, skip_unchanged=False):
        '''
        Get inventory of cached content for all files in the selection.

        :param skip_unchanged:
            If ``True`` only inventory of modified files is
            yielded (:py:meth:`flag_modified` must be called beforehand).
        :type skip_unchanged:
            bool

        This generator yields tuples ``((format, path), nuts)`` where ``path``
        is the path to the file, ``format`` is the format assignation or
        ``'detect'`` and ``nuts`` is a list of
        :py:class:`~pyrocko.squirrel.model.Nut` objects representing the
        contents of the file.
        '''

        if skip_unchanged:
            where = '''
                WHERE %(db)s.%(file_states)s.file_state == 0
            '''
        else:
            where = ''

        nfiles = execute_get1(self._conn, self._sql('''
            SELECT
                COUNT()
            FROM %(db)s.%(file_states)s
        ''' + where), ())[0]

        def gen():
            sql = self._sql('''
                SELECT
                    %(db)s.%(file_states)s.format,
                    files.path,
                    files.format,
                    files.mtime,
                    files.size,
                    nuts.file_segment,
                    nuts.file_element,
                    kind_codes.kind_id,
                    kind_codes.codes,
                    nuts.tmin_seconds,
                    nuts.tmin_offset,
                    nuts.tmax_seconds,
                    nuts.tmax_offset,
                    kind_codes.deltat
                FROM %(db)s.%(file_states)s
                LEFT OUTER JOIN files
                    ON %(db)s.%(file_states)s.file_id = files.file_id
                LEFT OUTER JOIN nuts
                    ON files.file_id = nuts.file_id
                LEFT OUTER JOIN kind_codes
                    ON nuts.kind_codes_id == kind_codes.kind_codes_id
            ''' + where + '''
                ORDER BY %(db)s.%(file_states)s.file_id
            ''')

            nuts = []
            format_path = None
            for values in self._conn.execute(sql):
                if format_path is not None and values[1] != format_path[1]:
                    yield format_path, nuts
                    nuts = []

                if values[2] is not None:
                    nuts.append(model.Nut(values_nocheck=values[1:]))

                format_path = values[:2]

            if format_path is not None:
                yield format_path, nuts

        return GeneratorWithLen(gen(), nfiles)

    def flag_modified(self, check=True):
        '''
        Mark files which have been modified.

        :param check:
            If ``True`` query modification times of known files on disk. If
            ``False``, only flag unknown files.
        :type check:
            bool

        Assumes file state is 0 for newly added files, 1 for files added again
        to the selection (forces check), or 2 for all others (no checking is
        done for those).

        Sets file state to 0 for unknown or modified files, 2 for known and not
        modified files.
        '''

        sql = self._sql('''
            UPDATE %(db)s.%(file_states)s
            SET file_state = 0
            WHERE (
                SELECT mtime
                FROM files
                WHERE files.file_id == %(db)s.%(file_states)s.file_id) IS NULL
                AND file_state == 1
        ''')

        self._conn.execute(sql)

        if not check:

            sql = self._sql('''
                UPDATE %(db)s.%(file_states)s
                SET file_state = 2
                WHERE file_state == 1
            ''')

            self._conn.execute(sql)

            return

        def iter_file_states():
            sql = self._sql('''
                SELECT
                    files.file_id,
                    files.path,
                    files.format,
                    files.mtime,
                    files.size
                FROM %(db)s.%(file_states)s
                INNER JOIN files
                    ON %(db)s.%(file_states)s.file_id == files.file_id
                WHERE %(db)s.%(file_states)s.file_state == 1
                ORDER BY %(db)s.%(file_states)s.file_id
            ''')

            for (file_id, path, fmt, mtime_db,
                    size_db) in self._conn.execute(sql):

                try:
                    mod = io.get_backend(fmt)
                    file_stats = mod.get_stats(path)

                except FileLoadError:
                    yield 0, file_id
                    continue
                except io.UnknownFormat:
                    continue

                if (mtime_db, size_db) != file_stats:
                    yield 0, file_id
                else:
                    yield 2, file_id

        # could better use callback function here...

        sql = self._sql('''
            UPDATE %(db)s.%(file_states)s
            SET file_state = ?
            WHERE file_id = ?
        ''')

        self._conn.executemany(sql, iter_file_states())


__all__ = [
    'Selection',
]
