# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

import time
import logging
from builtins import str as newstr

from pyrocko.io_common import FileLoadError
from pyrocko.progress import progress

from .backends import \
    mseed, sac, datacube, stationxml, textfiles, virtual, yaml, tdms_idas

from ..model import to_kind_ids

backend_modules = [
    mseed, sac, datacube, stationxml, textfiles, virtual, yaml, tdms_idas]


logger = logging.getLogger('psq.io')


def make_task(*args, **kwargs):
    kwargs['logger'] = logger
    return progress.task(*args, **kwargs)


def update_format_providers():
    '''Update global mapping from file format to io backend module.'''

    global g_format_providers
    g_format_providers = {}
    for mod in backend_modules:
        for format in mod.provided_formats():
            if format not in g_format_providers:
                g_format_providers[format] = []

            g_format_providers[format].append(mod)


g_format_providers = {}
update_format_providers()


class FormatDetectionFailed(FileLoadError):
    '''
    Exception raised when file format detection fails.
    '''

    def __init__(self, path):
        FileLoadError.__init__(
            self, 'format detection failed for file: %s' % path)


class UnknownFormat(Exception):
    '''
    Exception raised when user requests an unknown file format.
    '''

    def __init__(self, format):
        Exception.__init__(
            self, 'unknown format: %s' % format)


def get_backend(fmt):
    '''
    Get squirrel io backend module for a given file format.

    :param fmt:
        Format identifier.
    :type fmt:
        str
    '''

    try:
        return g_format_providers[fmt][0]
    except KeyError:
        raise UnknownFormat(fmt)


def detect_format(path):
    '''
    Determine file type from first 512 bytes.

    :param path:
        Path to file.
    :type path:
        str
    '''

    if path.startswith('virtual:'):
        return 'virtual'

    try:
        with open(path, 'rb') as f:
            data = f.read(512)

    except (OSError, IOError):
        raise FormatDetectionFailed(path)

    fmt = None
    for mod in backend_modules:
        fmt = mod.detect(data)
        if fmt is not None:
            return fmt

    raise FormatDetectionFailed(path)


def supported_formats():
    '''
    Get list of file formats supported by Squirrel.
    '''
    return sorted(g_format_providers.keys())


g_content_kinds = ['waveform', 'station', 'channel', 'response', 'event']


def supported_content_kinds():
    '''
    Get list of supported content kinds offered through Squirrel.
    '''
    return g_content_kinds


def iload(
        paths,
        segment=None,
        format='detect',
        database=None,
        check=True,
        commit=True,
        skip_unchanged=False,
        content=g_content_kinds):

    '''
    Iteratively load content or index/reindex meta-information from files.

    :param paths:
        Iterator yielding file names to load from or a Squirrel selection
        object providing the file names.
    :type paths:
        iterator yielding :py:class:`str` or
        :py:class:`~pyrocko.squirrel.selection.Selection`

    :param segment:
        File-specific segment identifier (can only be used when loading from a
        single file).
    :type segment:
        int

    :param format:
        File format identifier or ``'detect'`` for autodetection. When loading
        from a selection, per-file format assignation is taken from the hint in
        the selection and this flag is ignored.
    :type format:
        str

    :param database:
        Database to use for meta-information caching. When loading from a
        selection, this should be ``None`` and the database from the selection
        is used.
    :type database:
        :py:class:`~pyrocko.squirrel.database.Database`

    :param check:
        If ``True``, investigate modification time and file sizes of known
        files to debunk modified files (pessimistic mode), or ``False`` to
        deactivate checks (optimistic mode).
    :type check:
        bool

    :param commit:
        Flag, whether to commit updated information to the meta-information
        database.
    :type commit:
        bool

    :param skip_unchanged:
        If ``True``, only yield index nuts for new / modified files.
    :type skip_unchanged:
        bool

    :param content:
        Selection of content types to load.
    :type content:
        :py:class:`list` of :py:class:`str`

    This generator yields :py:class:`~pyrocko.squirrel.model.Nut` objects for
    individual pieces of information found when reading the given files. Such a
    nut may represent a waveform, a station, a channel, an event or other data
    type. The nut itself only contains the meta-information. The actual content
    information is attached to the nut if requested. All nut meta-information
    is stored in the squirrel meta-information database. If possible, this
    function avoids accessing the actual disk files and provides the requested
    information straight from the database. Modified files are recognized and
    reindexed as needed.
    '''

    from ..selection import Selection

    n_db = 0
    n_load = 0
    selection = None
    kind_ids = to_kind_ids(content)

    if isinstance(paths, (str, newstr)):
        paths = [paths]
    else:
        if segment is not None:
            raise TypeError(
                'iload: segment argument can only be used when loading from '
                'a single file')

        if isinstance(paths, Selection):
            selection = paths
            if database is not None:
                raise TypeError(
                    'iload: database argument must be None when called with a '
                    'selection')

            database = selection.get_database()

        if skip_unchanged and not isinstance(paths, Selection):
            raise TypeError(
                'iload: need selection when called with "skip_unchanged=True"')

    temp_selection = None
    if database:
        if not selection:
            temp_selection = database.new_selection(paths)
            selection = temp_selection

        if skip_unchanged:
            selection.flag_modified(check)
            it = selection.undig_grouped(skip_unchanged=True)
        else:
            it = selection.undig_grouped()

    else:
        it = (((format, path), []) for path in paths)

    try:
        n_files_total = len(it)
    except TypeError:
        n_files_total = None

    task = None
    if progress is not None:
        if not kind_ids:
            task = make_task('Indexing files', n_files_total)
        else:
            task = make_task('Loading files', n_files_total)

    n_files = 0
    tcommit = time.time()
    database_modified = False
    for (format, path), old_nuts in it:
        if task is not None:
            condition = '(nuts: %i from file, %i from cache)\n  %s' % (
                n_load, n_db, path)
            task.update(n_files, condition)

        n_files += 1
        if database and commit and database_modified:
            tnow = time.time()
            if tnow - tcommit > 20. or n_files % 1000 == 0:
                database.commit()
                tcommit = tnow

        try:
            if check and old_nuts and old_nuts[0].file_modified():
                old_nuts = []

            if segment is not None:
                old_nuts = [
                    nut for nut in old_nuts if nut.file_segment == segment]

            if old_nuts:
                db_only_operation = not kind_ids or all(
                    nut.kind_id in kind_ids and nut.content_in_db
                    for nut in old_nuts)

                if db_only_operation:
                    # logger.debug('using cached information for file %s, '
                    #              % path)

                    for nut in old_nuts:
                        if nut.kind_id in kind_ids:
                            database.undig_content(nut)

                        n_db += 1
                        yield nut

                    continue

            if format == 'detect':
                if old_nuts and not old_nuts[0].file_modified():
                    format_this = old_nuts[0].file_format
                else:
                    format_this = detect_format(path)
            else:
                format_this = format

            mod = get_backend(format_this)
            mtime, size = mod.get_stats(path)

            logger.debug('reading file %s' % path)
            nuts = []
            for nut in mod.iload(format_this, path, segment, content):
                nut.file_path = path
                nut.file_format = format_this
                nut.file_mtime = mtime
                nut.file_size = size

                nuts.append(nut)
                n_load += 1
                yield nut

            if database and nuts != old_nuts:
                if segment is not None:
                    nuts = mod.iload(format_this, path, None, [])
                    for nut in nuts:
                        nut.file_path = path
                        nut.file_format = format_this
                        nut.file_mtime = mtime

                database.dig(nuts)
                database_modified = True

        except FileLoadError:
            logger.error('An error occured while reading file: %s' % path)
            if database:
                database.reset(path)
                database_modified = True

    if task is not None:
        condition = '(nuts: %i from file, %i from cache)' % (n_load, n_db)
        task.update(n_files, condition)
        task.done(condition)

    if database:
        if commit and database_modified:
            database.commit()

        if temp_selection:
            del temp_selection

    logger.debug('iload: from cache: %i, from files: %i, files: %i' % (
        n_db, n_load, n_files))


__all__ = [
    'iload',
    'detect_format',
    'supported_formats',
    'supported_content_kinds',
    'get_backend',
    'FormatDetectionFailed',
    'UnknownFormat',
]
