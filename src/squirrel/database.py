# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

import sys
import os
import logging
import sqlite3
import re

from pyrocko.io_common import FileLoadError
from pyrocko import util
from pyrocko.guts import Object, Int, List, Dict, Tuple, String
from . import error, io
from .model import Nut, to_kind_id, to_kind, separator

logger = logging.getLogger('pyrocko.squirrel.database')

guts_prefix = 'squirrel'


def execute_get1(connection, sql, args):
    return list(connection.execute(sql, args))[0]


g_databases = {}


def get_database(path=None):
    path = os.path.abspath(path)
    if path not in g_databases:
        g_databases[path] = Database(path)

    return g_databases[path]


class Database(object):
    '''
    Shared meta-information database used by Squirrel.
    '''

    def __init__(self, database_path=':memory:', log_statements=False):
        self._database_path = database_path
        try:
            util.ensuredirs(database_path)
            logger.debug('Opening connection to database: %s' % database_path)
            self._conn = sqlite3.connect(database_path)
        except sqlite3.OperationalError:
            raise error.SquirrelError(
                'Cannot connect to database: %s' % database_path)

        self._conn.text_factory = str
        self._tables = {}

        self._initialize_db()
        self._need_commit = False
        if log_statements:
            self._conn.set_trace_callback(self._log_statement)

    def _log_statement(self, statement):
        logger.debug(statement)

    def get_connection(self):
        return self._conn

    def _register_table(self, s):
        m = re.search(r'(\S+)\s*\(([^)]+)\)', s)
        table_name = m.group(1)
        dtypes = m.group(2)
        table_header = []
        for dele in dtypes.split(','):
            table_header.append(dele.split()[:2])

        self._tables[table_name] = table_header

        return s

    def _initialize_db(self):
        c = self._conn
        with c:
            if 1 == len(list(
                    c.execute(
                        '''
                            SELECT name FROM sqlite_master
                                WHERE type = 'table' AND name = '{files}'
                        '''))):
                return

            c.execute(
                '''PRAGMA recursive_triggers = true''')

            c.execute(self._register_table(
                '''
                    CREATE TABLE IF NOT EXISTS files (
                        file_id integer PRIMARY KEY,
                        path text,
                        format text,
                        mtime float,
                        size integer)
                '''))

            c.execute(
                '''
                    CREATE UNIQUE INDEX IF NOT EXISTS index_files_path
                    ON files (path)
                ''')

            c.execute(self._register_table(
                '''
                    CREATE TABLE IF NOT EXISTS nuts (
                        nut_id integer PRIMARY KEY AUTOINCREMENT,
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
                '''))

            c.execute(
                '''
                    CREATE UNIQUE INDEX IF NOT EXISTS index_nuts_file_element
                    ON nuts (file_id, file_segment, file_element)
                ''')

            c.execute(self._register_table(
                '''
                    CREATE TABLE IF NOT EXISTS kind_codes (
                        kind_codes_id integer PRIMARY KEY,
                        kind_id integer,
                        codes text,
                        deltat float)
                '''))

            c.execute(
                '''
                    CREATE UNIQUE INDEX IF NOT EXISTS index_kind_codes
                    ON kind_codes (kind_id, codes, deltat)
                ''')

            c.execute(self._register_table(
                '''
                    CREATE TABLE IF NOT EXISTS kind_codes_count (
                        kind_codes_id integer PRIMARY KEY,
                        count integer)
                '''))

            c.execute(
                '''
                    CREATE INDEX IF NOT EXISTS index_nuts_file_id
                    ON nuts (file_id)
                ''')

            c.execute(
                '''
                    CREATE TRIGGER IF NOT EXISTS delete_nuts_on_delete_file
                    BEFORE DELETE ON files FOR EACH ROW
                    BEGIN
                      DELETE FROM nuts where file_id == old.file_id;
                    END
                ''')

            # trigger only on size to make silent update of mtime possible
            c.execute(
                '''
                    CREATE TRIGGER IF NOT EXISTS delete_nuts_on_update_file
                    BEFORE UPDATE OF size ON files FOR EACH ROW
                    BEGIN
                      DELETE FROM nuts where file_id == old.file_id;
                    END
                ''')

            c.execute(
                '''
                    CREATE TRIGGER IF NOT EXISTS increment_kind_codes
                    BEFORE INSERT ON nuts FOR EACH ROW
                    BEGIN
                        INSERT OR IGNORE INTO kind_codes_count
                        VALUES (new.kind_codes_id, 0);
                        UPDATE kind_codes_count
                        SET count = count + 1
                        WHERE new.kind_codes_id == kind_codes_id;
                    END
                ''')

            c.execute(
                '''
                    CREATE TRIGGER IF NOT EXISTS decrement_kind_codes
                    BEFORE DELETE ON nuts FOR EACH ROW
                    BEGIN
                        UPDATE kind_codes_count
                        SET count = count - 1
                        WHERE old.kind_codes_id == kind_codes_id;
                    END
                ''')

    def dig(self, nuts):
        '''
        Store or update content meta-information.

        Given ``nuts`` are assumed to represent an up-to-date and complete
        inventory of a set of files. Any old information about these files is
        first pruned from the database (via database triggers). If such content
        is part of a live selection, it is also removed there. Then the new
        content meta-information is inserted into the main database. The
        content is not automatically inserted into the live selections again.
        It is in the responsibility of the selection object to perform this
        step.
        '''

        nuts = list(nuts)

        if not nuts:
            return

        c = self._conn.cursor()
        files = set()
        kind_codes = set()
        for nut in nuts:
            files.add((
                nut.file_path,
                nut.file_format,
                nut.file_mtime,
                nut.file_size))
            kind_codes.add((nut.kind_id, nut.codes, nut.deltat or 0.0))

        c.executemany(
            'INSERT OR IGNORE INTO files VALUES (NULL,?,?,?,?)', files)

        c.executemany(
            '''UPDATE files SET
                format = ?, mtime = ?, size = ?
                WHERE path == ?
            ''',
            ((x[1], x[2], x[3], x[0]) for x in files))

        c.executemany(
            'INSERT OR IGNORE INTO kind_codes VALUES (NULL,?,?,?)', kind_codes)

        c.executemany(
            '''
                INSERT INTO nuts VALUES
                    (NULL, (
                        SELECT file_id FROM files
                        WHERE path == ?
                     ),?,?,?,
                     (
                        SELECT kind_codes_id FROM kind_codes
                        WHERE kind_id == ? AND codes == ? AND deltat == ?
                     ), ?,?,?,?,?)
            ''',
            ((nut.file_path, nut.file_segment, nut.file_element,
              nut.kind_id,
              nut.kind_id, nut.codes, nut.deltat or 0.0,
              nut.tmin_seconds, nut.tmin_offset,
              nut.tmax_seconds, nut.tmax_offset,
              nut.kscale) for nut in nuts))

        self._need_commit = True
        c.close()

    def undig(self, path):
        sql = '''
            SELECT
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
            FROM files
            INNER JOIN nuts ON files.file_id = nuts.file_id
            INNER JOIN kind_codes
                ON nuts.kind_codes_id == kind_codes.kind_codes_id
            WHERE path == ?
        '''

        return [Nut(values_nocheck=row)
                for row in self._conn.execute(sql, (path,))]

    def undig_all(self):
        sql = '''
            SELECT
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
            FROM files
            INNER JOIN nuts ON files.file_id == nuts.file_id
            INNER JOIN kind_codes
                ON nuts.kind_codes_id == kind_codes.kind_codes_id
        '''

        nuts = []
        path = None
        for values in self._conn.execute(sql):
            if path is not None and values[0] != path:
                yield path, nuts
                nuts = []

            if values[1] is not None:
                nuts.append(Nut(values_nocheck=values))

            path = values[0]

        if path is not None:
            yield path, nuts

    def undig_many(self, paths):
        selection = self.new_selection(paths)

        for (_, path), nuts in selection.undig_grouped():
            yield path, nuts

        del selection

    def new_selection(self, paths=None):
        from .selection import Selection
        selection = Selection(self)
        if paths:
            selection.add(paths)
        return selection

    def commit(self):
        if self._need_commit:
            self._conn.commit()
            self._need_commit = False

    def undig_content(self, nut):
        return None

    def remove(self, path):
        '''
        Prune content meta-inforamation about a given file.

        All content pieces belonging to file ``path`` are removed from the
        main database and any attached live selections (via database triggers).
        '''

        self._conn.execute(
            'DELETE FROM files WHERE path = ?', (path,))

    def reset(self, path):
        '''
        Prune information associated with a given file, but keep the file path.

        This method is called when reading a file failed. File attributes,
        format, size and modification time are set to NULL. File content
        meta-information is removed from the database and any attached live
        selections (via database triggers).
        '''

        self._conn.execute(
            '''
                UPDATE files SET
                    format = NULL,
                    mtime = NULL,
                    size = NULL
                WHERE path = ?
            ''', (path,))

    def silent_touch(self, path):
        '''
        Update modification time of file without initiating reindexing.

        Useful to prolong validity period of data with expiration date.
        '''

        c = self._conn
        with c:

            sql = 'SELECT format, size FROM files WHERE path = ?'
            fmt, size = execute_get1(c, sql, (path,))

            mod = io.get_backend(fmt)
            mod.touch(path)
            file_stats = mod.get_stats(path)

            if file_stats[1] != size:
                raise FileLoadError(
                    'Silent update for file "%s" failed: size has changed.'
                    % path)

            sql = '''
                UPDATE files
                SET mtime = ?
                WHERE path = ?
            '''
            c.execute(sql, (file_stats[0], path))

    def _iter_counts(self, kind=None, kind_codes_count='kind_codes_count'):
        args = []
        sel = ''
        if kind is not None:
            sel = 'AND kind_codes.kind_id == ?'
            args.append(to_kind_id(kind))

        sql = ('''
            SELECT
                kind_codes.kind_id,
                kind_codes.codes,
                kind_codes.deltat,
                %(kind_codes_count)s.count
            FROM %(kind_codes_count)s
            INNER JOIN kind_codes
                ON %(kind_codes_count)s.kind_codes_id
                    == kind_codes.kind_codes_id
            WHERE %(kind_codes_count)s.count > 0
                ''' + sel + '''
        ''') % {'kind_codes_count': kind_codes_count}

        for kind_id, codes, deltat, count in self._conn.execute(sql, args):
            yield (
                to_kind(kind_id),
                tuple(codes.split(separator)),
                deltat), count

    def _iter_deltats(self, kind=None, kind_codes_count='kind_codes_count'):
        args = []
        sel = ''
        if kind is not None:
            assert isinstance(kind, str)
            sel = 'AND kind_codes.kind_id == ?'
            args.append(to_kind_id(kind))

        sql = ('''
            SELECT DISTINCT kind_codes.deltat FROM %(kind_codes_count)s
            INNER JOIN kind_codes
                ON %(kind_codes_count)s.kind_codes_id
                    == kind_codes.kind_codes_id
            WHERE %(kind_codes_count)s.count > 0
                ''' + sel + '''
            ORDER BY kind_codes.deltat
        ''') % {'kind_codes_count': kind_codes_count}

        for row in self._conn.execute(sql, args):
            yield row[0]

    def _iter_codes(self, kind=None, kind_codes_count='kind_codes_count'):
        args = []
        sel = ''
        if kind is not None:
            assert isinstance(kind, str)
            sel = 'AND kind_codes.kind_id == ?'
            args.append(to_kind_id(kind))

        sql = ('''
            SELECT DISTINCT kind_codes.codes FROM %(kind_codes_count)s
            INNER JOIN kind_codes
                ON %(kind_codes_count)s.kind_codes_id
                    == kind_codes.kind_codes_id
            WHERE %(kind_codes_count)s.count > 0
                ''' + sel + '''
            ORDER BY kind_codes.codes
        ''') % {'kind_codes_count': kind_codes_count}

        for row in self._conn.execute(sql, args):
            yield tuple(row[0].split(separator))

    def _iter_kinds(self, codes=None, kind_codes_count='kind_codes_count'):
        args = []
        sel = ''
        if codes is not None:
            assert isinstance(codes, tuple)
            sel = 'AND kind_codes.codes == ?'
            args.append(separator.join(codes))

        sql = ('''
            SELECT DISTINCT kind_codes.kind_id FROM %(kind_codes_count)s
            INNER JOIN kind_codes
                ON %(kind_codes_count)s.kind_codes_id
                    == kind_codes.kind_codes_id
            WHERE %(kind_codes_count)s.count > 0
                ''' + sel + '''
            ORDER BY kind_codes.kind_id
        ''') % {'kind_codes_count': kind_codes_count}

        for row in self._conn.execute(sql, args):
            yield to_kind(row[0])

    def iter_kinds(self, codes=None):
        return self._iter_kinds(codes=codes)

    def iter_codes(self, kind=None):
        return self._iter_codes(kind=kind)

    def iter_counts(self, kind=None):
        return self._iter_counts(kind=kind)

    def get_kinds(self, codes=None):
        return list(self.iter_kinds(codes=codes))

    def get_codes(self, kind=None):
        return list(self.iter_codes(kind=kind))

    def get_counts(self, kind=None):
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

    def get_nfiles(self):
        sql = '''SELECT COUNT(*) FROM files'''
        for row in self._conn.execute(sql):
            return row[0]

    def get_nnuts(self):
        sql = '''SELECT COUNT(*) FROM nuts'''
        for row in self._conn.execute(sql):
            return row[0]

    def get_total_size(self):
        sql = '''
            SELECT SUM(files.size) FROM files
        '''

        for row in self._conn.execute(sql):
            return row[0] or 0

    def get_stats(self):
        return DatabaseStats(
            nfiles=self.get_nfiles(),
            nnuts=self.get_nnuts(),
            kinds=self.get_kinds(),
            codes=self.get_codes(),
            counts=self.get_counts(),
            total_size=self.get_total_size())

    def __str__(self):
        return str(self.get_stats())

    def print_tables(self, stream=None):
        for table in [
                'files',
                'nuts',
                'kind_codes',
                'kind_codes_count']:

            self.print_table(table, stream=stream)

    def print_table(self, name, stream=None):

        if stream is None:
            stream = sys.stdout

        class hstr(str):
            def __repr__(self):
                return self

        w = stream.write
        w('\n')
        w('\n')
        w(name)
        w('\n')
        sql = 'SELECT * FROM %s' % name
        tab = []
        if name in self._tables:
            headers = self._tables[name]
            tab.append([None for _ in headers])
            tab.append([hstr(x[0]) for x in headers])
            tab.append([hstr(x[1]) for x in headers])
            tab.append([None for _ in headers])

        for row in self._conn.execute(sql):
            tab.append([x for x in row])

        widths = [
            max((len(repr(x)) if x is not None else 0) for x in col)
            for col in zip(*tab)]

        for row in tab:
            w(' '.join(
                (repr(x).ljust(wid) if x is not None else ''.ljust(wid, '-'))
                for (x, wid) in zip(row, widths)))

            w('\n')

        w('\n')


class DatabaseStats(Object):
    '''
    Container to hold statistics about contents cached in meta-information db.
    '''

    nfiles = Int.T(
        help='number of files in database')
    nnuts = Int.T(
        help='number of index nuts in database')
    codes = List.T(
        Tuple.T(content_t=String.T()),
        help='available code sequences in database, e.g. '
             '(agency, network, station, location) for stations nuts.')
    kinds = List.T(
        String.T(),
        help='available content types in database')
    total_size = Int.T(
        help='aggregated file size of files referenced in database')
    counts = Dict.T(
        String.T(), Dict.T(Tuple.T(content_t=String.T()), Int.T()),
        help='breakdown of how many nuts of any content type and code '
             'sequence are available in database, ``counts[kind][codes]``')

    def __str__(self):
        kind_counts = dict(
            (kind, sum(self.counts[kind].values())) for kind in self.kinds)

        codes = ['.'.join(x) for x in self.codes]

        if len(codes) > 20:
            scodes = '\n' + util.ewrap(codes[:10], indent='  ') \
                + '\n  [%i more]\n' % (len(codes) - 20) \
                + util.ewrap(codes[-10:], indent='  ')
        else:
            scodes = '\n' + util.ewrap(codes, indent='  ') \
                if codes else '<none>'

        s = '''
Available codes:               %s
Number of files:               %i
Total size of known files:     %s
Number of index nuts:          %i
Available content kinds:       %s''' % (
            scodes,
            self.nfiles,
            util.human_bytesize(self.total_size),
            self.nnuts,
            ', '.join('%s: %i' % (
                kind, kind_counts[kind]) for kind in sorted(self.kinds)))

        return s


__all__ = [
    'Database',
    'DatabaseStats',
]
