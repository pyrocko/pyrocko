import time
import sqlite3
import json
import struct
import logging
import math
import hashlib

import numpy as num

from pyrocko import util, model
from pyrocko import orthodrome
from pyrocko import moment_tensor as pmt

from .error import SquirrelError

logger = logging.getLogger('psq.elk')

km = 1000.


def unpack_application_id(b):
    return struct.unpack('>i', b)[0]


ELK1 = unpack_application_id(b'ELK1')

# Increment when the table layout changes. Stored as SQLite user_version.
SCHEMA_VERSION = 2

# Event attributes which are managed by the database and are therefore not
# stored in the extras column.
ELK_EXTRAS = (
    'elk_id', 'group_id', 'sequence_id', 'catalog_id',
    'catalog_time_modified', 'catalog_version', 'time_created',
    'superseded_by')

# Quantization of source parameters for the content hash. Values are
# multiplied with the given factor and rounded to integers, so that
# insignificant floating point differences, e.g. from different parsers, do
# not change the hash.
CONTENT_HASH_VERSION = 'elk1'
CONTENT_HASH_FACTORS = (
    ('time', 10**6),
    ('lat', 10**7),
    ('lon', 10**7),
    ('north_shift', 10**3),
    ('east_shift', 10**3),
    ('depth', 10**3),
    ('magnitude', 10**4))


# Stand-in for unknown magnitudes in the spatio-temporal index, which needs
# numbers. Below any real magnitude, so that such events come last.
RTREE_MAGNITUDE_UNKNOWN = -100.

# Columns of the events table, except for the integer primary key, in the
# order used for insertion.
EVENT_COLUMNS = (
    'elk_id', 'group_id', 'sequence_id', 'superseded_by', 'content_hash',
    'source', 'source_id', 'source_time_modified', 'source_version',
    'time_created', 'flag', 'time', 'lat', 'lon', 'north_shift',
    'east_shift', 'depth', 'magnitude', 'magnitude_type', 'duration',
    'region', 'tags', 'extras')


def new_elk_id():
    '''
    Create new time-ordered event ID (UUID version 7).
    '''

    try:
        from uuid import uuid7
    except ImportError:
        from uuid_backport import uuid7

    return str(uuid7())


def content_hash(
        source, source_id, time, lat, lon, north_shift, east_shift, depth,
        magnitude, magnitude_type):

    '''
    Get hash identifying an event revision from a specific catalog.

    Two events with the same hash are considered to be the same event. The
    hash covers catalog name, catalog event ID and quantized source
    parameters, but not moment tensor, duration, region, tags and extras.
    '''

    parts = [CONTENT_HASH_VERSION, source or '', source_id or '']
    for (_, factor), value in zip(CONTENT_HASH_FACTORS, (
            time, lat, lon, north_shift, east_shift, depth, magnitude)):

        parts.append('' if value is None else '%i' % round(value * factor))

    parts.append(magnitude_type or '')

    return hashlib.blake2b(
        '\x1f'.join(parts).encode('utf-8'), digest_size=16).hexdigest()


def scaling(magnitude):
    length = 10 * num.exp(1.15 * magnitude)
    duration = length / (0.75 * 3200.0)
    return duration, length


def effective_latlon(lat, lon, north, east):
    return tuple(float(x) for x in orthodrome.ne_to_latlon_proj(
        lat, lon, north, east))


def surface_distance(
        a_lat, a_lon, a_north, a_east, b_lat, b_lon, b_north, b_east):

    if a_lat == b_lat and a_lon == b_lon:
        return math.sqrt((b_north - a_north)**2 + (b_east - a_east)**2)
    else:
        if a_north != 0.0 or a_east != 0.0:
            a_lat, a_lon = effective_latlon(a_lat, a_lon, a_north, a_east)
        if b_north != 0.0 or b_east != 0.0:
            b_lat, b_lon = effective_latlon(b_lat, b_lon, b_north, b_east)

        return float(orthodrome.distance_proj(a_lat, a_lon, b_lat, b_lon))


def elk_distance(
        a_time, a_magnitude, a_lat, a_lon, a_north, a_east,
        b_time, b_magnitude, b_lat, b_lon, b_north, b_east):

    if a_magnitude is None or b_magnitude is None:
        return None

    norm_magnitude = 1.0
    norm_time, norm_distance = scaling(min(a_magnitude, b_magnitude))
    norm_time *= 10
    norm_distance *= 50

    d_time = abs(a_time - b_time)
    d_magnitude = abs(a_magnitude - b_magnitude)
    distance = surface_distance(
        a_lat, a_lon, a_north, a_east, b_lat, b_lon, b_north, b_east)

    return float(
        d_time / norm_time
        + d_magnitude / norm_magnitude
        + distance / norm_distance)


def connection_hook(connection):
    connection.create_function(
        'elk_distance', 12, elk_distance, deterministic=True)


def connect(path):
    util.ensuredirs(path)
    connection = sqlite3.connect(path)
    connection_hook(connection)
    return connection


def attach(connection, prefix, path):
    connection.execute('''
        ATTACH DATABASE ? AS ?
    ''', (path, prefix))


class ElkError(SquirrelError):
    pass


def _first(rows):
    # Works on cursors and on the result lists returned by Squirrel's
    # database connection.
    for row in rows:
        return row

    return None


def init_database(connection, prefix):
    util.experimental_feature_used('pyrocko.squirrel.elk')
    cursor = connection.cursor()

    cursor.execute('''
        PRAGMA %s.application_id = %i
    ''' % (prefix, ELK1))

    cursor.execute('''
        PRAGMA %s.user_version = %i
    ''' % (prefix, SCHEMA_VERSION))

    cursor.execute('''
        CREATE TABLE %s.events (
            event_id INTEGER PRIMARY KEY,
            elk_id TEXT NOT NULL UNIQUE,
            group_id TEXT,
            sequence_id TEXT,
            superseded_by TEXT,
            content_hash TEXT NOT NULL,
            source TEXT,
            source_id TEXT,
            source_time_modified REAL,
            source_version TEXT,
            time_created REAL,
            flag TEXT,
            time REAL,
            lat REAL,
            lon REAL,
            north_shift REAL,
            east_shift REAL,
            depth REAL,
            magnitude REAL,
            magnitude_type TEXT,
            duration REAL,
            region TEXT,
            tags TEXT,
            extras TEXT )
    ''' % prefix)

    cursor.execute('''
        CREATE TABLE %s.moment_tensors (
            elk_id TEXT PRIMARY KEY,
            mnn REAL,
            mee REAL,
            mdd REAL,
            mne REAL,
            mnd REAL,
            med REAL )
    ''' % prefix)

    # Maps IDs which have been given up in favour of an older ID of the same
    # event, e.g. when merging independently created databases.
    cursor.execute('''
        CREATE TABLE %s.aliases (
            alias_id TEXT PRIMARY KEY,
            elk_id TEXT NOT NULL )
    ''' % prefix)

    cursor.execute('''
        CREATE INDEX %s.index_time ON events ( time )
    ''' % prefix)

    cursor.execute('''
        CREATE INDEX %s.index_group_id ON events ( group_id )
    ''' % prefix)

    cursor.execute('''
        CREATE INDEX %s.index_source ON events ( source, source_id )
    ''' % prefix)

    cursor.execute('''
        CREATE UNIQUE INDEX %s.index_content_hash ON events ( content_hash )
    ''' % prefix)

    # Spatio-temporal index for the selection of the largest events in a
    # given region and time span. Events are points, the R-tree stores their
    # coordinates as 32-bit floats, rounded outwards. Cartesian offsets
    # (north_shift, east_shift) are not taken into account. The integer
    # primary key of the events table is used to link the two, because, unlike
    # the implicit rowid, it is stable under VACUUM.
    cursor.execute('''
        CREATE VIRTUAL TABLE %s.events_rtree USING rtree(
            event_id,
            time_min, time_max,
            magnitude_min, magnitude_max,
            lat_min, lat_max,
            lon_min, lon_max )
    ''' % prefix)

    rtree_values = '''
        new.event_id,
        new.time, new.time,
        coalesce(new.magnitude, %(m)g), coalesce(new.magnitude, %(m)g),
        new.lat, new.lat,
        new.lon, new.lon''' % dict(m=RTREE_MAGNITUDE_UNKNOWN)

    cursor.execute('''
        CREATE TRIGGER %s.events_rtree_insert AFTER INSERT ON events
        BEGIN
            INSERT INTO events_rtree VALUES ( %s );
        END
    ''' % (prefix, rtree_values))

    cursor.execute('''
        CREATE TRIGGER %s.events_rtree_update
        AFTER UPDATE OF time, magnitude, lat, lon ON events
        BEGIN
            INSERT OR REPLACE INTO events_rtree VALUES ( %s );
        END
    ''' % (prefix, rtree_values))

    cursor.execute('''
        CREATE TRIGGER %s.events_rtree_delete AFTER DELETE ON events
        BEGIN
            DELETE FROM events_rtree WHERE event_id = old.event_id;
        END
    ''' % prefix)

    connection.commit()


def check_database(connection, prefix):
    util.experimental_feature_used('pyrocko.squirrel.elk')
    cursor = connection.cursor()
    application_id = _first(cursor.execute(
        'PRAGMA %s.application_id' % prefix))[0]
    user_version = _first(cursor.execute(
        'PRAGMA %s.user_version' % prefix))[0]
    cursor.close()

    if application_id != ELK1:
        raise ElkError('Not an ELK database.')

    if user_version != SCHEMA_VERSION:
        raise ElkError(
            'Unsupported ELK database schema version: %i (expected %i).' % (
                user_version, SCHEMA_VERSION))


def serialize_extras(extras):
    return json.dumps(
        dict(
            (k, v)
            for (k, v)
            in extras.items()
            if k not in ELK_EXTRAS))


def _event_hash(ev, source_id):
    return content_hash(
        ev.catalog or '',
        source_id,
        ev.time,
        ev.lat,
        ev.lon,
        ev.north_shift,
        ev.east_shift,
        ev.depth,
        ev.magnitude,
        ev.magnitude_type or '')


def _rename_elk_id(cursor, prefix, old_id, new_id):
    for table, column in [
            ('events', 'elk_id'),
            ('events', 'group_id'),
            ('events', 'sequence_id'),
            ('events', 'superseded_by'),
            ('moment_tensors', 'elk_id'),
            ('aliases', 'elk_id')]:

        cursor.execute('''
            UPDATE %s.%s SET %s = ? WHERE %s = ?
        ''' % (prefix, table, column, column), (new_id, old_id))

    cursor.execute('''
        INSERT OR REPLACE INTO %s.aliases VALUES (?, ?)
    ''' % prefix, (old_id, new_id))


def add_events(connection, prefix, events):
    '''
    Add events to database.

    Events already in the database (same catalog, catalog event ID and source
    parameters) are skipped. New events get a new time-ordered ID, unless
    they carry an ID in ``extras['elk_id']``, e.g. when they have been
    obtained from another ELK database. If the same event is known under two
    different IDs, the older ID is kept and the newer one is registered as
    alias.
    '''

    check_database(connection, prefix)

    cursor = connection.cursor()
    try:
        _add_events(connection, cursor, prefix, events)
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()

    connection.commit()


def _add_events(connection, cursor, prefix, events):
    events = sorted(events, key=lambda ev: ev.time)

    time_created = time.time()

    new = {}
    revised = set()
    n_input_duplicates = 0
    n_database_duplicates = 0
    n_aliased = 0
    for ev in events:
        elk_id = ev.extras.get('elk_id')
        if elk_id is not None:
            source_id = ev.extras.get('catalog_id', '')
        else:
            source_id = ev.name or ''

        chash = _event_hash(ev, source_id)

        if chash in new:
            n_input_duplicates += 1
            other_id, other_ev = new[chash][:2]
            if _newer_revision(
                    _source_time_modified(ev),
                    _source_time_modified(other_ev)):
                new[chash] = (other_id, ev) + new[chash][2:]

            if elk_id is not None and elk_id != other_id:
                keep_id, give_id = sorted((elk_id, other_id))
                new[chash] = (keep_id,) + new[chash][1:]
                cursor.execute('''
                    INSERT OR REPLACE INTO %s.aliases VALUES (?, ?)
                ''' % prefix, (give_id, keep_id))
                n_aliased += 1

            continue

        row = _first(cursor.execute('''
            SELECT elk_id FROM %s.events WHERE content_hash = ?
        ''' % prefix, (chash,)))

        if row is not None:
            n_database_duplicates += 1
            (existing_id,) = row
            if _update_source_revision(cursor, prefix, existing_id, ev):
                revised.add((ev.catalog or '', source_id))

            if elk_id is not None and elk_id != existing_id:
                if elk_id < existing_id:
                    _rename_elk_id(cursor, prefix, existing_id, elk_id)
                else:
                    cursor.execute('''
                        INSERT OR REPLACE INTO %s.aliases VALUES (?, ?)
                    ''' % prefix, (elk_id, existing_id))

                n_aliased += 1

            continue

        if elk_id is not None:
            elk_id = resolve_elk_id(connection, prefix, elk_id)
            if _first(cursor.execute('''
                    SELECT 1 FROM %s.events WHERE elk_id = ?
                    ''' % prefix, (elk_id,))):

                raise ElkError(
                    'Event ID %s is already used for an event with different '
                    'source parameters.' % elk_id)

        new[chash] = (
            elk_id or new_elk_id(),
            ev,
            source_id,
            ev.extras.get('time_created', time_created))

    if n_aliased:
        logger.info(
            'Registered %i alias ID%s for events known under different IDs.'
            % (n_aliased, util.plural_s(n_aliased)))

    if not new:
        logger.info('No new events. Nothing to store.')
        _update_superseded(cursor, prefix, revised)
        return

    logger.info(
        'Inserting %i new events (%i already in database, %i duplicates in '
        'input)' % (
            len(new),
            n_database_duplicates,
            n_input_duplicates))

    cursor.executemany(
        'INSERT INTO %s.moment_tensors VALUES (%s)' % (
            prefix, ','.join('?' * 7)), (
                (elk_id,) + tuple(ev.moment_tensor.m6())
                for (elk_id, ev, _, _) in new.values()
                if ev.moment_tensor is not None))

    cursor.executemany(
        'INSERT INTO %s.events ( %s ) VALUES ( %s )' % (
            prefix,
            ', '.join(EVENT_COLUMNS),
            ', '.join('?' * len(EVENT_COLUMNS))), ((
                elk_id,
                None,
                None,
                None,
                chash,
                ev.catalog or '',
                source_id,
                _source_time_modified(ev),
                ev.extras.get('catalog_version'),
                ev_time_created,
                '',
                ev.time,
                ev.lat,
                ev.lon,
                ev.north_shift,
                ev.east_shift,
                ev.depth,
                ev.magnitude,
                ev.magnitude_type or '',
                ev.duration,
                ev.region,
                json.dumps(ev.tags),
                serialize_extras(ev.extras))
            for (chash, (elk_id, ev, source_id, ev_time_created))
            in new.items()))

    _update_superseded(cursor, prefix, revised | set(
        (ev.catalog or '', source_id)
        for (_, ev, source_id, _) in new.values()))

    _update_groups(cursor, prefix)


def _source_time_modified(ev):
    t = ev.extras.get('catalog_time_modified')
    return None if t is None else float(t)


def _newer_revision(t, t_ref):
    return t is not None and (t_ref is None or t > t_ref)


def _update_source_revision(cursor, prefix, elk_id, ev):
    '''
    Update revision info of a stored event if the given one is newer.

    This happens when a catalog re-issues an earlier revision of an event.
    Returns ``True`` if the stored info has been changed.
    '''

    t = _source_time_modified(ev)
    (t_stored,) = _first(cursor.execute('''
        SELECT source_time_modified FROM %s.events WHERE elk_id = ?
    ''' % prefix, (elk_id,)))

    if not _newer_revision(t, t_stored):
        return False

    cursor.execute('''
        UPDATE %s.events
        SET source_time_modified = ?, source_version = ?
        WHERE elk_id = ?
    ''' % prefix, (t, ev.extras.get('catalog_version'), elk_id))

    return True


def _update_superseded(cursor, prefix, keys):
    '''
    Mark older revisions of events from the same catalog as superseded.

    Among all events with same catalog and catalog event ID, the most recent
    revision is current. Revisions are ordered by the modification time given
    by the catalog, if available for all of them, otherwise by the time they
    have been added to the database. If there are several most recent
    revisions, the oldest ID among them is used as reference for the others.
    '''

    for source, source_id in keys:
        if not source or not source_id:
            continue

        rows = list(cursor.execute('''
            SELECT elk_id, source_time_modified, time_created FROM %s.events
            WHERE source = ? AND source_id = ?
            ORDER BY elk_id
        ''' % prefix, (source, source_id)))

        if len(rows) < 2:
            continue

        if all(t_source is not None for (_, t_source, _) in rows):
            rows = [(elk_id, t_source) for (elk_id, t_source, _) in rows]
        else:
            rows = [(elk_id, t_added) for (elk_id, _, t_added) in rows]

        tmax = max(t for (_, t) in rows)
        current = [elk_id for (elk_id, t) in rows if t == tmax]
        if len(current) > 1:
            logger.warning(
                'Several revisions of event %s from catalog %s with same '
                'time. Cannot decide which one is current.' % (
                    source_id, source))

        cursor.executemany('''
            UPDATE %s.events SET superseded_by = ? WHERE elk_id = ?
        ''' % prefix, (
            (None if t == tmax else current[0], elk_id)
            for (elk_id, t) in rows))


def _update_groups(cursor, prefix):
    '''
    Assign group IDs to ungrouped events.

    Groups are the connected components of the graph linking similar events.
    The oldest event ID in a group is used as group ID.
    '''

    ungrouped = list(cursor.execute('''
        SELECT elk_id, time, lat, lon, north_shift, east_shift, magnitude
        FROM %s.events
        WHERE group_id IS NULL
        ORDER BY elk_id
    ''' % prefix))

    for (elk_id, etime, lat, lon, north_shift, east_shift, magnitude) \
            in ungrouped:

        if magnitude is None:
            group_ids = []
        else:
            delta_time, _ = scaling(magnitude)
            delta_time *= 10.
            group_ids = [row[0] for row in cursor.execute('''
                SELECT DISTINCT group_id
                FROM %s.events
                WHERE
                    time > ? AND
                    time < ? AND
                    group_id IS NOT NULL AND
                    elk_distance(
                        time, magnitude, lat, lon, north_shift, east_shift,
                        ?, ?, ?, ?, ?, ?) < 1
            ''' % prefix, (
                etime - delta_time,
                etime + delta_time,
                etime,
                magnitude,
                lat,
                lon,
                north_shift,
                east_shift))]

        group_ids = sorted(set(group_ids) | {elk_id})
        group_id = group_ids[0]

        cursor.execute('''
            UPDATE %s.events SET group_id = ? WHERE elk_id = ?
        ''' % prefix, (group_id, elk_id))

        merge_ids = [x for x in group_ids[1:] if x != elk_id]
        if merge_ids:
            cursor.execute('''
                UPDATE %s.events SET group_id = ? WHERE group_id IN ( %s )
            ''' % (prefix, ','.join('?'*len(merge_ids))),
                [group_id] + merge_ids)


def resolve_elk_id(connection, prefix, elk_id):
    '''
    Get current ID for a given ID, which may have been given up as alias.
    '''

    row = _first(connection.execute('''
        SELECT elk_id FROM %s.aliases WHERE alias_id = ?
    ''' % prefix, (elk_id,)))

    return row[0] if row is not None else elk_id


def get_events(
        connection,
        prefix,
        nlatest=None,
        filter=None,
        include_superseded=False):

    '''
    Query events.

    Limits are treated as half-open intervals, e.g. [tmin, tmax).
    '''

    check_database(connection, prefix)

    if filter is None:
        filter = model.event.EventFilter()
    elif isinstance(filter, dict):
        filter = model.event.EventFilter(**filter)

    cursor = connection.cursor()
    where, args = filter.sql_condition()

    if not include_superseded:
        where = (where + ' AND ' if where else 'WHERE ') \
            + 'a.superseded_by IS NULL'

    if nlatest is not None:
        limit = 'LIMIT ?'
        args.append(nlatest)
    else:
        limit = ''

    sql = _sql_select_events(prefix) + where + '''
    ORDER BY ( SELECT -time FROM ''' + prefix + '''.events b
               WHERE b.elk_id = a.group_id ),
               -a.time
''' + limit

    events = [_row_to_event(row) for row in cursor.execute(sql, args)]
    cursor.close()

    return events


def get_event_groups(
        connection,
        prefix,
        filter=None,
        limit=None,
        after=None,
        include_superseded=False):

    '''
    Query groups of events, page by page.

    Groups are ordered by descending group key ``(time, group_id)``, where
    ``time`` is the time of the event the group is named after. Only groups
    with at least one event matching the filter are returned and only the
    matching events of each group are included.

    :param limit:
        Maximum number of groups to return.
    :param after:
        Group key of the last group of the previous page. Only groups with
        smaller key are returned.

    :returns:
        List of ``(group_key, events)`` tuples.
    '''

    check_database(connection, prefix)

    if filter is None:
        filter = model.event.EventFilter()
    elif isinstance(filter, dict):
        filter = model.event.EventFilter(**filter)

    where, filter_args = filter.sql_condition(table='a')
    conditions = [where[len('WHERE '):]] if where else []
    if not include_superseded:
        conditions.append('a.superseded_by IS NULL')

    condition = ' AND '.join(conditions) if conditions else '1'

    args = []
    after_condition = ''
    if after is not None:
        after_condition = '''
            AND ( g.time < ? OR ( g.time = ? AND g.elk_id < ? ) )'''
        args.extend([after[0], after[0], after[1]])

    args.extend(filter_args)

    if limit is not None:
        limit_clause = 'LIMIT ?'
        args.append(limit)
    else:
        limit_clause = ''

    cursor = connection.cursor()

    group_keys = list(cursor.execute('''
        SELECT g.time, g.elk_id FROM ''' + prefix + '''.events g
        WHERE g.elk_id = g.group_id''' + after_condition + '''
            AND EXISTS (
                SELECT 1 FROM ''' + prefix + '''.events a
                WHERE a.group_id = g.elk_id AND ''' + condition + ''' )
        ORDER BY g.time DESC, g.elk_id DESC
    ''' + limit_clause, args))

    if not group_keys:
        cursor.close()
        return []

    group_ids = [group_id for (_, group_id) in group_keys]
    groups = dict((group_id, []) for group_id in group_ids)
    sql = _sql_select_events(prefix) + '''
        WHERE a.group_id IN ( %s ) AND %s
        ORDER BY a.time DESC
    ''' % (','.join('?' * len(group_ids)), condition)

    for row in cursor.execute(sql, group_ids + filter_args):
        ev = _row_to_event(row)
        groups[ev.extras['group_id']].append(ev)

    cursor.close()

    return [
        ((time, group_id), groups[group_id])
        for (time, group_id) in group_keys]


def split_lon_range(lon_min, lon_max):
    '''
    Get longitude range as list of non-wrapping ranges.

    If ``lon_min > lon_max``, the range crosses the date line and is split in
    two. Without limits, ``[(None, None)]`` is returned.
    '''

    if lon_min is None and lon_max is None:
        return [(None, None)]

    lon_min = -180. if lon_min is None else lon_min
    lon_max = 180. if lon_max is None else lon_max
    if lon_min <= lon_max:
        return [(lon_min, lon_max)]
    else:
        # Range crossing the date line.
        return [(lon_min, 180.), (-180., lon_max)]


def get_largest_events(
        connection,
        prefix,
        n,
        filter=None,
        lat_min=None,
        lat_max=None,
        lon_min=None,
        lon_max=None,
        include_superseded=False):

    '''
    Get the largest events in a given region and time span.

    Events are selected by magnitude, using the spatio-temporal index.
    Events without magnitude come last. Group membership is not taken into
    account: several events of a group may be selected. They can be combined
    using ``extras['group_id']``. Cartesian offsets of the events
    (``north_shift``, ``east_shift``) are not taken into account.

    :param n:
        Maximum number of events to return.
    :param filter:
        Event filter (time, magnitude and depth limits).
    :param lat_min, lat_max, lon_min, lon_max:
        Region limits [deg]. If ``lon_min > lon_max``, the region crosses the
        date line.

    :returns:
        List of events, ordered by descending magnitude.
    '''

    check_database(connection, prefix)

    if filter is None:
        filter = model.event.EventFilter()
    elif isinstance(filter, dict):
        filter = model.event.EventFilter(**filter)

    where, filter_args = filter.sql_condition(table='a')
    exact_conditions = [where[len('WHERE '):]] if where else []
    if not include_superseded:
        exact_conditions.append('a.superseded_by IS NULL')

    # Conditions on the R-tree only select candidates; exact limits are
    # applied to the events table (the R-tree's coordinates are rounded).
    index_conditions = []
    index_args = []
    for (name, vmin, vmax) in [
            ('time', filter.time_min, filter.time_max),
            ('lat', lat_min, lat_max)]:

        if vmin is not None:
            index_conditions.append('r.%s_max >= ?' % name)
            index_args.append(vmin)

        if vmax is not None:
            index_conditions.append('r.%s_min <= ?' % name)
            index_args.append(vmax)

    for (name, vmin, vmax) in [('lat', lat_min, lat_max)]:
        if vmin is not None:
            exact_conditions.append('a.%s >= ?' % name)
            filter_args.append(vmin)

        if vmax is not None:
            exact_conditions.append('a.%s <= ?' % name)
            filter_args.append(vmax)

    lon_ranges = split_lon_range(lon_min, lon_max)

    def query(select, magnitude_min, suffix='', suffix_args=()):
        results = []
        for (lon_range_min, lon_range_max) in lon_ranges:
            conditions = list(index_conditions)
            args = list(index_args)
            if lon_range_min is not None:
                conditions.append('r.lon_max >= ?')
                args.append(lon_range_min)
                conditions.append('r.lon_min <= ?')
                args.append(lon_range_max)

            conditions.extend(exact_conditions)
            args.extend(filter_args)

            if lon_range_min is not None:
                conditions.append('a.lon >= ? AND a.lon <= ?')
                args.extend([lon_range_min, lon_range_max])

            if magnitude_min is not None:
                conditions.append('r.magnitude_max >= ?')
                conditions.append('a.magnitude >= ?')
                args.extend([magnitude_min, magnitude_min])

            sql = select + '''
                WHERE ''' + (' AND '.join(conditions) or '1') + suffix

            results.extend(
                connection.execute(sql, args + list(suffix_args)))

        return results

    select_count = '''
        SELECT COUNT(*) FROM ''' + prefix + '''.events_rtree r
        JOIN ''' + prefix + '''.events a ON a.event_id = r.event_id '''

    select_min_magnitude = '''
        SELECT MIN(a.magnitude) FROM ''' + prefix + '''.events_rtree r
        JOIN ''' + prefix + '''.events a ON a.event_id = r.event_id '''

    # Lower the magnitude threshold until there are enough candidates. Steps
    # are estimated from the Gutenberg-Richter relation with b = 1. If this
    # takes long, there are probably not enough events in the region: the
    # smallest magnitude there is then used to stop early.
    magnitude_min = 10.
    if filter.magnitude_max is not None:
        magnitude_min = min(magnitude_min, filter.magnitude_max)

    magnitude_floor = None
    istep = 0
    while True:
        count = sum(
            row[0] for row in query(select_count, magnitude_min))

        if count >= n:
            break

        if count == 0:
            magnitude_min -= 1.0
        else:
            magnitude_min -= max(0.1, math.log10(n / count) + 0.1)

        istep += 1
        if magnitude_floor is None and istep >= 4:
            magnitudes = [
                row[0] for row in query(select_min_magnitude, None)
                if row[0] is not None]

            magnitude_floor = min(magnitudes) if magnitudes else 10.

        if filter.magnitude_min is not None \
                and magnitude_min <= filter.magnitude_min:
            magnitude_min = filter.magnitude_min
            break

        if magnitude_floor is not None and magnitude_min <= magnitude_floor:
            # All events, including the ones without magnitude.
            magnitude_min = None
            break

    select = _sql_select_events(prefix) + '''
        JOIN ''' + prefix + '''.events_rtree r ON r.event_id = a.event_id '''

    rows = query(
        select,
        magnitude_min,
        '''
        ORDER BY a.magnitude IS NULL, a.magnitude DESC, a.time DESC
        LIMIT ?''',
        (n,))

    events = [_row_to_event(row) for row in rows]
    events.sort(key=lambda ev: (
        ev.magnitude is None,
        -(ev.magnitude or 0.0),
        -ev.time))

    return events[:n]


def _sql_select_events(prefix):
    return '''
    SELECT a.elk_id, a.group_id, a.superseded_by, a.source, a.source_id,
        a.source_time_modified, a.source_version, a.time_created, a.time,
        a.lat, a.lon, a.north_shift, a.east_shift, a.depth, a.magnitude,
        a.magnitude_type, a.duration, a.region, a.tags, a.extras,
        c.mnn, c.mee, c.mdd, c.mne, c.mnd, c.med
    FROM ''' + prefix + '''.events a
    LEFT JOIN ''' + prefix + '''.moment_tensors c
    ON c.elk_id = a.elk_id
'''


def _row_to_event(row):
    (elk_id, group_id, superseded_by, source, source_id,
     source_time_modified, source_version, time_created,
     time, lat, lon, north_shift, east_shift, depth, magnitude,
     magnitude_type, duration, region, tags, extras,
     mnn, mee, mdd, mne, mnd, med) = row

    if mnn is not None:
        mt = pmt.as_mt((mnn, mee, mdd, mne, mnd, med))
    else:
        mt = None

    extras = json.loads(extras) if extras else {}
    extras.update(
        elk_id=elk_id,
        time_created=time_created,
        catalog_id=source_id,
        group_id=group_id)

    if source_time_modified is not None:
        extras['catalog_time_modified'] = source_time_modified

    if source_version is not None:
        extras['catalog_version'] = source_version

    if superseded_by is not None:
        extras['superseded_by'] = superseded_by

    ev = model.Event(
        name=elk_id,
        time=time,
        lat=lat,
        lon=lon,
        north_shift=north_shift,
        east_shift=east_shift,
        depth=depth,
        magnitude=magnitude,
        magnitude_type=magnitude_type or None,
        duration=duration,
        region=region,
        catalog=source,
        moment_tensor=mt,
        tags=json.loads(tags) if tags else [],
        extras=extras)

    return ev
