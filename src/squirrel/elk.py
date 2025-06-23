import time
import sqlite3
import json
import struct
import logging
import math

import numpy as num
import requests

from pyrocko import util, model
from pyrocko import orthodrome
from pyrocko import moment_tensor as pmt

logger = logging.getLogger('psq.elk')

km = 1000.


def unpack_application_id(b):
    return struct.unpack('>i', b)[0]


ELK1 = unpack_application_id(b'ELK1')


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

    norm_magnitude = 1.0
    norm_time, norm_distance = scaling(min(a_magnitude, b_magnitude))
    norm_time *= 10
    norm_distance *= 50

    d_time = abs(a_time - b_time)
    d_magnitude = abs(a_magnitude - b_magnitude)
    distance = surface_distance(
        a_lat, a_lon, a_north, a_east, b_lat, b_lon, b_north, b_east)

    d = d_time / norm_time \
        + d_magnitude / norm_magnitude \
        + distance / norm_distance

    print(d)

    return d


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


class ElkError(Exception):
    pass


def register_elk_ids(n=1):
    registrations = []
    while n > 0:
        nthis = min(1000, n)
        try:
            r = requests.get(
                'https://elk.pyrocko.org/register', params=dict(n=nthis))
            r.raise_for_status()
            registrations.extend(r.json())

        except (requests.RequestException) as e:
            raise ElkError(
                'Failed to obtain elk IDs. %s' % str(e))

        n -= nthis

    return registrations


def init_database(connection, prefix):
    cursor = connection.cursor()

    cursor.execute('''
        PRAGMA %s.application_id = %i
    ''' % (prefix, ELK1))

    cursor.execute('''
        CREATE TABLE %s.events (
            elk_id INTEGER PRIMARY KEY,
            group_id INTEGER,
            sequence_id INTEGER,
            source TEXT,
            source_id TEXT,
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
            elk_id INTEGER PRIMARY KEY,
            mnn REAL,
            mee REAL,
            mdd REAL,
            mne REAL,
            mnd REAL,
            med REAL )
    ''' % prefix)

    cursor.execute('''
        CREATE INDEX %s.index_time ON events ( time )
    ''' % prefix)

    cursor.execute('''
        CREATE INDEX %s.index_group_id ON events ( group_id )
    ''' % prefix)

    cursor.execute('''
        CREATE UNIQUE INDEX %s.index_unique ON events (
            source,
            source_id,
            time,
            lat,
            lon,
            north_shift,
            east_shift,
            depth,
            magnitude,
            magnitude_type )
    ''' % prefix)

    connection.commit()


def serialize_extras(extras):
    return json.dumps(
        dict(
            (k, v)
            for (k, v)
            in extras.items()
            if k not in ('elk_id', 'group_id', 'sequence_id')))


def add_events(connection, prefix, events):
    events = sorted(events, key=lambda ev: ev.time)

    time_created = time.time()

    cursor = connection.cursor()

    new_events = []
    have_in_input = set()
    n_input_duplicates = 0
    n_database_duplicates = 0
    for ev in events:
        key = (
            ev.catalog or '',
            ev.name or '',
            ev.time,
            ev.lat,
            ev.lon,
            ev.north_shift,
            ev.east_shift,
            ev.depth,
            ev.magnitude,
            ev.magnitude_type or '')

        if key in have_in_input:
            n_input_duplicates += 1
            continue

        sql = '''
            SELECT elk_id from %s.events WHERE
                source IS ? AND
                source_id IS ? AND
                time IS ? AND (
                    lat IS ? AND
                    lon IS ? AND
                    north_shift IS ? AND
                    east_shift IS ? AND
                    depth IS ? AND
                    magnitude IS ? AND
                    magnitude_type IS ?
                )
''' % prefix

        res = list(cursor.execute(sql, key))
        if not res:
            new_events.append(ev)
            have_in_input.add(key)
        else:
            n_database_duplicates += 1

    if not new_events:
        logger.info('No new events. Nothing to store.')
        return

    logger.info(
        'Inserting %i new events (%i already in database, %i duplicates in '
        'input)' % (
            len(new_events),
            n_database_duplicates,
            n_input_duplicates))

    logger.info(
        'Registering Elk IDs for %i new events.' % len(new_events))

    elk_ids = [int(x['elk_id'][3:]) for x in register_elk_ids(len(new_events))]

    cursor.executemany(
        'INSERT INTO %s.moment_tensors VALUES (%s)' % (
            prefix, ','.join('?' * 7)), (
                (elk_id,) + tuple(event.moment_tensor.m6())
                for (elk_id, event)
                in zip(elk_ids, new_events)
                if event.moment_tensor is not None))

    cursor.executemany(
        'INSERT INTO %s.events VALUES (%s)' % (
            prefix, ','.join('?'*19)), ((
                elk_id,
                -1,
                -1,
                event.catalog or '',
                event.name or '',
                time_created,
                '',
                event.time,
                event.lat,
                event.lon,
                event.north_shift,
                event.east_shift,
                event.depth,
                event.magnitude,
                event.magnitude_type or '',
                event.duration,
                event.region,
                ', '.join(event.tags),
                serialize_extras(event.extras))
            for (elk_id, event) in zip(elk_ids, new_events)))

    # update group_id

    res = cursor.execute('''
        SELECT elk_id, time, lat, lon, north_shift, east_shift, depth,
               magnitude
        FROM %s.events
        WHERE group_id = -1
        ORDER BY elk_id
    ''' % prefix)

    for (elk_id, etime, lat, lon, north_shift, east_shift, depth,
         magnitude) in res:

        delta_time, _ = scaling(magnitude)
        delta_time *= 10.
        res = cursor.execute('''
            SELECT group_id
            FROM %s.events
            WHERE
                time > ? AND
                time < ? AND
                group_id != -1 AND
                elk_distance(
                    time, magnitude, lat, lon, north_shift, east_shift,
                    ?, ?, ?, ?, ?, ?) < 1
            ORDER BY elk_id
        ''' % prefix, (
            etime - delta_time,
            etime + delta_time,
            etime,
            magnitude,
            lat,
            lon,
            north_shift,
            east_shift))

        group_ids = [x[0] for x in res]

        group_ids.append(elk_id)
        group_ids = sorted(set(group_ids))
        group_id = group_ids[0]

        cursor.execute('''
            UPDATE %s.events SET group_id = ? WHERE elk_id = ?
        ''' % prefix, (group_id, elk_id))

        if len(group_ids) > 0:
            cursor.execute('''
                UPDATE %s.events SET group_id = ? WHERE group_id IN ( %s )
            ''' % (prefix, ','.join('?'*(len(group_ids)-1))), group_ids)

    cursor.close()
    connection.commit()


def get_events(
        connection,
        prefix,
        nlatest=None,
        filter=None):

    '''
    Query events.

    Limits are treated as half-open intervals, e.g. [tmin, tmax).
    '''

    if isinstance(filter, dict):
        filter = model.event.EventFilter(**filter)

    cursor = connection.cursor()
    where, args = filter.sql_condition()

    if nlatest is not None:
        limit = 'LIMIT ?'
        args.append(nlatest)
    else:
        limit = ''

    sql = '''
    SELECT a.elk_id, a.group_id, a.source, a.source_id, a.time_created,
        a.time, a.lat, a.lon, a.north_shift, a.east_shift, a.depth,
        a.magnitude, a.magnitude_type, a.region,
        c.mnn, c.mee, c.mdd, c.mne, c.mnd, c.med
    FROM ''' + prefix + '''.events a
    LEFT JOIN ''' + prefix + '''.moment_tensors c
    ON c.elk_id = a.elk_id
''' + where + '''
    ORDER BY ( SELECT -time FROM ''' + prefix + '''.events b
               WHERE b.elk_id = a.group_id ),
               -a.time
''' + limit

    events = []
    for row in cursor.execute(sql, args):
        (elk_id, group_id, source, source_id, time_created, time, lat, lon,
         north_shift, east_shift, depth, magnitude, magnitude_type, region,
         mnn, mee, mdd, mne, mnd, med) = row

        if mnn is not None:
            mt = pmt.as_mt((mnn, mee, mdd, mne, mnd, med))
        else:
            mt = None

        ev = model.Event(
            name='elk%i' % elk_id,
            time=time,
            lat=lat,
            lon=lon,
            north_shift=north_shift,
            east_shift=east_shift,
            depth=depth,
            magnitude=magnitude,
            magnitude_type=magnitude_type or None,
            region=region,
            catalog=source,
            moment_tensor=mt,
            extras=dict(
                time_created=time_created,
                catalog_id=source_id,
                group_id='elk%i' % group_id))

        events.append(ev)

    cursor.close()

    return events
