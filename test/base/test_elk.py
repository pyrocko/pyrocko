import os
import shutil
import tempfile
import unittest
import uuid
from unittest import mock

from pyrocko import util, model
from pyrocko import moment_tensor as pmt
from pyrocko.model.event import EventFilter
from pyrocko.squirrel import elk

km = 1000.


def make_event(catalog, name, time, magnitude=5.0, lat=10., lon=20., **kwargs):
    return model.Event(
        catalog=catalog,
        name=name,
        time=time,
        lat=lat,
        lon=lon,
        depth=10*km,
        magnitude=magnitude,
        **kwargs)


class ElkTestCase(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp(prefix='pyrocko-elk')
        # Cleanups run in reverse order, so this runs after all connections
        # have been closed (open files cannot be removed on Windows).
        self.addCleanup(shutil.rmtree, self.tempdir)
        self.t0 = util.str_to_time('2020-01-01 00:00:00')

    def connect(self, name='events.elk'):
        conn = elk.connect(os.path.join(self.tempdir, name))
        self.addCleanup(conn.close)
        return conn

    def make_db(self, name='events.elk'):
        conn = self.connect(name)
        elk.init_database(conn, 'main')
        return conn

    def make_squirrel(self):
        from pyrocko import squirrel
        from pyrocko.squirrel import database as dbm

        sq = squirrel.Squirrel(
            database=os.path.join(self.tempdir, 'squirrel.sqlite'))
        self.addCleanup(dbm.close_database, sq.get_database())
        return sq

    def get_events(self, conn, **kwargs):
        return dict(
            (ev.extras['catalog_id'], ev)
            for ev in elk.get_events(conn, 'main', **kwargs))

    def test_ids_and_duplicates(self):
        conn = self.make_db()
        events = [
            make_event('A', 'a%i' % i, self.t0 + i * 3600.)
            for i in range(10)]

        elk.add_events(conn, 'main', events)
        elk.add_events(conn, 'main', events + events[:3])

        evs = elk.get_events(conn, 'main')
        self.assertEqual(len(evs), 10)
        for ev in evs:
            self.assertEqual(uuid.UUID(ev.name).version, 7)
            self.assertEqual(ev.extras['elk_id'], ev.name)

    def test_content_hash_quantization(self):
        args = ('A', 'a1', self.t0 + 0.1, 10.1, 20.1, 0., 0., 10300., 5.1,
                'mb')

        h = elk.content_hash(*args)
        noisy = list(args)
        noisy[3] = 10.1 + 1e-12
        noisy[7] = 10.3 * 1000.
        self.assertEqual(h, elk.content_hash(*noisy))

        changed = list(args)
        changed[8] = 5.2
        self.assertNotEqual(h, elk.content_hash(*changed))

    def test_grouping(self):
        conn = self.make_db()
        events_a = [
            make_event('A', 'a%i' % i, self.t0 + i * 3600.)
            for i in range(5)]

        events_b = [
            make_event('B', 'b%i' % i, self.t0 + i * 3600. + 1.0,
                       magnitude=5.2, lat=10.05)
            for i in range(5)]

        elk.add_events(conn, 'main', events_a)
        elk.add_events(conn, 'main', events_b)

        evs = self.get_events(conn)
        for i in range(5):
            a = evs['a%i' % i]
            b = evs['b%i' % i]
            self.assertEqual(a.extras['group_id'], a.name)
            self.assertEqual(b.extras['group_id'], a.name)

        self.assertEqual(
            len(set(ev.extras['group_id'] for ev in evs.values())), 5)

    def test_group_join(self):
        conn = self.make_db()
        a = make_event('A', 'a', self.t0, lat=10.0)
        c = make_event('C', 'c', self.t0, lat=11.6)
        b = make_event('B', 'b', self.t0, lat=10.8)

        elk.add_events(conn, 'main', [a])
        elk.add_events(conn, 'main', [c])
        evs = self.get_events(conn)
        self.assertNotEqual(
            evs['a'].extras['group_id'], evs['c'].extras['group_id'])

        elk.add_events(conn, 'main', [b])
        evs = self.get_events(conn)
        self.assertEqual(
            set(ev.extras['group_id'] for ev in evs.values()),
            {evs['a'].name})

    def test_missing_magnitude(self):
        conn = self.make_db()
        elk.add_events(conn, 'main', [
            make_event('A', 'a', self.t0),
            make_event('B', 'b', self.t0, magnitude=None)])

        evs = self.get_events(conn)
        self.assertEqual(evs['b'].extras['group_id'], evs['b'].name)
        self.assertEqual(evs['a'].extras['group_id'], evs['a'].name)

    def test_roundtrip(self):
        conn = self.make_db()
        mt = pmt.MomentTensor(strike=10., dip=20., rake=30., magnitude=5.)
        ev = make_event(
            'A', 'a', self.t0, duration=2.0, region='Somewhere',
            tags=['x:1', 'y, z'], extras={'foo': 'bar'}, moment_tensor=mt)

        elk.add_events(conn, 'main', [ev])
        ev2 = elk.get_events(conn, 'main')[0]
        self.assertEqual(ev2.catalog, 'A')
        self.assertEqual(ev2.extras['catalog_id'], 'a')
        self.assertEqual(ev2.duration, 2.0)
        self.assertEqual(ev2.region, 'Somewhere')
        self.assertEqual(ev2.tags, ['x:1', 'y, z'])
        self.assertEqual(ev2.extras['foo'], 'bar')
        self.assertTrue(
            abs(ev2.moment_tensor.moment - mt.moment) / mt.moment < 1e-9)

    def test_superseded(self):
        conn = self.make_db()
        elk.add_events(conn, 'main', [make_event('A', 'a', self.t0)])
        old = elk.get_events(conn, 'main')[0]

        elk.add_events(
            conn, 'main', [make_event('A', 'a', self.t0 + 1.0, magnitude=5.1)])

        evs = elk.get_events(conn, 'main')
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0].magnitude, 5.1)
        self.assertEqual(evs[0].extras['group_id'], old.name)

        evs = elk.get_events(conn, 'main', include_superseded=True)
        self.assertEqual(len(evs), 2)
        old = [ev for ev in evs if ev.name == old.name][0]
        self.assertEqual(old.extras['superseded_by'], evs[0].name)

    def _revision(self, magnitude, tmodified, version=None):
        extras = {'catalog_time_modified': self.t0 + tmodified}
        if version is not None:
            extras['catalog_version'] = version

        return make_event(
            'A', 'a', self.t0, magnitude=magnitude, extras=extras)

    def test_superseded_by_source_time(self):
        conn = self.make_db()

        # newer revision arrives first, older one later
        elk.add_events(conn, 'main', [self._revision(5.1, 200., 'v2')])
        elk.add_events(conn, 'main', [self._revision(5.0, 100., 'v1')])

        evs = elk.get_events(conn, 'main')
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0].magnitude, 5.1)
        self.assertEqual(
            evs[0].extras['catalog_time_modified'], self.t0 + 200.)
        self.assertEqual(evs[0].extras['catalog_version'], 'v2')

    def test_reissued_revision(self):
        conn = self.make_db()

        # catalog goes back to an earlier solution: A -> B -> A
        elk.add_events(conn, 'main', [self._revision(5.0, 100.)])
        elk.add_events(conn, 'main', [self._revision(5.1, 200.)])
        self.assertEqual(elk.get_events(conn, 'main')[0].magnitude, 5.1)

        elk.add_events(conn, 'main', [self._revision(5.0, 300.)])
        evs = elk.get_events(conn, 'main')
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0].magnitude, 5.0)
        self.assertEqual(
            evs[0].extras['catalog_time_modified'], self.t0 + 300.)

        self.assertEqual(
            len(elk.get_events(conn, 'main', include_superseded=True)), 2)

    def test_revision_info_not_in_extras_column(self):
        conn = self.make_db()
        elk.add_events(conn, 'main', [self._revision(5.0, 100., 'v1')])
        (extras,) = conn.execute('SELECT extras FROM events').fetchone()
        self.assertEqual(extras, '{}')

    def test_merge_keeps_older_ids(self):
        events = [
            make_event('A', 'a%i' % i, self.t0 + i * 3600.)
            for i in range(3)]

        conn_old = self.make_db('old.elk')
        elk.add_events(conn_old, 'main', events)
        ids_old = dict(
            (k, ev.name) for (k, ev) in self.get_events(conn_old).items())

        conn_new = self.make_db('new.elk')
        elk.add_events(conn_new, 'main', events)
        ids_new = dict(
            (k, ev.name) for (k, ev) in self.get_events(conn_new).items())

        self.assertTrue(all(ids_old[k] < ids_new[k] for k in ids_old))

        # merge old into new: older ids win, newer ones become aliases
        elk.add_events(conn_new, 'main', elk.get_events(conn_old, 'main'))
        evs = self.get_events(conn_new)
        self.assertEqual(len(evs), 3)
        for k, ev in evs.items():
            self.assertEqual(ev.name, ids_old[k])
            self.assertEqual(ev.extras['group_id'], ids_old[k])
            self.assertEqual(
                elk.resolve_elk_id(conn_new, 'main', ids_new[k]), ids_old[k])

        # merge new into old: nothing changes, aliases registered
        elk.add_events(conn_old, 'main', elk.get_events(conn_new, 'main'))
        self.assertEqual(
            dict(
                (k, ev.name) for (k, ev) in self.get_events(conn_old).items()),
            ids_old)

    def test_schema_check(self):
        conn = self.connect('other.elk')
        with self.assertRaises(elk.ElkError):
            elk.get_events(conn, 'main')

    def test_filter(self):
        conn = self.make_db()
        elk.add_events(conn, 'main', [
            make_event('A', 'a%i' % i, self.t0 + i * 3600., magnitude=4.+i)
            for i in range(5)])

        filter = EventFilter(
            magnitude_min=5.0, time_max=self.t0 + 3.5 * 3600.)

        names = sorted(
            ev.extras['catalog_id']
            for ev in elk.get_events(conn, 'main', filter=filter))
        self.assertEqual(names, ['a1', 'a2', 'a3'])

        condition = filter.get_condition()
        evs = [
            make_event('A', 'a%i' % i, self.t0 + i * 3600., magnitude=4.+i)
            for i in range(5)]
        self.assertEqual(
            [ev.name for ev in evs if condition(ev)], ['a1', 'a2', 'a3'])

    def _add_grouped(self, conn, n):
        # Groups of two events from catalogs A and B, B with larger
        # magnitude.
        elk.add_events(conn, 'main', [
            make_event('A', 'a%i' % i, self.t0 + i * 3600.)
            for i in range(n)])
        elk.add_events(conn, 'main', [
            make_event('B', 'b%i' % i, self.t0 + i * 3600. + 1.0,
                       magnitude=5.2, lat=10.05)
            for i in range(n)])

    def test_event_groups_paging(self):
        conn = self.make_db()
        self._add_grouped(conn, 7)

        keys = []
        after = None
        while True:
            page = elk.get_event_groups(conn, 'main', limit=3, after=after)
            if not page:
                break

            self.assertLessEqual(len(page), 3)
            for key, evs in page:
                self.assertEqual(
                    sorted(ev.catalog for ev in evs), ['A', 'B'])
                self.assertEqual(
                    set(ev.extras['group_id'] for ev in evs), {key[1]})
                keys.append(key)

            after = page[-1][0]

        self.assertEqual(len(keys), 7)
        self.assertEqual(keys, sorted(keys, reverse=True))
        self.assertEqual(
            [time for (time, _) in keys],
            [self.t0 + i * 3600. for i in reversed(range(7))])

    def test_event_groups_filter(self):
        conn = self.make_db()
        self._add_grouped(conn, 3)
        elk.add_events(conn, 'main', [
            make_event('A', 'single', self.t0 + 10 * 3600.)])

        groups = elk.get_event_groups(
            conn, 'main', filter=EventFilter(magnitude_min=5.1))

        # Only groups with matching events, only matching events in groups.
        self.assertEqual(len(groups), 3)
        for _, evs in groups:
            self.assertEqual([ev.catalog for ev in evs], ['B'])

        groups = elk.get_event_groups(
            conn, 'main', filter=EventFilter(
                time_min=self.t0 + 0.5 * 3600.,
                time_max=self.t0 + 1.5 * 3600.))

        self.assertEqual(len(groups), 1)
        self.assertEqual(
            sorted(ev.extras['catalog_id'] for ev in groups[0][1]),
            ['a1', 'b1'])

    def test_event_groups_superseded(self):
        conn = self.make_db()
        elk.add_events(conn, 'main', [make_event('A', 'a', self.t0)])
        elk.add_events(
            conn, 'main', [make_event('A', 'a', self.t0 + 1.0, magnitude=5.1)])

        groups = elk.get_event_groups(conn, 'main')
        self.assertEqual(len(groups), 1)
        self.assertEqual([ev.magnitude for ev in groups[0][1]], [5.1])

        groups = elk.get_event_groups(conn, 'main', include_superseded=True)
        self.assertEqual(len(groups[0][1]), 2)

    def _random_events(self, n, seed=0, magnitude_min=3.0):
        import numpy as num
        rstate = num.random.RandomState(seed)
        events = []
        for i in range(n):
            # Gutenberg-Richter distributed magnitudes (b = 1).
            magnitude = magnitude_min \
                + rstate.exponential(1.0 / num.log(10.))
            events.append(make_event(
                'A', 'r%i' % i,
                self.t0 + rstate.uniform(0., 365. * 24. * 3600.),
                magnitude=None if i % 50 == 0 else float(magnitude),
                lat=float(rstate.uniform(-90., 90.)),
                lon=float(rstate.uniform(-180., 180.))))

        return events

    def _largest_brute_force(
            self, events, n, filter=None, lat_min=None, lat_max=None,
            lon_min=None, lon_max=None):

        condition = (filter or EventFilter()).get_condition()

        def in_lon(lon):
            if lon_min is None and lon_max is None:
                return True
            lo = -180. if lon_min is None else lon_min
            hi = 180. if lon_max is None else lon_max
            if lo <= hi:
                return lo <= lon <= hi
            else:
                return lon >= lo or lon <= hi

        selected = [
            ev for ev in events
            if condition(ev)
            and (lat_min is None or ev.lat >= lat_min)
            and (lat_max is None or ev.lat <= lat_max)
            and in_lon(ev.lon)]

        selected.sort(key=lambda ev: (
            ev.magnitude is None, -(ev.magnitude or 0.0), -ev.time))

        return [ev.name for ev in selected[:n]]

    def test_largest_events(self):
        conn = self.make_db()
        events = self._random_events(3000)
        elk.add_events(conn, 'main', events)

        tday = 24. * 3600.
        cases = [
            dict(n=10),
            dict(n=100),
            dict(n=5000),
            dict(n=20, filter=EventFilter(
                time_min=self.t0 + 30 * tday, time_max=self.t0 + 60 * tday)),
            dict(n=20, filter=EventFilter(
                time_min=self.t0 + 30 * tday, time_max=self.t0 + 30.1 * tday)),
            dict(n=50, lat_min=-10., lat_max=40., lon_min=0., lon_max=60.),
            dict(n=50, lon_min=150., lon_max=-150.),
            dict(n=50, filter=EventFilter(magnitude_min=4.0)),
            dict(n=50, filter=EventFilter(
                magnitude_min=3.2, magnitude_max=3.3)),
            dict(n=50, filter=EventFilter(magnitude_max=3.5)),
            dict(n=10, filter=EventFilter(magnitude_min=20.)),
        ]

        for kwargs in cases:
            expected = self._largest_brute_force(events, **kwargs)
            got = [
                ev.extras['catalog_id']
                for ev in elk.get_largest_events(conn, 'main', **kwargs)]

            self.assertEqual(got, expected, kwargs)

        # All events, the ones without magnitude last.
        evs = elk.get_largest_events(conn, 'main', 5000)
        self.assertEqual(len(evs), 3000)
        self.assertTrue(all(ev.magnitude is None for ev in evs[-60:]))

    def test_largest_events_micro(self):
        # Small magnitudes, e.g. icequakes, must work as well.
        conn = self.make_db()
        events = self._random_events(1000, magnitude_min=-1.0)
        elk.add_events(conn, 'main', events)
        for n in [10, 100, 2000]:
            expected = self._largest_brute_force(events, n)
            got = [
                ev.extras['catalog_id']
                for ev in elk.get_largest_events(conn, 'main', n)]

            self.assertEqual(got, expected, n)

    def test_largest_events_superseded(self):
        conn = self.make_db()
        elk.add_events(conn, 'main', [
            make_event('A', 'a', self.t0, magnitude=7.0),
            make_event('A', 'b', self.t0 + 3600., magnitude=6.0)])
        elk.add_events(conn, 'main', [
            make_event('A', 'a', self.t0, magnitude=5.0)])

        evs = elk.get_largest_events(conn, 'main', 10)
        self.assertEqual(
            [(ev.extras['catalog_id'], ev.magnitude) for ev in evs],
            [('b', 6.0), ('a', 5.0)])

    def test_superseded_same_time_created(self):
        # With coarse clock resolution (Windows), revisions added shortly
        # after each other get the same time_created.
        conn = self.make_db()
        with mock.patch.object(elk.time, 'time', return_value=self.t0):
            elk.add_events(conn, 'main', [
                make_event('A', 'a', self.t0, magnitude=7.0)])
            elk.add_events(conn, 'main', [
                make_event('A', 'a', self.t0, magnitude=5.0)])

        evs = elk.get_events(conn, 'main')
        self.assertEqual([ev.magnitude for ev in evs], [5.0])

    def test_rtree_after_vacuum(self):
        conn = self.make_db()
        events = self._random_events(500)
        elk.add_events(conn, 'main', events[:250])
        elk.add_events(conn, 'main', events[250:])
        expected = [
            ev.extras['catalog_id']
            for ev in elk.get_largest_events(conn, 'main', 30)]

        conn.execute('VACUUM')
        conn.close()

        conn = self.connect()
        n_rtree, = conn.execute(
            'SELECT COUNT(*) FROM events_rtree').fetchone()
        self.assertEqual(n_rtree, 500)
        got = [
            ev.extras['catalog_id']
            for ev in elk.get_largest_events(conn, 'main', 30)]
        self.assertEqual(got, expected)

    def test_squirrel_largest_events(self):
        conn = self.make_db()
        elk.add_events(conn, 'main', [
            make_event('A', 'a%i' % i, self.t0 + i * 3600., magnitude=4. + i)
            for i in range(4)])
        conn.close()

        plain_path = os.path.join(self.tempdir, 'events.yaml')
        model.dump_events([
            make_event('C', 'c%i' % i, self.t0 + i * 3600. + 1800.,
                       magnitude=4.5 + i, lon=-170.)
            for i in range(3)], plain_path)

        sq = self.make_squirrel()
        sq.add([os.path.join(self.tempdir, 'events.elk'), plain_path])

        self.assertEqual(
            [ev.magnitude for ev in sq.get_largest_events(3)],
            [7.0, 6.5, 6.0])

        self.assertEqual(
            [ev.magnitude for ev in sq.get_largest_events(
                10, tmax=self.t0 + 7000.)],
            [5.5, 5.0, 4.5, 4.0])

        self.assertEqual(
            [ev.magnitude for ev in sq.get_largest_events(
                10, lon_min=170., lon_max=-160.)],
            [6.5, 5.5, 4.5])

    def test_squirrel_event_groups(self):
        path = os.path.join(self.tempdir, 'events.elk')
        conn = self.make_db()
        self._add_grouped(conn, 4)
        conn.close()

        plain_path = os.path.join(self.tempdir, 'events.yaml')
        model.dump_events([
            make_event('C', 'c%i' % i, self.t0 + i * 3600. + 1800.)
            for i in range(3)], plain_path)

        sq = self.make_squirrel()
        sq.add([path, plain_path])

        sizes = []
        after = None
        while True:
            page = sq.get_event_groups(limit=2, after=after)
            if not page:
                break

            sizes.extend(len(evs) for _, evs in page)
            after = page[-1][0]

        # Elk groups of two, interleaved with plain events of their own.
        self.assertEqual(sizes, [2, 1, 2, 1, 2, 1, 2])

        groups = sq.get_event_groups(
            tmin=self.t0 + 1000., tmax=self.t0 + 4000.)
        self.assertEqual(
            [[ev.catalog for ev in evs] for _, evs in groups],
            [['B', 'A'], ['C']])


if __name__ == '__main__':
    util.setup_logging('test_elk', 'warning')
    unittest.main()
