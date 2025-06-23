import os
import shutil
import tempfile
import unittest
import uuid

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
        self.t0 = util.str_to_time('2020-01-01 00:00:00')

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def make_db(self, name='events.elk'):
        conn = elk.connect(os.path.join(self.tempdir, name))
        elk.init_database(conn, 'main')
        return conn

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
        conn = elk.connect(os.path.join(self.tempdir, 'other.elk'))
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


if __name__ == '__main__':
    util.setup_logging('test_elk', 'warning')
    unittest.main()
