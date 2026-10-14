import shutil
import tempfile
from pyrocko import util
from pyrocko.util import HTTPError, URLErrorSSL
from pyrocko.client import catalog, globalcmt
from pyrocko import moment_tensor
import unittest
import requests
from .. import common


def near(a, b, eps):
    return abs(a-b) < eps


def is_the_haiti_event_geofon(ev):
    assert ev.name == 'gfz2010avtm'
    assert ev.catalog == 'GEOFON'
    assert near(ev.magnitude, 7.2, 0.1)
    assert near(ev.lat, 18.36, 0.01)
    assert near(ev.lon, -72.57, 0.01)
    assert near(ev.depth, 20000., 500.)
    assert ev.region == 'Haiti Region'
    assert 'evaluation_mode:manual' in ev.tags
    assert 'geofon_status:M' in ev.tags
    assert ev.extras['catalog_time_modified'] > util.stt('2010-01-12 00:00:00')


# Real NDK records: old solution (O), record without hypocenter catalog code,
# merged numbers in CENTROID line, timestamp with space for leading zero,
# standard solution (Haiti 2010), quick solution (Q).
NDK_SAMPLE = '\n'.join([
    'MLI  1976/01/01 01:29:39.6 -28.61 -177.64  59.0 6.2 0.0 '
    'KERMADEC ISLANDS REGION',
    'M010176A         B:  0    0   0 S:  0    0   0 M: 12   3'
    '0 135 CMT: 1 BOXHD:  9.4',
    'CENTROID:     13.8 0.2 -29.25 0.02 -176.96 0.01  47.8  0'
    '.6 FREE O-00000000000000',
    '26  7.680 0.090  0.090 0.060 -7.770 0.070  1.390 0.160  '
    '4.520 0.160 -3.260 0.060',
    'V10   8.940 75 283   1.260  2  19 -10.190 15 110   9.560'
    ' 202 30   93  18 60   88',
    '     1979/03/15 12:52:29.3  23.16  101.14  33.0 5.6 6.2 '
    'YUNNAN PROV, CHINA',
    'B031579B         B:  8   22  45 S:  0    0   0 M:  0    '
    '0   0 CMT: 1 BOXHD:  4.1',
    'CENTROID:      1.1 0.2  23.24 0.04  101.61 0.06  10.0  0'
    '.0 FIX  O-00000000000000',
    '25  0.235 0.045 -1.217 0.037  0.983 0.057 -0.706 0.145  '
    '0.094 0.080 -0.401 0.044',
    'V10   1.120 17 255   0.430 62 129  -1.550 21 352   1.330'
    '  33 62   -3 124 87 -152',
    'PDE  1996/11/20 19:42:56.1  10.30  127.43  33.0 5.6 5.3 '
    'PHILIPPINE ISLANDS REGIO',
    'B112096E         B: 35   58  45 S:  0    0   0 M:  0    '
    '0   0 CMT: 1 BOXHD: 22.0',
    'CENTROID:      1.060.0  10.47 0.03  127.42 0.03  15.0  0'
    '.0 BDY  O-00000000000000',
    '24 -4.983 0.133 -0.092 0.153  5.075 0.173 -0.433 0.443  '
    '0.715 0.469 -2.898 0.141',
    'V10   6.430  4 246  -1.390  2 156  -5.050 86  45   5.740'
    ' 338 41  -88 154 49  -92',
    'PDE  2005/01/21 13:45:14.1  -1.13  -80.83  10.0 5.9 5.9 '
    'NEAR COAST OF ECUADOR',
    'C200501211345A   B: 64  149  40 S: 74  177  50 M: 57   7'
    '8 125 CMT: 1 TRIHD:  2.4',
    'CENTROID:      3.8 0.1  -1.23 0.01  -81.05 0.01  12.0  0'
    '.0 BDY  S-20050419 94323',
    '25  0.679 0.006  0.027 0.006 -0.706 0.007  0.060 0.017 -'
    '0.982 0.017  0.000 0.005',
    'V10   1.191 63  84   0.025  2 178  -1.216 27 269   1.203'
    '   3 18   96 177 72   88',
    'PDEW 2010/01/12 21:53:10.2  18.45  -72.55  13.0 0.0 7.0 '
    'HAITI REGION',
    'C201001122153A   B: 96  262  50 S: 91  235  50 M: 97  22'
    '0 150 CMT: 1 TRIHD:  8.1',
    'CENTROID:      6.9 0.1  18.61 0.00  -72.62 0.01  12.0  0'
    '.0 FIX  S-20100319115423',
    '26  1.740 0.019 -3.870 0.016  2.130 0.019 -0.531 0.100 -'
    '1.060 0.097  2.740 0.014',
    'V10   3.853 29 112   1.082 61 289  -4.935  1  21   4.394'
    ' 152 69  159 250 71   22',
    'PDEQ 2026/01/01 01:53:01.7 -45.67   96.32  10.0 0.0 6.0 '
    'SOUTHEAST INDIAN RIDGE',
    'C202601010153A   B: 84  167  40 S: 74  140  50 M: 43   6'
    '3 125 CMT: 1 TRIHD:  2.5',
    'CENTROID:      5.8 0.1 -45.57 0.01   96.09 0.01  12.0  0'
    '.0 FIX  Q-20251231223117',
    '25 -0.003 0.010 -1.380 0.012  1.380 0.011 -0.037 0.027  '
    '0.058 0.028 -0.431 0.011',
    'V10   1.448  2 261  -0.005 87 105  -1.446  1 351   1.447'
    '  36 87    1 306 89  177',
])


class CatalogTestCase(unittest.TestCase):

    @common.require_internet
    @common.skip_on(requests.ConnectionError)
    def testGeofon(self):

        cat = catalog.Geofon()

        tmin = util.str_to_time('2010-01-12 21:50:00')
        tmax = util.str_to_time('2010-01-13 03:17:00')

        names = cat.get_event_names(
            time_range=(tmin, tmax), nmax=10, magmin=5.)

        assert len(names) > 0
        ident = None
        for name in names:
            ev = cat.get_event(name)
            if ev.magnitude >= 7:
                is_the_haiti_event_geofon(ev)
                ident = ev.name

        assert ident is not None

        cat.flush()

    @common.require_internet
    @common.skip_on(requests.ConnectionError)
    def testGeofonMomentMagnitude(self):
        # Preferred magnitude is mb 5.61 of an automatic origin, but there is
        # a confirmed moment tensor solution with Mw 5.95.
        cat = catalog.Geofon()
        ev = cat.get_event('gfz2017afjw')
        assert ev.magnitude_type == 'Mw'
        assert near(ev.magnitude, 5.95, 0.01)
        assert 'geofon_status:C' in ev.tags

        tmin = util.str_to_time('2017-01-03 22:00:00')
        tmax = util.str_to_time('2017-01-03 23:00:00')
        names = cat.get_event_names(time_range=(tmin, tmax), magmin=5.9)
        assert names == ['gfz2017afjw']
        cat.flush()

    @common.require_internet
    @common.skip_on(requests.ConnectionError)
    def testGeofonSingleEvent(self):
        cat = catalog.Geofon()
        ev = cat.get_event('gfz2010avtm')
        is_the_haiti_event_geofon(ev)
        cat.flush()

        ev = cat.get_event('gfz2019hkck')
        assert ev.name == 'gfz2019hkck'

    @common.require_internet
    @common.skip_on(requests.ConnectionError)
    def testGeofonMT(self):
        cat = catalog.Geofon()
        tmin = util.str_to_time('2014-01-01 00:00:00')
        tmax = util.str_to_time('2017-01-01 00:00:00')
        events_a = cat.get_events((tmin, tmax), magmin=8)
        events_b = [
            cat.get_event('gfz2015sfdd'),
            cat.get_event('gfz2014gkgf')]

        for events in [events_a, events_b]:
            self.assertEqual(len(events), 2)
            mt1, mt2 = [ev.moment_tensor for ev in events]
            angle = moment_tensor.kagan_angle(mt1, mt2)
            self.assertEqual(round(angle - 7.7, 1), 0.0)

        ev = cat.get_event('gfz2020vimx')
        assert isinstance(ev.moment_tensor, moment_tensor.MomentTensor)

    @unittest.skip("don't want to annoy our colleagues at GEOFON")
    @common.skip_on(requests.ConnectionError)
    def testGeofonMassive(self):
        cat = catalog.Geofon(get_moment_tensors=False)
        tmin = util.str_to_time('2010-01-01 00:00:00')
        tmax = util.str_to_time('2018-01-01 00:00:00')
        events = cat.get_events((tmin, tmax))
        print(len(events))

    @common.require_internet
    @common.skip_on_vagrant
    @common.skip_on_ci
    @common.skip_on(HTTPError)
    @common.skip_on(catalog.ISCBlocked)
    def testISC(self):
        cat = catalog.ISC()
        tmin = util.stt('2009-02-22 05:00:00')
        tmax = util.stt('2009-02-22 19:00:00')
        events = cat.get_events((tmin, tmax), magmin=5)

        ev = events[-1]
        cat.flush()
        ev2 = cat.get_event(ev.name)
        self.assertEqual(ev2.name, ev.name)
        self.assertEqual(ev2.time, ev.time)
        self.assertEqual(ev2.magnitude, ev.magnitude)

        with self.assertRaises(catalog.NotFound):
            cat.get_event('smi:ISC/evid=99999999999')

        markers = cat.get_phase_markers(
            (tmin, tmax), phases=['XXX'], station_codes=['XXXXX', 'YYYYY'])

        self.assertEqual(len(markers), 0)

        markers = cat.get_phase_markers(
            (tmin, tmax), phases=['P', 'PcP'], station_codes=['WRA'])

        self.assertEqual(len(markers), 76)

    @common.require_internet
    @common.skip_on(HTTPError)
    def testGlobalCMT(self):

        def is_the_haiti_event(ev):
            assert near(ev.magnitude, 7.0, 0.1)
            assert near(ev.lat, 18.61, 0.01)
            assert near(ev.lon, -72.62, 0.01)
            assert near(ev.depth, 12000., 1.)
            assert ev.region.lower() == 'haiti region'

        cat = catalog.GlobalCMT()

        tmin = util.str_to_time('2010-01-12 21:50:00')
        tmax = util.str_to_time('2010-01-13 03:17:00')

        names = cat.get_event_names(time_range=(tmin, tmax), magmin=5.)
        ident = None
        for name in names:
            ev = cat.get_event(name)
            if ev.magnitude > 7:
                is_the_haiti_event(ev)
                ident = ev.name

        assert ident is not None
        cat.flush()
        ev = cat.get_event(ident)
        is_the_haiti_event(ev)

    @common.require_internet
    @common.skip_on(HTTPError, URLErrorSSL)
    def testUSGS(self):

        def is_the_haiti_event(ev):
            assert near(ev.magnitude, 7.0, 0.1)
            assert near(ev.lat, 18.443, 0.01)
            assert near(ev.lon, -72.571, 0.01)
            assert near(ev.depth, 13000., 1.)

        cat = catalog.USGS()

        tmin = util.str_to_time('2010-01-12 21:50:00')
        tmax = util.str_to_time('2010-01-13 03:17:00')

        names = cat.get_event_names(time_range=(tmin, tmax), magmin=5.)
        assert len(names) == 13
        for name in names:
            ev = cat.get_event(name)
            if ev.magnitude >= 7.:
                is_the_haiti_event(ev)
                ident = ev.name

        assert ident is not None
        cat.flush()
        ev = cat.get_event(ident)
        is_the_haiti_event(ev)

    def testNDKParse(self):
        events = globalcmt.parse_ndk(NDK_SAMPLE)
        self.assertEqual(
            [ev.name for ev in events],
            ['010176A', '031579B', '112096E', '200501211345A',
             '201001122153A', '202601010153A'])

        self.assertEqual(
            [ev.extras.get('catalog_version') for ev in events],
            ['O', 'O', 'O', 'S', 'S', 'Q'])

        self.assertTrue(all(ev.catalog == 'gCMT' for ev in events))

        tmods = [ev.extras.get('catalog_time_modified') for ev in events]
        self.assertEqual(tmods[:3], [None, None, None])
        self.assertEqual(tmods[3], util.stt('2005-04-19 09:43:23'))
        self.assertEqual(tmods[4], util.stt('2010-03-19 11:54:23'))
        self.assertEqual(tmods[5], util.stt('2025-12-31 22:31:17'))

        haiti = events[4]
        self.assertTrue(near(
            haiti.time, util.stt('2010-01-12 21:53:17.1'), 1e-6))
        self.assertEqual((haiti.lat, haiti.lon), (18.61, -72.62))
        self.assertEqual(haiti.depth, 12000.)
        self.assertTrue(near(haiti.duration, 16.2, 1e-9))
        self.assertTrue(near(haiti.magnitude, 7.0, 0.1))
        self.assertEqual(haiti.region, 'HAITI REGION')
        mrr, mtt, mpp = haiti.moment_tensor.m_up_south_east().diagonal()
        self.assertTrue(near(mrr, 1.74e19, 1e10))
        self.assertTrue(near(mtt, -3.87e19, 1e10))

        with self.assertRaises(globalcmt.NDKError):
            globalcmt.parse_ndk('\n'.join(NDK_SAMPLE.splitlines()[:7]))

    @common.require_internet
    @common.skip_on(requests.ConnectionError)
    def testGlobalCMTNDK(self):
        cache_dir = tempfile.mkdtemp(prefix='pyrocko-gcmt-ndk')
        try:
            cat = globalcmt.GlobalCMTNDK(cache_dir=cache_dir)

            # monthly file
            events = cat.get_events((
                util.stt('2023-02-06 00:00:00'),
                util.stt('2023-02-07 00:00:00')), magmin=7.)

            names = [ev.name for ev in events]
            self.assertIn('202302060117A', names)
            for ev in events:
                self.assertEqual(ev.extras['catalog_version'], 'S')

            cat.flush()
            ev = cat.get_event('202302060117A')
            self.assertTrue(near(ev.magnitude, 7.8, 0.1))

            # recent events, quick solutions
            tmax = util.day_start(util.time.time())
            events = cat.get_events((tmax - 30*24*3600., tmax), magmin=5.5)
            self.assertTrue(events)
            self.assertIn(
                'Q', set(ev.extras['catalog_version'] for ev in events))

            with self.assertRaises(catalog.NotFound):
                cat.get_event('209901010000A')

        finally:
            shutil.rmtree(cache_dir)


if __name__ == '__main__':
    util.setup_logging('test_catalog', 'debug')
    unittest.main()
