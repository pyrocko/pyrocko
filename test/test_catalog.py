from __future__ import division, print_function, absolute_import
from pyrocko import util
from pyrocko.client import catalog
from pyrocko import moment_tensor
import unittest
from . import common


def near(a, b, eps):
    return abs(a-b) < eps


class CatalogTestCase(unittest.TestCase):

    @common.require_internet
    def testGeofon(self):
        def is_the_haiti_event(ev):
            assert near(ev.magnitude, 7.2, 0.001)
            assert near(ev.lat, 18.37, 0.001)
            assert near(ev.lon, -72.55, 0.001)
            assert near(ev.depth, 17000., 1.)
            assert ev.region == 'Haiti Region'

        cat = catalog.Geofon()

        tmin = util.ctimegm('2010-01-12 21:50:00')
        tmax = util.ctimegm('2010-01-13 03:17:00')

        names = cat.get_event_names(
            time_range=(tmin, tmax), nmax=10, magmin=5.)

        assert len(names) > 0
        ident = None
        for name in names:
            ev = cat.get_event(name)
            if ev.magnitude >= 7:
                is_the_haiti_event(ev)
                ident = ev.name

        assert ident is not None

        cat.flush()
        ev = cat.get_event(ident)
        is_the_haiti_event(ev)

    @common.require_internet
    def testGeofonMT(self):
        cat = catalog.Geofon()
        tmin = util.ctimegm('2014-01-01 00:00:00')
        tmax = util.ctimegm('2017-01-01 00:00:00')
        events = cat.get_events((tmin, tmax), magmin=8)
        self.assertEqual(len(events), 2)
        mt1, mt2 = [ev.moment_tensor for ev in events]
        angle = moment_tensor.kagan_angle(mt1, mt2)
        self.assertEqual(round(angle - 7.7, 1), 0.0)

    @common.require_internet
    def testGlobalCMT(self):

        def is_the_haiti_event(ev):
            assert near(ev.magnitude, 7.0, 0.1)
            assert near(ev.lat, 18.61, 0.01)
            assert near(ev.lon, -72.62, 0.01)
            assert near(ev.depth, 12000., 1.)
            assert ev.region.lower() == 'haiti region'

        cat = catalog.GlobalCMT()

        tmin = util.ctimegm('2010-01-12 21:50:00')
        tmax = util.ctimegm('2010-01-13 03:17:00')

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
    def testUSGS(self):

        def is_the_haiti_event(ev):
            assert near(ev.magnitude, 7.0, 0.1)
            assert near(ev.lat, 18.443, 0.01)
            assert near(ev.lon, -72.571, 0.01)
            assert near(ev.depth, 13000., 1.)

        cat = catalog.USGS()

        tmin = util.ctimegm('2010-01-12 21:50:00')
        tmax = util.ctimegm('2010-01-13 03:17:00')

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


if __name__ == "__main__":
    util.setup_logging('test_catalog', 'debug')
    unittest.main()
