from pyrocko import catalog, util
import unittest

def near(a,b,eps):
    return abs(a-b) < eps

class CatalogTestCase(unittest.TestCase):
    
    def testGeofon(self):
        
        cat = catalog.Geofon()
        
        tmin = util.ctimegm('2010-01-12 21:50:00')
        tmax = util.ctimegm('2010-01-13 03:17:00')
        
        names = cat.get_event_names( time_range=(tmin,tmax))
        for name in names:
            ev = cat.get_event(name)
            if ev.magnitude >= 7:
                assert near(ev.magnitude, 7.2, 0.001)
                assert near(ev.lat, 18.37, 0.001)
                assert near(ev.lon, -72.55, 0.001)
                assert near(ev.depth, 17., 0.001)
                assert ev.region == 'Haiti Region'
        
if __name__ == "__main__":
    util.setup_logging('test_catalog', 'warning')
    unittest.main()

