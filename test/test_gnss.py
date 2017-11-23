import unittest

from pyrocko import util
from pyrocko.gnss import ngl


class TestNGLCatalog(unittest.TestCase):

    def setUp(self):
        self.ngl = ngl.NGLCatalog()

    def test_station_import(self):
        print self.ngl.stations[0]

    def test_search_stations(self):
        search = self.ngl.search_station(
            latitude=-12.4666,
            longitude=130.844,
            maxradius=10.)
        print search

    def test_get_station(self):
        sta = self.ngl.get_station(station_id='00NA')
        print sta

    def test_getevent(self):
        steps = self.ngl.get_event(station_id='00NA')
        for step in steps:
            print step

    def test_download_data(self):
        data = self.ngl.get_displacement('00NA')
        print data


if __name__ == '__main__':
    util.setup_logging('test_gnss.test_getevent', 'debug')
    unittest.main()
