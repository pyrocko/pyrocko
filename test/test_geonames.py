# python 2/3

import unittest
from pyrocko.datasets import geonames
from pyrocko import util


class GeonamesTestCase(unittest.TestCase):

    def test_geonames(self):

        cities = geonames.get_cities(53.6, 10.0, 100e3, 200000)
        self.assertEqual(
            sorted(c.asciiname for c in cities),
            ['Bremen', 'Hamburg', 'Kiel', 'Luebeck'])


if __name__ == "__main__":
    util.setup_logging('test_geonames', 'warning')
    unittest.main()
