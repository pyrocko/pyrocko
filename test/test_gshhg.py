import unittest
import numpy as num

from pyrocko import gshhg, util


class TestGSHHG(unittest.TestCase):
    def setUp(self):
        self.gshhg = gshhg.GSHHG.get_intermediate()

    def test_crude(self):
        pass

    def test_loading_points(self):
        for ipoly in xrange(10):
            self.gshhg.polygons[ipoly].points

    def test_contains_point(self):
        poly = self.gshhg.polygons[-1]
        p_within = num.array(
            [poly.south + (poly.north - poly.south) / 2,
             poly.west + (poly.east - poly.west) / 2])

        p_outside = num.array(
            [poly.north + (poly.north - poly.south) / 2,
             poly.east + (poly.east - poly.west) / 2])

        assert poly.contains_point(p_within) is True
        assert poly.contains_point(p_outside) is False

    def test_contains_points(self):
        poly = self.gshhg.polygons[-1]

        points = num.array(
                [num.random.uniform(poly.south, poly.north, size=100),
                 num.random.uniform(poly.east, poly.west, size=100)]).T
        assert num.all(poly.contains_points(points))


if __name__ == "__main__":
    util.setup_logging('test_moment_tensor', 'warning')
    unittest.main()
