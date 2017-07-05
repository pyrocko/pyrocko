import unittest
import numpy as num

from pyrocko import gshhg, util

plot = False


class BB(object):
    west = 5.
    east = 20.
    south = 35.
    north = 55.

    tpl = (west, east, south, north)


class BBLakes(object):
    west = 4.41
    east = 13.14
    south = 42.15
    north = 49.32

    tpl = (west, east, south, north)


class GSHHGTest(unittest.TestCase):
    def setUp(self):
        self.gshhg = gshhg.GSHHG.get_intermediate()

    @unittest.skip('')
    def test_crude(self):
        pass

    @unittest.skip('')
    def test_polygon_loading_points(self):
        for ipoly in xrange(10):
            self.gshhg.polygons[ipoly].points

    @unittest.skip('')
    def test_polygon_contains_point(self):
        poly = self.gshhg.polygons[-1]
        p_within = num.array(
            [poly.south + (poly.north - poly.south) / 2,
             poly.west + (poly.east - poly.west) / 2])

        p_outside = num.array(
            [poly.north + (poly.north - poly.south) / 2,
             poly.east + (poly.east - poly.west) / 2])

        assert poly.contains_point(p_within) is True
        assert poly.contains_point(p_outside) is False

    @unittest.skip('')
    def test_polygon_level_selection(self):
        for p in self.gshhg.polygons:
            p.is_land()
            p.is_island_in_lake()
            p.is_pond_in_island_in_lake()
            p.is_antarctic_icefront()
            p.is_antarctic_grounding_line()

    # @unittest.skip('')
    def test_polygon_contains_points(self):
        poly = self.gshhg.polygons[0]

        points = num.array(
                [num.random.uniform(BB.south, BB.north, size=100),
                 num.random.uniform(BB.east, BB.west, size=100)]).T
        pts = poly.contains_points(points)

        if plot:
            import matplotlib.pyplot as plt
            import cartopy.crs as ccrs

            points = points[pts]

            colors = num.ones(points.shape[0]) * pts[pts]
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_global()

            self.plot_polygons([poly], ax)
            ax.scatter(points[:, 1], points[:, 0], c=colors, s=.5,
                       transform=ccrs.Geodetic())
            plt.show()

    @unittest.skip('')
    def test_bounding_box_select(self):
        print 'test...'
        p = self.gshhg.get_polygons_within(*BB.tpl)
        print len(self.gshhg.polygons), len(p)

        if plot:
            import matplotlib.pyplot as plt
            import cartopy.crs as ccrs

            from matplotlib.patches import Rectangle

            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_global()
            self.plot_polygons(p, ax)

            ax.add_patch(
                Rectangle([BB.west, BB.south],
                          width=BB.east-BB.west, height=BB.north-BB.south,
                          alpha=.2))
            plt.show()

    def test_mask_land(self):
        poly = self.gshhg.get_polygons_within(*BBLakes.tpl)

        points = num.array(
                [num.random.uniform(BBLakes.south, BBLakes.north, size=1000),
                 num.random.uniform(BBLakes.east, BBLakes.west, size=1000)]).T
        pts = self.gshhg.get_land_mask(points)

        if plot:
            import matplotlib.pyplot as plt
            import cartopy.crs as ccrs

            points = points[pts]
            colors = num.ones(points.shape[0]) * pts[pts]

            # colors = num.ones(points.shape[0]) * pts
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_global()

            self.plot_polygons(poly, ax)
            ax.scatter(points[:, 1], points[:, 0], c=colors, s=.5,
                       transform=ccrs.Geodetic())
            plt.show()

    @unittest.skip('')
    def test_is_point_on_land(self):

        point = (46.455289, 6.494283)
        p = self.gshhg.get_polygons_at(*point)
        assert self.gshhg.is_point_on_land(*point) is False

        if plot:
            import matplotlib.pyplot as plt
            import cartopy.crs as ccrs

            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_global()
            self.plot_polygons(p, ax)
            ax.scatter(point[1], point[0], transform=ccrs.Geodetic())

            plt.show()

    @staticmethod
    def plot_polygons(polygons, ax, **kwargs):
        from matplotlib.patches import Polygon
        import numpy as num
        import cartopy.crs as ccrs

        args = {
            'edgecolor': 'red',
            'facecolor': 'none',
            'alpha': .3
        }
        args.update(kwargs)

        map(ax.add_patch, [Polygon(num.fliplr(p.points),
                                   transform=ccrs.Geodetic(),
                                   **args)
                           for p in polygons])


if __name__ == "__main__":
    plot = True
    util.setup_logging('test_gshhg', 'debug')
    unittest.main(exit=False)
