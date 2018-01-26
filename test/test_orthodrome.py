from __future__ import division, print_function, absolute_import
import unittest
import math
import random

import numpy as num
import logging

from pyrocko import orthodrome, util
from pyrocko import orthodrome_ext
from .common import Benchmark
from pyrocko import config

logger = logging.getLogger('pyrocko.test.test_orthodrome')
benchmark = Benchmark()

earth_oblateness = 1./298.257223563
earthradius_equator = 6378.14 * 1000.
earthradius = config.config().earthradius

r2d = 180./math.pi
d2r = 1./r2d
km = 1000.

plot = False


def random_lat(mi=-90., ma=90., rstate=None, size=None):
    if rstate is None:
        rstate = num.random
    mi_ = 0.5*(math.sin(mi * math.pi/180.)+1.)
    ma_ = 0.5*(math.sin(ma * math.pi/180.)+1.)
    return num.arcsin(rstate.uniform(mi_, ma_, size=size)*2.-1.)*180./math.pi


def random_lon(mi=-180., ma=180., rstate=None, size=None):
    if rstate is None:
        rstate = num.random
    mi_ = 0.5*(math.sin(mi * math.pi/180.)+1.)
    ma_ = 0.5*(math.sin(ma * math.pi/180.)+1.)
    return num.arcsin(rstate.uniform(mi_, ma_, size=size)*2.-1.)*180./math.pi


def light(color, factor=0.5):
    return tuple(1-(1-c)*factor for c in color)


def dark(color, factor=0.5):
    return tuple(c*factor for c in color)


def have_geographiclib():
    try:
        from geographiclib.geodesic import Geodesic  # noqa
        return True

    except ImportError:
        return False


class OrthodromeTestCase(unittest.TestCase):

    @classmethod
    def tearDownClass(self):
        print(benchmark)

    def get_critical_random_locations(self, ntest):
        '''
        Create list of random (lat1, lon1, lat2, lon2) including critical
        locations.
        '''

        nasty_locations = [
            (0., 0., 0., 0.),
            (90., 0., -90., 0.),
            (0., -180., 0., 180.),
        ]

        assert ntest >= len(nasty_locations)

        lats1 = num.random.uniform(-90., 90., ntest)
        lats2 = num.random.uniform(-90., 90., ntest)
        lons1 = num.random.uniform(-180., 180., ntest)
        lons2 = num.random.uniform(-180., 180., ntest)

        for i, llll in enumerate(nasty_locations):
            lats1[i], lons1[i], lats2[i], lons2[i] = llll

        return lats1, lons1, lats2, lons2

    def testRaisesValueError(self):
        for lat1, lon1, lat2, lon2 in [
                (91., 0., 0., 0.),
                (0., 181., 0., 0.),
                (0., 0., 91., 0.),
                (0., 0., 0., 181.), ]:
            for f in [1, -1]:

                lat1 *= f
                lon1 *= f
                lat2 *= f
                lon2 *= f

                with self.assertRaises(ValueError):
                    orthodrome_ext.azibazi(lat1, lon1, lat2, lon2)

                with self.assertRaises(ValueError):
                    orthodrome_ext.distance_accurate50m(lat1, lon1, lat2, lon2)

    @benchmark
    def testAziBaziPython(self):
        ntest = 10000
        lats1, lons1, lats2, lons2 = self.get_critical_random_locations(ntest)
        for i in range(ntest):
            orthodrome.azibazi(
                float(lats1[i]), float(lons1[i]),
                float(lats2[i]), float(lons2[i]),
                implementation='python')

    @benchmark
    def testAziBaziC(self):
        ntest = 10000
        lats1, lons1, lats2, lons2 = self.get_critical_random_locations(ntest)
        for i in range(ntest):
            orthodrome.azibazi(
                lats1[i], lons1[i], lats2[i], lons2[i],
                implementation='c')

    def testAziBaziPythonC(self):
        ntest = 100
        lats1, lons1, lats2, lons2 = self.get_critical_random_locations(ntest)

        for i in range(ntest):
            azibazi_py = orthodrome.azibazi(
                float(lats1[i]), float(lons1[i]),
                float(lats2[i]), float(lons2[i]),
                implementation='python')
            azibazi_c = orthodrome.azibazi(
                lats1[i], lons1[i], lats2[i], lons2[i],
                implementation='c')

            num.testing.assert_almost_equal(azibazi_py, azibazi_c)

    @benchmark
    def testAziBaziArrayPython(self):
        ntest = 10000
        locs = self.get_critical_random_locations(ntest)
        orthodrome.azibazi_numpy(
            *locs,
            implementation='python')

    @benchmark
    def testAziBaziArrayC(self):
        ntest = 10000
        locs = self.get_critical_random_locations(ntest)
        orthodrome.azibazi_numpy(
            *locs,
            implementation='c')

    def testAziBaziArrayPythonC(self):
        ntest = 10000

        lats1, lons1, lats2, lons2 = self.get_critical_random_locations(ntest)

        azis_c, bazis_c = orthodrome.azibazi_numpy(
            lats1, lons1, lats2, lons2,
            implementation='c')

        azis_py, bazis_py = orthodrome.azibazi_numpy(
            lats1, lons1, lats2, lons2,
            implementation='python')

        for i in range(ntest):
            azi_py, bazi_py = orthodrome.azibazi(
                float(lats1[i]), float(lons1[i]),
                float(lats2[i]), float(lons2[i]),
                implementation='python')

            azi_c, bazi_c = orthodrome.azibazi(
                lats1[i], lons1[i], lats2[i], lons2[i],
                implementation='c')

            num.testing.assert_almost_equal(azi_py, azis_py[i])
            num.testing.assert_almost_equal(bazi_py, bazis_py[i])
            num.testing.assert_almost_equal(azi_c, azis_c[i])
            num.testing.assert_almost_equal(bazi_c, bazis_c[i])

        num.testing.assert_almost_equal(azis_py, azis_c)
        num.testing.assert_almost_equal(bazis_py, bazis_c)

    @benchmark
    def testDistancePython(self):
        ntest = 1000
        lats1, lons1, lats2, lons2 = self.get_critical_random_locations(ntest)
        loc1 = orthodrome.Loc(0., 0.)
        loc2 = orthodrome.Loc(0., 0.)
        for i in range(ntest):
            loc1.lat, loc1.lon = lats1[i], lons1[i]
            loc2.lat, loc2.lon = lats2[i], lons2[i]
            orthodrome.distance_accurate50m(
                loc1, loc2,
                implementation='python')

    @benchmark
    def testDistanceC(self):
        ntest = 1000
        lats1, lons1, lats2, lons2 = self.get_critical_random_locations(ntest)
        for i in range(ntest):
            orthodrome.distance_accurate50m(
                lats1[i], lons1[i], lats2[i], lons2[i],
                implementation='c')

    def testDistancePythonC(self):
        ntest = 100
        lats1, lons1, lats2, lons2 = self.get_critical_random_locations(ntest)
        for i in range(ntest):
            dist_py = orthodrome.distance_accurate50m(
                lats1[i], lons1[i], lats2[i], lons2[i],
                implementation='python')
            dist_c = orthodrome.distance_accurate50m(
                lats1[i], lons1[i], lats2[i], lons2[i],
                implementation='c')
            num.testing.assert_almost_equal(dist_py, dist_c)

    @benchmark
    def testDistanceArrayPython(self):
        ntest = 10000
        locs = self.get_critical_random_locations(ntest)
        orthodrome.distance_accurate50m_numpy(*locs, implementation='python')

    @benchmark
    def testDistanceArrayC(self):
        ntest = 10000
        locs = self.get_critical_random_locations(ntest)
        orthodrome.distance_accurate50m_numpy(*locs, implementation='c')

    def testDistanceArrayPythonC(self):
        ntest = 10000
        locs = self.get_critical_random_locations(ntest)
        a = orthodrome.distance_accurate50m_numpy(
            *locs,
            implementation='python')
        b = orthodrome.distance_accurate50m_numpy(
            *locs,
            implementation='c')
        num.testing.assert_array_almost_equal(a, b)

    def testGridDistances(self):
        for i in range(100):
            gsize = random.uniform(0., 1.)*2.*10.**random.uniform(4., 7.)
            north_grid, east_grid = num.meshgrid(
                num.linspace(-gsize/2., gsize/2., 11),
                num.linspace(-gsize/2., gsize/2., 11))

            north_grid = north_grid.flatten()
            east_grid = east_grid.flatten()

            lon = random.uniform(-180., 180.)
            lat = random.uniform(-90., 90.)

            lat_grid, lon_grid = orthodrome.ne_to_latlon(
                lat, lon, north_grid, east_grid)
            lat_grid_alt, lon_grid_alt = \
                orthodrome.ne_to_latlon_alternative_method(
                    lat, lon, north_grid, east_grid)

            for la, lo, no, ea in zip(
                    lat_grid, lon_grid, north_grid, east_grid):
                a = orthodrome.Loc(lat=la, lon=lo)
                b = orthodrome.Loc(lat=lat, lon=lon)

                cd = orthodrome.cosdelta(a, b)
                assert cd <= 1.0
                d = num.arccos(cd)*earthradius
                d2 = math.sqrt(no**2+ea**2)
                assert not (abs(d-d2) > 1.0e-3 and d2 > 1.)

    def test_wrap(self):
        assert orthodrome.wrap(11, -10, 10) == -9
        assert orthodrome.wrap(10, -10, 10) == -10
        assert orthodrome.wrap(10.001, -10, 10) == -9.999

    @unittest.skip('needs inspection')
    def test_local_distances(self):

        for reflat, reflon in [
                (0.0, 0.0),
                (10.0, 10.0),
                (90.0, 0.0),
                (-90.0, 0.0),
                (0.0, 180.0),
                (0.0, -180.0),
                (90.0, 180.0)]:

            north, east = serialgrid(num.linspace(-10*km, 10*km, 21),
                                     num.linspace(-10*km, 10*km, 21))

            lat, lon = orthodrome.ne_to_latlon2(reflat, reflon, north, east)
            north2, east2 = orthodrome.latlon_to_ne2(reflat, reflon, lat, lon)
            dist1 = num.sqrt(north**2 + east**2)
            dist2 = num.sqrt(north2**2 + east2**2)
            dist3 = orthodrome.distance_accurate15nm(reflat, reflon, lat, lon)
            assert num.all(num.abs(dist1-dist2) < 0.0001)
            assert num.all(num.abs(dist1-dist3) < 0.0001)

    def test_midpoint(self):
        center_lons = num.linspace(0., 180., 5)
        center_lats = [0., 89.]
        npoints = 10000
        half_side_length = 1000000.
        distance_error_max = 50000.
        for lat in center_lats:
            for lon in center_lons:
                n = num.random.uniform(
                    -half_side_length, half_side_length, npoints)
                e = num.random.uniform(
                    -half_side_length, half_side_length, npoints)
                dlats, dlons = orthodrome.ne_to_latlon(lat, lon, n, e)
                clat, clon = orthodrome.geographic_midpoint(dlats, dlons)
                d = orthodrome.distance_accurate50m_numpy(
                    clat, clon, lat, lon)[0]

                if plot:
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.scatter(n, e)
                    c_n, c_e = orthodrome.latlon_to_ne_numpy(
                        lat, lon, clat, clon)
                    ax.plot(c_n, c_e, 'ro')
                    plt.show()

                self.assertTrue(d < distance_error_max, 'Distance %s > %s' %
                                (d, distance_error_max) +
                                '(maximum error)\n tested lat/lon: %s/%s' %
                                (lat, lon))

    @unittest.skipUnless(have_geographiclib(), 'geographiclib not available')
    def test_geodetic_to_ecef(self):
        orthodrome.geodetic_to_ecef(23., 0., 0.)

        a = orthodrome.earthradius_equator
        b = orthodrome.earthradius_equator * (1. - orthodrome.earth_oblateness)

        points = [
            ((90., 0., 0.), (0., 0., b)),
            ((-90., 0., 10.), (0., 0., -b-10.)),
            ((0., 0., 0.), (a, 0., 0.))]

        for p in points:
            num.testing.assert_almost_equal(orthodrome.geodetic_to_ecef(*p[0]),
                                            p[1])

    @unittest.skipUnless(have_geographiclib(), 'geographiclib not available')
    def test_ecef_to_geodetic(self):
        ncoords = 5
        lats = num.random.uniform(-90., 90, size=ncoords)
        lons = num.random.uniform(-180., 180, size=ncoords)
        alts = num.random.uniform(0, 10, size=ncoords)

        coords = num.array([lats, lons, alts]).T

        for ic in range(coords.shape[0]):
            xyz = orthodrome.geodetic_to_ecef(*coords[ic, :])
            latlonalt = orthodrome.ecef_to_geodetic(*xyz)

            num.testing.assert_almost_equal(coords[ic, :], latlonalt)

    def test_rotations(self):
        for lat in num.linspace(-90., 90., 20):
            for lon in num.linspace(-180., 180., 20):
                point = num.array([lat, lon], dtype=num.float)
                xyz = orthodrome.latlon_to_xyz(point)
                rot = orthodrome.rot_to_00(point[0], point[1])
                p2 = num.dot(rot, xyz)
                num.testing.assert_allclose(p2, [1., 0., 0.], atol=1.0e-7)

    def test_rotation2(self):
        eps = 1.0e-7
        lats = num.linspace(50., 60., 20)
        lons = num.linspace(170., 180., 20)
        lats2 = num.repeat(lats, lons.size)
        lons2 = num.tile(lons, lats.size)
        points = num.vstack((lats2, lons2)).T
        xyz = orthodrome.latlon_to_xyz(points)
        rot = orthodrome.rot_to_00(lats[0], lons[0])
        xyz2 = num.dot(rot, xyz.T).T
        points2 = orthodrome.xyz_to_latlon(xyz2)
        assert num.all(points2[:, 1] > -eps)

    def test_point_in_polygon(self):
        if plot:
            from pyrocko.plot import mpl_graph_color

            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon

            axes = plt.gca()

        nip = 100

        for i in range(1):
            np = 3
            points = num.zeros((np, 2))
            points[:, 0] = random_lat(size=3)
            points[:, 1] = random_lon(size=3)

            points_ip = num.zeros((nip*points.shape[0], 2))
            for ip in range(points.shape[0]):
                n, e = orthodrome.latlon_to_ne_numpy(
                    points[ip % np, 0], points[ip % np, 1],
                    points[(ip+1) % np, 0], points[(ip+1) % np, 1])

                ns = num.arange(nip) * n / nip
                es = num.arange(nip) * e / nip
                lats, lons = orthodrome.ne_to_latlon(
                    points[ip % np, 0], points[ip % np, 1], ns, es)

                points_ip[ip*nip:(ip+1)*nip, 0] = lats
                points_ip[ip*nip:(ip+1)*nip, 1] = lons

            if plot:
                color = mpl_graph_color(i)
                axes.add_patch(
                    Polygon(
                        num.fliplr(points_ip),
                        facecolor=light(color),
                        edgecolor=color,
                        alpha=0.5))

            points_xyz = orthodrome.latlon_to_xyz(points_ip.T)
            center_xyz = num.mean(points_xyz, axis=0)

            assert num.all(
                orthodrome.distances3d(
                    points_xyz, center_xyz[num.newaxis, :]) < 1.0)

            lat, lon = orthodrome.xyz_to_latlon(center_xyz)
            rot = orthodrome.rot_to_00(lat, lon)

            points_rot_xyz = num.dot(rot, points_xyz.T).T
            points_rot_pro = orthodrome.stereographic(points_rot_xyz)  # noqa

            poly_xyz = orthodrome.latlon_to_xyz(points_ip)
            poly_rot_xyz = num.dot(rot, poly_xyz.T).T
            groups = orthodrome.spoly_cut([poly_rot_xyz], axis=0)
            num.zeros(points.shape[0], dtype=num.int)

            if plot:
                for group in groups:
                    for poly_rot_group_xyz in group:

                        axes.set_xlim(-180., 180.)
                        axes.set_ylim(-90., 90.)

                    plt.show()

    def test_point_in_region(self):
        testdata = [
            ((-20., 180.), (-180., 180., -90., 90.), True),
            ((-20., 180.), (170., -170., -90., 90.), True),
            ((-20., 160.), (170., -170., -90., 90.), False),
            ((-20., -160.), (170., -170., -90., 90.), False),
        ]

        for point, region, in_region in testdata:
            assert bool(orthodrome.point_in_region(point, region)) == in_region


def serialgrid(x, y):
    return num.repeat(x, y.size), num.tile(y, x.size)


def plot_erroneous_ne_to_latlon():
    import gmtpy
    import random
    import subprocess
    import time

    while True:
        w, h = 20, 15

        gsize = random.uniform(0., 1.)*4.*10.**random.uniform(4., 7.)
        north_grid, east_grid = num.meshgrid(
            num.linspace(-gsize/2., gsize/2., 11),
            num.linspace(-gsize/2., gsize/2., 11))

        north_grid = north_grid.flatten()
        east_grid = east_grid.flatten()

        lat_delta = gsize/earthradius*r2d*2.
        lon = random.uniform(-180., 180.)
        lat = random.uniform(-90., 90.)

        print(gsize/1000.)

        lat_grid, lon_grid = orthodrome.ne_to_latlon(
            lat, lon, north_grid, east_grid)
        lat_grid_alt, lon_grid_alt = \
            orthodrome.ne_to_latlon_alternative_method(
                lat, lon, north_grid, east_grid)

        maxerrlat = num.max(num.abs(lat_grid-lat_grid_alt))
        maxerrlon = num.max(num.abs(lon_grid-lon_grid_alt))
        eps = 1.0e-8
        if maxerrlon > eps or maxerrlat > eps:
            print(lat, lon, maxerrlat, maxerrlon)

            gmt = gmtpy.GMT(
                config={
                    'PLOT_DEGREE_FORMAT': 'ddd.xxxF',
                    'PAPER_MEDIA': 'Custom_%ix%i' % (w*gmtpy.cm, h*gmtpy.cm),
                    'GRID_PEN_PRIMARY': 'thinnest/0/50/0'})

            south = max(-85., lat - 0.5*lat_delta)
            north = min(85., lat + 0.5*lat_delta)

            lon_delta = lat_delta/math.cos(lat*d2r)

            delta = lat_delta/360.*earthradius*2.*math.pi
            scale_km = gmtpy.nice_value(delta/10.)/1000.

            west = lon - 0.5*lon_delta
            east = lon + 0.5*lon_delta

            x, y = (west, east), (south, north)
            xax = gmtpy.Ax(mode='min-max', approx_ticks=4.)
            yax = gmtpy.Ax(mode='min-max', approx_ticks=4.)
            scaler = gmtpy.ScaleGuru(data_tuples=[(x, y)], axes=(xax, yax))
            scaler['R'] = '-Rg'
            layout = gmt.default_layout()
            mw = 2.5*gmtpy.cm
            layout.set_fixed_margins(
                mw, mw, mw/gmtpy.golden_ratio, mw/gmtpy.golden_ratio)
            widget = layout.get_widget()
            # widget['J'] =  ('-JT%g/%g'  % (lon, lat)) + '/%(width)gp'
            widget['J'] = (
                '-JE%g/%g/%g' % (lon, lat, min(lat_delta/2., 180.)))\
                + '/%(width)gp'
            aspect = gmtpy.aspect_for_projection(*(widget.J() + scaler.R()))
            widget.set_aspect(aspect)

            gmt.psbasemap(
                B='5g5',
                L=('x%gp/%gp/%g/%g/%gk' % (
                    widget.width()/2., widget.height()/7.,
                    lon, lat, scale_km)),
                *(widget.JXY()+scaler.R()))

            gmt.psxy(
                in_columns=(lon_grid, lat_grid),
                S='x10p', W='1p/200/0/0', *(widget.JXY()+scaler.R()))
            gmt.psxy(
                in_columns=(lon_grid_alt, lat_grid_alt),
                S='c10p', W='1p/0/0/200', *(widget.JXY()+scaler.R()))

            gmt.save('orthodrome.pdf')
            subprocess.call(['xpdf', '-remote', 'ortho', '-reload'])
            time.sleep(2)
        else:
            print('ok', gsize, lat, lon)


if __name__ == "__main__":
    plot = False
    util.setup_logging('test_orthodrome', 'warning')
    unittest.main(exit=False)
    print(benchmark)
