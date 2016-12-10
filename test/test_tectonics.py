import unittest
import numpy as num
from numpy.testing import assert_allclose

from pyrocko import tectonics, util


class TectonicsTestCase(unittest.TestCase):

    def test_boundaries(self):
        bird = tectonics.PeterBird2003()
        boundaries = bird.get_boundaries()
        assert len(boundaries) == 229

    def test_boundaries_types(self):
        bird = tectonics.PeterBird2003()
        boundaries = bird.get_boundaries()
        for boundary in boundaries:
            if (boundary.name1, boundary.kind, boundary.name2) == (
                    'PS', '/', 'PA'):
                tt = boundary.split_types()
                assert len(tt) == 1
                assert tt[0][0] == 'SUB'

            if (boundary.name1, boundary.kind, boundary.name2) == (
                    'AS', '-', 'AT'):

                lastpoint = None
                for typ, part in boundary.split_types():
                    if lastpoint is not None:
                        assert_allclose(lastpoint, part[0, :])
                        lastpoint = part[:, -1]

                lastpoint = None
                types = []
                for typ, part in boundary.split_types(
                        [['OSR'],
                         ['SUB'],
                         ['OTF', 'OCB', 'CTF', 'CCB', 'CRB']]):

                    if lastpoint is not None:
                        assert_allclose(lastpoint, part[0, :])
                        lastpoint = part[:, -1]

                    types.append(typ)

                assert types == [['OSR'], ['CTF', 'CRB']]

    def test_plates(self):
        bird = tectonics.PeterBird2003()
        plates = bird.get_plates()

        assert len(plates) == 52

        point = num.array([-25., 135.], dtype=num.float)
        for plate in plates:
            plate.max_interpoint_distance()
            assert (plate.name == 'AU') == plate.contains_point(point)

        lats = num.linspace(-20., -10., 20)
        lons = num.linspace(-80., -70., 20)
        lats2 = num.repeat(lats, lons.size)
        lons2 = num.tile(lons, lats.size)
        points = num.vstack((lats2, lons2)).T

        full_names = []
        for plate in plates:
            if num.any(plate.contains_points(points)):
                full_names.append(bird.full_name(plate.name))

        assert full_names == [
            'South American Plate',
            'Nazca Plate',
            'Altiplano Plate']

    def test_rotations(self):
        for lat in num.linspace(-90., 90., 20):
            for lon in num.linspace(-180., 180., 20):
                point = num.array([lat, lon], dtype=num.float)
                xyz = tectonics.latlon_to_xyz(point)
                rot = tectonics.rot_to_00(point[0], point[1])
                p2 = num.dot(rot, xyz)
                assert_allclose(p2, [1., 0., 0.], atol=1.0e-7)

    def test_rotation2(self):
        eps = 1.0e-7
        lats = num.linspace(50., 60., 20)
        lons = num.linspace(170., 180., 20)
        lats2 = num.repeat(lats, lons.size)
        lons2 = num.tile(lons, lats.size)
        points = num.vstack((lats2, lons2)).T
        xyz = tectonics.latlon_to_xyz(points)
        rot = tectonics.rot_to_00(lats[0], lons[0])
        xyz2 = num.dot(rot, xyz.T).T
        points2 = tectonics.xyz_to_latlon(xyz2)
        assert num.all(points2[:, 1] > -eps)

    def test_velocities(self):
        gsrm = tectonics.GSRM1()
        lats, lons, vnorth, veast, vnorth_err, veast_err, corr = \
            gsrm.get_velocities('AF', region=(20, 30, 0., 10.))
        assert 0.0 < num.mean(num.sqrt(veast**2 + vnorth**2)) < 0.01
        assert_allclose(
            [lats.min(), lats.max(), lons.min(), lons.max()],
            [0., 10., 20., 30.], atol=1e-5)

        assert gsrm.full_name('NU') == 'African Plate'
        assert 'EU' in gsrm.plate_names()


if __name__ == "__main__":
    util.setup_logging('test_tectonics', 'warning')
    unittest.main()
