import io
import os
import shutil
import tarfile
import tempfile
import unittest

import numpy as num

from pyrocko import util
from pyrocko.dataset import slab2

km = 1e3


def make_fake_xyz(lats, lons, values):
    lons_grid, lats_grid = num.meshgrid(lons, lats)
    # Slab2 files are ordered north to south
    data = num.vstack((
        lons_grid[::-1].ravel(),
        lats_grid[::-1].ravel(),
        values[::-1].ravel())).T

    f = io.StringIO()
    for lon, lat, value in data:
        f.write('%g,%g,%s\n' % (lon, lat, 'nan' if num.isnan(value) else
                                '%g' % value))

    return f.getvalue().encode('ascii')


def make_fake_archive(path, prefixes):
    lats = 10. + num.arange(5) * 0.05
    lons = 280. + num.arange(4) * 0.05

    lons_grid, lats_grid = num.meshgrid(lons, lats)
    dep = -(lons_grid - 280.) * 100. - 10.
    dep[0, 0] = num.nan
    dep[-1, :] = num.nan

    params = {
        'dep': dep,
        'unc': num.full_like(dep, 5.),
        'str': num.full_like(dep, 359.),
        'dip': num.full_like(dep, 20.),
        'thk': num.full_like(dep, 40.)}

    params['str'][:, ::2] = 1.

    for v in params.values():
        v[num.isnan(dep)] = num.nan

    with tarfile.open(path, 'w:gz') as tar:
        for prefix in prefixes:
            for abbr, values in params.items():
                data = make_fake_xyz(lats, lons, values)
                info = tarfile.TarInfo(
                    'Slab2Distribute_Mar2018/Slab2_TXT/'
                    '%s_slab2_%s_02.23.18.xyz' % (prefix, abbr))
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

            # files which should be ignored
            data = b'1,2,3\n'
            info = tarfile.TarInfo(
                'Slab2Distribute_Mar2018/Slab2_CONTOURS/'
                '%s_slab2_dep_02.23.18_contours.in' % prefix)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    return lats, lons, params


class Slab2TestCase(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp(prefix='pyrocko-test-slab2-')
        self.archive_path = os.path.join(self.tempdir, 'slab2.tar.gz')
        self.lats, self.lons, self.params = make_fake_archive(
            self.archive_path, ['sam', 'alu'])

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def get_slabs(self):
        return slab2.Slab2(
            archive_path=self.archive_path,
            data_dir=os.path.join(self.tempdir, 'cache'))

    def test_prefixes(self):
        assert len(slab2.get_slab_prefixes()) == 27
        assert len(slab2.get_slab_names()) == 27
        assert 'depth' in slab2.get_parameter_names()

    def test_get_slab(self):
        slabs = self.get_slabs()
        slab = slabs.get_slab('sam')
        str(slab)

        assert slab.name == 'South America'
        num.testing.assert_allclose(slab.lats, self.lats)
        num.testing.assert_allclose(slab.lons, self.lons)

        num.testing.assert_allclose(
            slab.depth, -self.params['dep'] * km, rtol=1e-6)
        num.testing.assert_allclose(
            slab.thickness, self.params['thk'] * km, rtol=1e-6)
        num.testing.assert_allclose(slab.dip, self.params['dip'])

        assert os.path.exists(os.path.join(self.tempdir, 'cache', 'sam.npz'))
        assert os.path.exists(os.path.join(self.tempdir, 'cache', 'alu.npz'))

        # load from cache, without archive
        os.unlink(self.archive_path)
        slab2 = self.get_slabs().get_slab('alu')
        num.testing.assert_allclose(slab2.depth, slab.depth)

        latlondepth = slab.get_latlondepth()
        assert latlondepth.shape == (num.sum(num.isfinite(slab.depth)), 3)

    def test_unknown_slab(self):
        slabs = self.get_slabs()
        with self.assertRaises(slab2.Slab2Error):
            slabs.get_slab('xyz')

        with self.assertRaises(slab2.Slab2Error):
            slabs.get_slab('cas')  # not in archive

        with self.assertRaises(slab2.Slab2Error):
            slabs.get_slab('sam').get_values('foo')

    def test_mesh(self):
        slab = self.get_slabs().get_slab('sam')
        vertices, faces = slab.get_mesh()

        # 5x4 grid, top row NaN, one NaN corner: 3x3 cells minus one
        assert faces.shape == (8, 4)
        assert vertices.shape == (15, 3)
        assert faces.max() == vertices.shape[0] - 1
        assert num.all(num.isfinite(vertices))
        r = num.sqrt(num.sum(vertices**2, axis=1))
        assert num.all(r < 1.0) and num.all(r > 0.9)

        dip = slab.get_face_values('dip')
        assert dip.shape == (8,)
        num.testing.assert_allclose(dip, 20.)

        strike = slab.get_face_values('strike')
        assert num.all(num.minimum(strike, 360. - strike) < 1e-3)

    def test_geometry(self):
        slab = self.get_slabs().get_slab('sam')
        geom = slab.get_geometry()
        assert geom.no_faces() == 8
        assert geom.get_vertices().shape == (15, 5)
        for parameter in slab2.get_parameter_names():
            assert geom.get_property(parameter).shape == (8,)


if __name__ == '__main__':
    util.setup_logging('test_slab2', 'warning')
    unittest.main()
