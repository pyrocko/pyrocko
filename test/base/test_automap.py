from __future__ import division, print_function, absolute_import
import os
import math
import tempfile
import shutil
import unittest
import numpy as num
from matplotlib import image, pyplot as plt

from pyrocko import util
from pyrocko.plot import automap, gmtpy

from .. import common

noshow = False
km = 1000.


@unittest.skipUnless(
    gmtpy.have_gmt(), 'GMT not available')
@unittest.skipUnless(
    gmtpy.have_pixmaptools(), '`pdftocairo` or `convert` not available')
class AutomapTestCase(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def fpath(self, fn):
        return os.path.join(self.tempdir, fn)

    def fpath_ref(self, fn):
        try:
            return common.test_data_file(fn)
        except util.DownloadError:
            return common.test_data_file_no_download(fn)

    def compare_with_ref(self, fname, tolerance, show=False):
        fpath = self.fpath(fname)
        fpath_ref = self.fpath_ref(fname)

        if not os.path.exists(fpath_ref):
            shutil.copy(fpath, fpath_ref)

        img = image.imread(fpath)
        img_ref = image.imread(fpath_ref)
        self.assertEqual(img.shape, img_ref.shape)
        d = num.abs(img - img_ref)
        merr = num.mean(d)
        if (merr > tolerance or show) and not noshow:
            fig = plt.figure()
            axes1 = fig.add_subplot(1, 3, 1, aspect=1.)
            axes2 = fig.add_subplot(1, 3, 2, aspect=1.)
            axes3 = fig.add_subplot(1, 3, 3, aspect=1.)
            axes1.imshow(img)
            axes1.set_title('Candidate')
            axes2.imshow(img_ref)
            axes2.set_title('Reference')
            axes3.imshow(d)
            axes3.set_title('Mean abs difference: %g' % merr)
            plt.show()
            plt.close(fig)

        assert merr <= tolerance

    def test_napoli(self):
        for version in gmtpy.all_installed_gmt_versions():
            m = automap.Map(
                gmtversion=version,
                lat=40.85,
                lon=14.27,
                radius=50.*km,
                width=20.,
                height=20.,
                margins=[2., 2., 2., 4.5],
                show_grid=True,
                show_topo=True,
                show_topo_scale=True,
                illuminate=True,
                custom_cities=[
                    automap.City(
                        'Pompeji', 40.7506, 14.4897, population=9000)],
                comment='This is output from test_napoli!')

            assert m.have_coastlines()

            m.draw_cities(include=['Pompeji', 'Capri'])
            fname = 'automap_test_napoli.png'
            fpath = self.fpath(fname)
            m.save(fpath)
            self.compare_with_ref(fname, 0.01, show=False)

    def test_fidshi(self):
        x = num.linspace(178., 182., 101)
        y = num.linspace(-17.5, -14.5, 101)
        z = num.sin((x[num.newaxis, :]-x[0])/(x[-1]-x[0])*2.*math.pi*5) * \
            num.sin((y[:, num.newaxis]-y[0])/(y[-1]-y[0])*2.*math.pi*5) * 5000.

        tile = automap.FloatTile(
            xmin=x[0],
            ymin=y[0],
            dx=x[1] - x[0],
            dy=y[1] - y[0],
            data=z)

        for version in gmtpy.all_installed_gmt_versions():
            m = automap.Map(
                gmtversion=version,
                lat=-16.5,
                lon=180.,
                radius=100.*km,
                width=20.,
                height=20.,
                margins=[2., 2., 2., 4.5],
                show_grid=True,
                show_topo=True,
                show_topo_scale=True,
                replace_topo_color_only=tile,
                illuminate=True)

            m.draw_cities()
            fname = 'automap_test_fidshi.png'
            fpath = self.fpath(fname)
            m.save(fpath)
            self.compare_with_ref(fname, 0.01, show=False)

    def test_new_zealand(self):
        m = automap.Map(
            lat=-42.57,
            lon=173.01,
            radius=1000.*km,
            width=20.,
            height=20.,
            color_dry=gmtpy.color_tup('aluminium1'),
            show_topo=False,
            show_rivers=False,
            show_plates=True)

        m.draw_cities()
        fname = 'new_zealand.pdf'
        fpath = self.fpath(fname)
        m.save(fpath)


if __name__ == "__main__":
    util.setup_logging('test_automap', 'warning')
    unittest.main()
