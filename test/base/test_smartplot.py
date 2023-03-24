import os
import tempfile
import shutil
import unittest
import numpy as num
from matplotlib import image, pyplot as plt

from pyrocko import util
from pyrocko.plot import smartplot

from .. import common


class SmartPlotTestCase(unittest.TestCase):

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
        self.assertEqual(img.shape[:2], img_ref.shape[:2])
        d = num.abs(img[:, :, :3] - img_ref[:, :, :3])
        merr = num.mean(d)
        if (merr > tolerance or show) and common.matplotlib_show():
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

    def get_config(self):
        inch = 2.54
        return smartplot.PlotConfig(
            font_size=10.,
            size_cm=(5*inch, 5*inch),
            margins_em=(7., 5., 7., 5.),
            separator_em=2.,
            colorbar_width_em=2.,
            label_offset_em=(2., 2.))

    def save_and_compare(self, p, name, dpis=[75, 150]):
        if common.matplotlib_show():
            p.show()
            p.reset_size()

        fnames = []
        for dpi in dpis:
            fname = 'smartplot_test_%s_dpi%i.png' % (name, dpi)
            fpath = self.fpath(fname)
            p.fig.savefig(fpath, dpi=dpi)
            fnames.append(fname)

        p.close()
        for fname in fnames:
            self.compare_with_ref(fname, 0.01, show=True)

    def test_basic(self):
        p = smartplot.Plot(['x'], ['y'], config=self.get_config())
        n = 100
        phi = num.arange(n) * 4. * num.pi / n

        x = num.sin(phi) * phi
        y = num.cos(phi) * phi
        p(0, 0).plot(x, y, 'o')
        self.save_and_compare(p, 'basic')

    def get_rng(self):
        return num.random.default_rng(20021977)

    def test_pair(self):
        p = smartplot.Plot(['x', 'x'], ['y'])
        n = 100
        rng = self.get_rng()
        x = num.arange(n) * 2.0
        y = rng.normal(size=n)
        p(0, 0).plot(x, y, 'o')
        y = rng.normal(size=n)
        p(1, 0).plot(x, y, 'o')
        self.save_and_compare(p, 'pair')


if __name__ == '__main__':
    plot = True
    util.setup_logging('test_smartplot', 'warning')
    unittest.main()
