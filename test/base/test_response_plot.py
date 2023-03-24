import unittest
import tempfile
import shutil
import os
import numpy as num
from matplotlib import image, pyplot as plt

from pyrocko import util
from pyrocko.plot import response

from .. import common

noshow = True


class ResponsePlotTestCase(unittest.TestCase):

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

    def test_response_plot(self):
        for fn, format in [
                ('test1.resp', 'resp'),
                ('test1.sacpz', 'sacpz')]:

            fpath_resp = common.test_data_file(fn)
            fname = 'test_response_plot_%s.png' % fn
            fpath_png = self.fpath(fname)
            resps, labels = response.load_response_information(
                fpath_resp, format)
            labels = [lab[len(fpath_resp)+1:] or 'dummy' for lab in labels]

            response.plot(
                responses=resps, labels=labels, filename=fpath_png, dpi=50)
            # self.compare_with_ref(fname, 0.01)


if __name__ == '__main__':
    util.setup_logging('test_response_plot', 'warning')
    unittest.main()
