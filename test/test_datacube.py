import time
import unittest

import common
from pyrocko import util, io, datacube_ext


class DataCubeTestCase(unittest.TestCase):

    def test_load(self):
        fpath = common.test_data_file('test1.cube')

        traces_h = io.load(fpath, getdata=False, format='detect')
        traces_d = io.load(fpath, getdata=True, format='detect')

        mimas = [
            (36725, 77299),
            (50456, 74265),
            (56080, 71058),
        ]

        for tr_h, tr_d, (mi, ma) in zip(traces_h, traces_d, mimas):
            assert tr_h.tmin == tr_d.tmin
            assert tr_h.tmax == tr_d.tmax
            assert tr_d.ydata.min() == mi
            assert tr_d.ydata.max() == ma

    def benchmark_load(self):


        fpath = common.test_data_file('test2.cube')
        for irep in range(2):
            for loadflag in (0,1,2):
                f = open(fpath, 'r')
                t0 = time.time()
                header, data_arrays, gps_tags, nsamples, bookmarks = datacube_ext.load(
                    f.fileno(), loadflag, 0, -1)

                f.close()
                t1 = time.time()
                print '%i %10.3f' % (loadflag, t1 - t0)



if __name__ == "__main__":
    util.setup_logging('test_io', 'warning')
    unittest.main()
