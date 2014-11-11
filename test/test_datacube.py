from pyrocko import util, io
import unittest
import common


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


if __name__ == "__main__":
    util.setup_logging('test_io', 'warning')
    unittest.main()
