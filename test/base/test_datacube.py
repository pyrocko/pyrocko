# python 2/3
from __future__ import division, print_function, absolute_import

import time
import unittest
from collections import defaultdict

import numpy as num

from .. import common
from pyrocko import util, io, datacube_ext
from pyrocko.io import datacube


class DataCubeTestCase(unittest.TestCase):

    def test_load(self):
        fpath = common.test_data_file('test1.cube')

        traces_h = io.load(fpath, getdata=False, format='detect')
        traces_d = io.load(fpath, getdata=True, format='detect')

        mimas = [
            (24464, 88087),
            (42794, 80074),
            (53039, 73741)]

        for tr_h, tr_d, (mi, ma) in zip(traces_h, traces_d, mimas):
            assert tr_h.tmin == tr_d.tmin
            assert tr_h.tmax == tr_d.tmax
            assert tr_d.ydata.min() == mi
            assert tr_d.ydata.max() == ma

    def test_load_partial(self):
        fpath = common.test_data_file('test2.cube')
        f = open(fpath, 'r')
        header, da1, gps_tags, nsamples, bookmarks = datacube_ext.load(
            f.fileno(), 2, 0, -1, None)

        for ioff in (0, 10, 1040000, 1048576, 2000000, 1000000):
            f.seek(0)
            header, da2, gps_tags, nsamples, _ = datacube_ext.load(
                f.fileno(), 2, ioff, 10, None)

            f.seek(0)
            header, da3, gps_tags, nsamples, _ = datacube_ext.load(
                f.fileno(), 2, ioff, 10, bookmarks)

            for a1, a2, a3 in zip(da1, da2, da3):
                assert num.all(a1[ioff:ioff+10] == a2) and num.all(a2 == a3)

        f.close()

    def test_interpolate_or_not(self):
        fpath = common.test_data_file('test2.cube')
        trs = {}
        for imode in ('off', 'sinc'):
            trs[imode] = list(datacube.iload(fpath, interpolation=imode))[:1]

            for tr in trs[imode]:
                tr.set_codes(location='i=%s' % imode)

        # import pylab as lab
        for cha in ['p0']:  # 'p1', 'p2']:
            t1 = [tr for tr in trs['off'] if tr.channel == cha][0]
            t2 = [tr for tr in trs['sinc'] if tr.channel == cha][0]
            it = 0
            nt = min(t1.ydata.size, t2.ydata.size)

            dd = []
            nb = int(600. / t1.deltat)
            while it < nt:
                y1 = t1.ydata[it:it+nb]
                y2 = t2.ydata[it:it+nb]
                dd.append(abs(num.mean(y1) - num.mean(y2)))
                assert dd[-1] < 1.0
                it += nb

            # t = num.arange(len(dd))*600.
            # d = num.array(dd)
            # lab.plot(t / 3600., d)

        # lab.show()

        # trace.snuffle(trs['off'] + trs['sinc'])

    def test_timing_context(self):
        fpath = common.test_data_file('test2.cube')
        datacube.get_extended_timing_context(fpath)

    def benchmark_load(self):
        mode = {
            0: 'get time range',
            1: 'get gps only',
            2: 'get samples'}

        fpath = common.test_data_file('test2.cube')
        for irep in range(2):
            for loadflag in (0, 1, 2):
                f = open(fpath, 'r')
                t0 = time.time()
                header, data_arrays, gps_tags, nsamples, bookmarks = \
                    datacube_ext.load(f.fileno(), loadflag, 0, -1, None)

                f.close()
                t1 = time.time()
                print('%s: %10.3f' % (mode[loadflag], t1 - t0))

            t0 = time.time()
            trs = io.load(fpath, format='datacube')
            t1 = time.time()
            print('with interpolation: %10.3f' % (t1 - t0))
            del trs

    def test_leapsecond(self):
        fns = map(common.test_data_file, ['leapsecond_dec.cube',
                                          'leapsecond_jan.cube'])

        trs = defaultdict(list)
        for fn in fns:
            for tr in datacube.iload(fn):
                trs[tr.channel].append(tr)

        for cha in trs.keys():
            tra, trb, trc = trs[cha]
            assert abs(
                tra.tmax - (util.stt('2017-01-01 00:00:01') - tra.deltat)) \
                < tra.deltat * 0.001
            assert abs(
                trb.tmin - util.stt('2017-01-01 00:00:00')) \
                < trb.deltat * 0.001

    def test_subsample_shift(self):
        for fn in ['test_gps_rect_100.cube', 'test_gps_rect_200.cube']:
            fpath = common.test_data_file(fn)

            datacube.APPLY_SUBSAMPLE_SHIFT_CORRECTION = False
            traces_uncorrected = io.load(fpath, getdata=True, format='detect')

            datacube.APPLY_SUBSAMPLE_SHIFT_CORRECTION = True
            traces_corrected = io.load(fpath, getdata=True, format='detect')

            traces_corrected2 = []
            for tr in traces_uncorrected:
                tr.set_codes(location='uncorrected')
                tr2 = tr.copy()
                tcorr = 0.199 * tr2.deltat + 0.0003
                tr2.shift(-tcorr)
                tr2.snap(interpolate=True)
                traces_corrected2.append(tr2)
                tr2.set_codes(location='corrected (comparison)')

            for tr, tr2 in zip(traces_corrected, traces_corrected2):
                tr.set_codes(location='corrected')
                tmin = tr.tmin + 1.0
                tmax = tr.tmax - 1.0
                data1 = tr.chop(tmin, tmax, inplace=False).get_ydata()
                data2 = tr2.chop(tmin, tmax, inplace=False).get_ydata()
                err = num.sqrt(num.sum((data1 - data2)**2))/data1.size
                assert err < 1.0

            # from pyrocko import trace
            # trace.snuffle(
            #     traces_uncorrected + traces_corrected + traces_corrected2)


if __name__ == "__main__":
    util.setup_logging('test_io', 'warning')
    unittest.main()
