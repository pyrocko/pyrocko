from pyrocko import util
import unittest
import time
from random import random


class UtilTestCase(unittest.TestCase):

    def testTime(self):

        for fmt, accu in zip(
                ['%Y-%m-%d %H:%M:%S.3FRAC', '%Y-%m-%d %H:%M:%S.2FRAC',
                 '%Y-%m-%d %H:%M:%S.1FRAC', '%Y-%m-%d %H:%M:%S'],
                [0.001, 0.01, 0.1, 1.]):

            ta = util.str_to_time('1960-01-01 10:10:10')
            tb = util.str_to_time('2020-01-01 10:10:10')

            for i in xrange(10000):
                t1 = ta + random() * (tb-ta)
                s = util.time_to_str(t1, format=fmt)
                t2 = util.str_to_time(s, format=fmt)
                assert abs(t1 - t2) < accu

    def testIterTimes(self):

        tmin = util.str_to_time('1999-03-20 20:10:10')
        tmax = util.str_to_time('2001-05-20 10:00:05')

        ii = 0
        for ymin, ymax in util.iter_years(tmin, tmax):
            for mmin, mmax in util.iter_months(ymin, ymax):
                ii += 1
                s1 = util.time_to_str(mmin)
                s2 = util.time_to_str(mmax)

        assert ii == 12*3
        assert s1 == '2001-12-01 00:00:00.000'
        assert s2 == '2002-01-01 00:00:00.000'

    def testTimeError(self):
        ok = False
        try:
            util.str_to_time('abc')
        except util.TimeStrError:
            ok = True

        assert ok

    def benchmark_stt_tts(self):
        for x in xrange(2):
            if x == 1:
                util.util_ext = None
            t = util.str_to_time('1999-03-20 20:10:10')
            tt1 = time.time()
            for i in xrange(10000):
                s = util.tts(t)
                t2 = util.stt(s)

            tt2 = time.time()
            print tt2 - tt1

    def test_consistency_merge(self):
        data = [
            ('a', 1, 2, 3.),
            ('a', 2, 2, 3.),
            ('a', 1, 2, 3.)]

        merged = util.consistency_merge(data, error='ignore')
        assert merged == (1, 2, 3.0)

    def test_leap_seconds(self):
        from_sys = {}
        import platform
        if platform.system() != 'Darwin':
            for t, n in util.read_leap_seconds(): # not available on Mac OS X
                from_sys[t] = n

        for t, n in util.read_leap_seconds2():
            if t in from_sys:
                assert from_sys[t] == n

    def test_plf_integration(self):
        import numpy as num

        x = num.array([1., 1., 3., 3.])
        y = num.array([0., 1., 1., 0.])
        x_edges = num.array([0.5, 1.5, 2.5, 3.5])
        yy = util.plf_integrate_piecewise(x_edges, x, y)
        assert num.all(num.abs(yy - num.array([0.5, 1.0, 0.5])) < 1e-6)

        x = num.array([0., 1., 2., 3.])
        y = num.array([0., 1., 1., 0.])
        x_edges = num.array([0., 1., 2., 3.])
        yy = util.plf_integrate_piecewise(x_edges, x, y)
        assert num.all(num.abs(yy - num.array([0.5, 1.0, 0.5])) < 1e-6)

if __name__ == "__main__":
    util.setup_logging('test_util', 'warning')
    unittest.main()
