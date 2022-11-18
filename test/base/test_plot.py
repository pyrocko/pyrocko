import unittest

import numpy as num

from pyrocko import plot, util


class UtilTestCase(unittest.TestCase):

    def test_time_tick_labels(self):

        tmin = util.str_to_time('2010-06-05 05:10:02')
        for dt in 10**num.linspace(-5, 9, 100):
            tmax = tmin + dt
            tinc_approx = (tmax - tmin) / 5.

            tinc, tinc_unit = plot.nice_time_tick_inc(tinc_approx)

            times, labels = plot.time_tick_labels(
                tmin, tmax, tinc, tinc_unit)
