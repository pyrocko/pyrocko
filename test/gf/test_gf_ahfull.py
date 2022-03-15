from __future__ import division, print_function, absolute_import
import sys
import random  # noqa
import math
import unittest
from tempfile import mkdtemp
import logging
import shutil

import numpy as num

from pyrocko import gf, util
from pyrocko.fomosto import ahfullgreen


logger = logging.getLogger('pyrocko.test.test_gf_ahfull')


r2d = 180. / math.pi
d2r = 1.0 / r2d
km = 1000.


def numeq(a, b, eps):
    return (num.all(num.asarray(a).shape == num.asarray(b).shape and
            num.abs(num.asarray(a) - num.asarray(b)) < eps))


class GFAhfullTestCase(unittest.TestCase):

    tempdirs = []
    if sys.version_info < (2, 7):
        from contextlib import contextmanager

        @contextmanager
        def assertRaises(self, exc):

            gotit = False
            try:
                yield None
            except exc:
                gotit = True

            assert gotit, 'expected to get a %s exception' % exc

        def assertIsNone(self, value):
            assert value is None, 'expected None but got %s' % value

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    @classmethod
    def tearDownClass(cls):
        for d in cls.tempdirs:
            shutil.rmtree(d)

    def test_create_default(self):
        d = mkdtemp(prefix='gfstore')
        self.tempdirs.append(d)
        ahfullgreen.init(d, None)
        store = gf.Store(d)
        store.make_travel_time_tables()
        ahfullgreen.build(d)


if __name__ == '__main__':
    util.setup_logging('test_gf_ahfull', 'warning')
    unittest.main()
