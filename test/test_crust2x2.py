import unittest
import numpy as num  # noqa
import logging
import shutil
from tempfile import mkdtemp
from pyrocko import crust2x2, util

logger = logging.getLogger('test_crust2x2.py')


class Crust2x2TestCase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = mkdtemp('pyrocko.crust2x2')
        self.db = crust2x2.Crust2()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_crust2(self):
        nprofiles = 25
        lons = num.random.random_integers(-180, 180, nprofiles)
        lats = num.random.random_integers(-90, 90, nprofiles)
        for i in xrange(nprofiles):
            self.db.get_profile(lats[i], lons[i])

    def test_profiles(self):
        p = self.db.get_profile(25, 30)
        p.elevation()
        p.get_layer(1)
        p.get_weeded()
        p.averages()


if __name__ == "__main__":
    util.setup_logging('test_crust2x2', 'info')
    unittest.main()
