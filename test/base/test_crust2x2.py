import unittest
import numpy as num  # noqa
import logging
import shutil
from tempfile import mkdtemp
from pyrocko.dataset import crust2x2
from pyrocko import util

logger = logging.getLogger('pyrocko.test.test_crust2x2')


class Crust2x2TestCase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = mkdtemp('pyrocko.crust2x2')
        self.db = crust2x2.Crust2()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_crust2(self):
        nprofiles = 25
        lons = num.random.randint(-180, 180, nprofiles)
        lats = num.random.randint(-90, 90, nprofiles)
        for i in range(nprofiles):
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
