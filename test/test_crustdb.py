import unittest
import numpy as num  # noqa
import logging
import shutil
from tempfile import mkdtemp
from os.path import join as pjoin
from pyrocko import crustdb, util

logger = logging.getLogger('test_gf_static.py')


class CrustDBTestCase(unittest.TestCase):

    def tearDown(self):
        return
        shutil.rmtree(self.tempdir)

    def test_database(self):
        self.tmpdir = mkdtemp('pyrocko.crustdb')
        db = crustdb.CrustDB(
            '/home/marius/Development/crustshot/data/gsc20130501.txt')
        tmpmap = pjoin(self.tmpdir, 'map.ps')
        print tmpmap
        db.plotMap(tmpmap, show_topo=False)


if __name__ == "__main__":
    util.setup_logging('test_crustdb', 'warning')
    unittest.main()
