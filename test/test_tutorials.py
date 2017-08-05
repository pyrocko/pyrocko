# python 2/3
import matplotlib
from distutils.dir_util import copy_tree
import tempfile
import sys
import shutil
import unittest
import os
from pyrocko import util
import pyrocko.tutorials as tutorials
from matplotlib import pyplot as plt

plt.switch_backend('Agg')

to_test = [
    'automap_example',
    'markers_example1',
    'trace_handling_example_pz',
    'beachball_example01',
    'beachball_example02',
    'beachball_example03',
    'beachball_example04',
    # 'gf_forward_example1',  # Takes long...
    'gf_forward_example2',
    'gf_forward_example3',
    'gf_forward_example4',
    'gshhg_example',
    'tectonics_example',
    'test_response_plot',
    'cake_raytracing',
    'catalog_search_globalcmt',
    'catalog_search_geofon',
    'make_hour_files',
    'convert_mseed_sac',
    'make_hour_files',
    'fdsn_request_geofon',
    'fdsn_stationxml_modify',
    'guts_usage',
    'orthodrome_example1',
    'orthodrome_example2',
    'orthodrome_example3',
    'orthodrome_example4',
    'orthodrome_example5',
]


class TestCases(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.tmpdir = tempfile.mkdtemp('pyrocko.tutorials')
        sys.path.append(self.tmpdir)
        p = tutorials.__path__[0]
        copy_tree(p, self.tmpdir)
        self.dn = open(os.devnull, 'w')
        sys.stdout = self.dn
        os.chdir(self.tmpdir)

    def tearDown(self):
        self.dn.close()
        sys.stdout = sys.__stdout__
        shutil.rmtree(self.tmpdir)
        os.chdir(self.cwd)


def make_test_function(m):
    def test(self):
        try:
            __import__(m)
        except Exception as e:
            self.fail(e)
    return test


if __name__ == '__main__':
    util.setup_logging('test_tutorials', 'warning')

    for m in to_test:
        test_function = make_test_function(m)
        setattr(TestCases, 'test_{0}'.format(m), test_function)

    unittest.main()
