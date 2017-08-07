# python 2/3
import matplotlib
matplotlib.use('Agg')  # noqa
import sys
import unittest
import os
from pyrocko import util
from pyrocko import example
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


def tutorial_run_dir():
    return os.path.join(os.path.split(__file__)[0], 'tutorial_run_dir')


class TutorialsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cwd = os.getcwd()
        cls.run_dir = tutorial_run_dir()
        util.ensuredir(cls.run_dir)
        cls.dn = open(os.devnull, 'w')
        sys.stdout = cls.dn
        os.chdir(cls.run_dir)

    @classmethod
    def tearDownClass(cls):
        cls.dn.close()
        sys.stdout = sys.__stdout__
        os.chdir(cls.cwd)


def make_test_function(m):
    def test(self):
        try:
            __import__('.examples.' + m)

        except example.util.DownloadError:
            raise unittest.SkipTest('could not download required data file')

        except Exception as e:
            self.fail(e)

    return test


for m in to_test:
    test_function = make_test_function(m)
    setattr(TutorialsTestCase, 'test_tutorials_{0}'.format(m), test_function)


if __name__ == '__main__':
    util.setup_logging('test_tutorials', 'warning')

    unittest.main()
