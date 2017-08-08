# python 2/3
import matplotlib
matplotlib.use('Agg')  # noqa
import sys
import unittest
import os
import imp
import glob
import traceback

from pyrocko import util
from pyrocko import example
from matplotlib import pyplot as plt

from pyrocko.gui import snuffler

plt.switch_backend('Agg')
op = os.path

test_dir = op.dirname(op.abspath(__file__))


def snuffle_dummy(*args, **kwargs):
    pass


snuffler.snuffle = snuffle_dummy


def tutorial_run_dir():
    return op.join(test_dir, 'example_run_dir')


class ExamplesTestCase(unittest.TestCase):

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


example_files = glob.glob(op.join(test_dir, 'examples/*.py'))


def make_test_function(test_name, fn):
    def test(self):
        try:
            imp.load_source(test_name, fn)

        except example.util.DownloadError:
            raise unittest.SkipTest('could not download required data file')

        except Exception as e:
            self.fail(traceback.format_exc(e))

    return test


for fn in example_files:
    test_name = op.splitext(op.split(fn)[-1])[0]
    test_function = make_test_function(test_name, fn)
    setattr(ExamplesTestCase, 'test_example_' + test_name, test_function)


if __name__ == '__main__':
    util.setup_logging('test_tutorials', 'warning')
    unittest.main()
