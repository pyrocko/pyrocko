# python 2/3
from __future__ import division, print_function, absolute_import
import sys
import unittest
import os
import glob

from . import common

from pyrocko import ExternalProgramMissing

from pyrocko import util
from pyrocko import example
from pyrocko import pile

from pyrocko.gui import snuffler


op = os.path

test_dir = op.dirname(op.abspath(__file__))


skip_examples = [
    'examples/trace_restitution_dseed.py'
]


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

        cls.snuffle_orig = snuffler.snuffle

        def snuffle_dummy(*args, **kwargs):
            pass

        snuffler.snuffle = snuffle_dummy

        cls.show_progress_force_off_orig = pile.show_progress_force_off
        pile.show_progress_force_off = True

    @classmethod
    def tearDownClass(cls):
        cls.dn.close()
        sys.stdout = sys.__stdout__
        os.chdir(cls.cwd)

        snuffler.snuffle = cls.snuffle_orig
        pile.show_progress_force_off = cls.show_progress_force_off_orig


example_files = [fn for fn in glob.glob(op.join(test_dir, 'examples', '*.py'))
                 if fn not in skip_examples]


def _make_function(test_name, fn):
    def f(self):
        imp = imp2 = None
        try:
            import imp

        except ImportError:
            import importlib.machinery as imp2

        try:
            if imp:
                imp.load_source(test_name, fn)
            else:
                imp2.SourceFileLoader(test_dir, fn)

        except example.util.DownloadError:
            raise unittest.SkipTest('could not download required data file')

        except ExternalProgramMissing as e:
            raise unittest.SkipTest(str(e))

        except Exception as e:
            raise e

    f.__name__ = 'test_example_' + test_name

    return f


for fn in sorted(example_files):
    test_name = op.splitext(op.split(fn)[-1])[0]
    setattr(
        ExamplesTestCase,
        'test_example_' + test_name,
        _make_function(test_name, fn))


if __name__ == '__main__':
    util.setup_logging('test_tutorials', 'warning')
    common.matplotlib_use_agg()
    unittest.main()
