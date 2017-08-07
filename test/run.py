import unittest
import pyrocko.util
from os import path
import matplotlib
matplotlib.use('Agg')


def get_test_suite(loader=None, pattern='test_*.py'):
    if loader is None:
        loader = unittest.defaultTestLoader

    return loader.discover(path.dirname(__file__), pattern)


def load_tests(loader, tests, pattern):
    return get_test_suite(loader)


all = get_test_suite()

gf = get_test_suite(pattern='test_gf_*.py')

if __name__ == '__main__':
    pyrocko.util.setup_logging('test_all', 'warning')
    unittest.main()
