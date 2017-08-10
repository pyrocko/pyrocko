import unittest
from os import path
from . import common

common.matplotlib_use_agg()


def get_test_suite():
    return unittest.defaultTestLoader.discover(path.dirname(__file__),
                                               pattern='test_*.py')
