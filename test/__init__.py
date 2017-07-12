import unittest
from os import path


def get_test_suite():
    return unittest.defaultTestLoader.discover(path.dirname(__file__),
                                               pattern='test_*.py')
