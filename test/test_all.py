import unittest
import optparse
import sys
import pyrocko.util

import matplotlib
matplotlib.use('Agg')  # noqa

from os import path


def get_test_suite():
    return unittest.defaultTestLoader.discover(path.dirname(__file__),
                                               pattern='test_*.py')


if __name__ == '__main__':
    pyrocko.util.setup_logging('test_all', 'warning')

    parser = optparse.OptionParser()
    parser.add_option('--filename', dest='filename')
    options, args = parser.parse_args()
    if options.filename:
        f = open(options.filename, 'w')
        runner = unittest.TextTestRunner(f)
        sys.argv[1:] = args
        unittest.main(testRunner=runner)
        f.close()
    else:
        suite = get_test_suite()
        unittest.TextTestRunner(verbosity=2).run(suite)
