import pyrocko.util

from test_gf import GFTestCase  # noqa
from test_gf_sources import GFSourcesTestCase  # noqa
from test_gf_qseis import GFQSeisTestCase  # noqa
from test_gf_stf import GFSTFTestCase  # noqa

import unittest
import optparse
import sys

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
        unittest.main()
