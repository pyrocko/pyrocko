import pyrocko.util

from test_orthodrome import OrthodromeTestCase  # noqa
from test_io import IOTestCase  # noqa
from test_pile import PileTestCase  # noqa
from test_moment_tensor import MomentTensorTestCase  # noqa
from test_trace import TraceTestCase  # noqa
from test_model import ModelTestCase  # noqa
from test_util import UtilTestCase  # noqa
from test_gf import GFTestCase  # noqa
from test_gf_sources import GFSourcesTestCase  # noqa
from test_gf_qseis import GFQSeisTestCase  # noqa
from test_gf_stf import GFSTFTestCase  # noqa
from test_parimap import ParimapTestCase  # noqa
from test_response import ResponseTestCase  # noqa
from test_datacube import DataCubeTestCase  # noqa
from test_fdsn import FDSNStationTestCase  # noqa
from test_ims import IMSTestCase  # noqa
from test_guts import GutsTestCase  # noqa
from test_parstack import ParstackTestCase  # noqa
from test_geonames import GeonamesTestCase  # noqa
from test_cake import CakeTestCase  # noqa
from test_beachball import BeachballTestCase  # noqa
from test_gmtpy import GmtPyTestCase  # noqa
from test_automap import AutomapTestCase  # noqa

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
