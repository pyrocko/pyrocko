import pyrocko.util

from test_orthodrome import OrthodromeTestCase
from test_io import IOTestCase
from test_pile import PileTestCase
from test_moment_tensor import MomentTensorTestCase
from test_trace import TraceTestCase
from test_model import ModelTestCase
from test_util import UtilTestCase
from test_gf import GFTestCase
from test_gf_sources import GFSourcesTestCase
from test_gf_qseis import GFQSeisTestCase
from test_parimap import ParimapTestCase
from test_response import ResponseTestCase
from test_datacube import DataCubeTestCase
from test_fdsn import FDSNStationTestCase
from test_ims import IMSTestCase
from test_guts import GutsTestCase

import unittest
import argparse
import sys

if __name__ == '__main__':
    pyrocko.util.setup_logging('test_all', 'warning')

    parser = argparse.ArgumentParser() 
    parser.add_argument('--filename',  default=False)
    parser.add_argument('unittest_args', nargs='*')
    args = parser.parse_args()
    if args.filename:
        f = open(args.filename, 'w')
        runner = unittest.TextTestRunner(f)
        sys.argv[1:] = args.unittest_args
        unittest.main(testRunner=runner)
        f.close()
    else:
        unittest.main()
    
