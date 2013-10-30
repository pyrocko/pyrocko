import pyrocko.util

from test_orthodrome import OrthodromeTestCase
from test_io import IOTestCase
from test_pile import PileTestCase
from test_moment_tensor import MomentTensorTestCase
from test_trace import TraceTestCase
from test_model import ModelTestCase
from test_util import UtilTestCase
from test_gf import GFTestCase

import unittest

if __name__ == '__main__':
    pyrocko.util.setup_logging('test_all', 'warning')
    unittest.main()
    
