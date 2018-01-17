from __future__ import division, print_function, absolute_import
import unittest
from . import common

from pyrocko import trace

class ObsPyCompatTestCase(unittest.TestCase):

    def test_trace(self):
        fn = common.test_data_file('IRISdata.mseed')
        import obspy
        from pyrocko import obspy_compat
        obspy_compat.plant()

        stream = obspy.read(fn)
        stream.snuffle()


if __name__ == "__main__":
    util.setup_logging('test_obspy_compat', 'warning')
    unittest.main()
