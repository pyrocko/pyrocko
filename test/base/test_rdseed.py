from __future__ import division, print_function, absolute_import
import os
import tempfile
import unittest
from .. import common

from pyrocko import util, trace
from pyrocko.io import rdseed
from pyrocko.io import eventdata


class RDSeedTestCase(unittest.TestCase):

    def test_read(self):
        if not rdseed.Programs.check():
            raise unittest.SkipTest('rdseed executeable not found.')

        fpath = common.test_data_file('test_stations.dseed')
        dseed = rdseed.SeedVolumeAccess(fpath)
        pyrocko_stations = dseed.get_pyrocko_stations()
        assert(len(pyrocko_stations) == 1)
        assert(len(dseed.get_pyrocko_events()) == 0)

        tr = trace.Trace(tmax=1., network='CX', station='PB01', channel='HHZ')
        assert(isinstance(dseed.get_pyrocko_response(tr, target='dis'),
                          trace.Evalresp))

        st = dseed.get_pyrocko_station(tr)
        assert(st.network == tr.network)
        assert(st.network == tr.network)

    def test_problems(self):
        p = eventdata.Problems()
        p.add('X', 'y')
        fn = tempfile.mkstemp()[1]
        p.dump(fn)

        pl = eventdata.Problems()
        pl.load(fn)

        os.remove(fn)

        assert p.__dict__ == pl.__dict__


if __name__ == '__main__':
    util.setup_logging('test_rdseed', 'warning')
    unittest.main()
