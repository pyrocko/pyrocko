from pyrocko import rdseed
from pyrocko import util
import unittest
import common


@unittest.skipIf(rdseed.Programs.check(), 'rdseed executeable not found.')
class RDSeedTestCase(unittest.TestCase):

    def test_read(self):
        fpath = common.test_data_file('test_stations.dseed')
        dseed = rdseed.SeedVolumeAccess(fpath)
        pyrocko_stations = dseed._get_stations_from_file()
        assert(len(pyrocko_stations) == 1)
        assert(dseed.get_pile().is_empty())


if __name__ == '__main__':
    util.setup_logging('test_rdseed', 'warning')
    unittest.main()
