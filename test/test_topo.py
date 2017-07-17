import unittest
from pyrocko import topo
from pyrocko import util


class TopoTestCase(unittest.TestCase):

    def test_srtm(self):
        srtm = topo.srtmgl3
        tiles = srtm.available_tilenames()

        srtm.download_tile(list(tiles)[-1])

    @unittest.skip('')
    def test_etopo(self):
        etopo = topo.etopo1
        etopo.download()


if __name__ == '__main__':
    util.setup_logging('test_topo', 'debug')
    unittest.main()
