import unittest
from pyrocko import topo
from pyrocko import util


class TopoTestCase(unittest.TestCase):

    def test_srtm(self):
        srtm = topo.srtmgl3
        tiles = srtm.available_tilenames()

        srtm.download_tile(list(tiles)[-1])
        srtm.get_tile(0, 0)

    def test_etopo(self):
        etopo = topo.etopo1
        etopo.download()
        etopo.make_tiles()
        etopo.get_tile(0, 0)


if __name__ == '__main__':
    util.setup_logging('test_topo', 'debug')
    unittest.main()
