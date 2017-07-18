import unittest
import numpy as num
from pyrocko import topo
from pyrocko import util


class TopoTestCase(unittest.TestCase):

    def test_srtm(self):
        srtm = topo.srtmgl3
        tiles = srtm.available_tilenames()

        srtm.download_tile(list(tiles)[-1])
        srtm.get_tile(0, 0)

    def test_etopo(self):
        topo.etopo1.make_tiles()

    def test_tile(self):
        tile1 = topo.tile.Tile(
            0., 0.,
            1., 1.,
            num.ones((100, 100)))
        tile2 = topo.tile.Tile(
            0., 0.,
            1., 1.,
            num.ones((100, 100)))

        topo.tile.combine([tile1, tile2])


if __name__ == '__main__':
    util.setup_logging('test_topo', 'debug')
    unittest.main()
