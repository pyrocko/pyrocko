from pyrocko.dataset.slab2 import Slab2, get_slab_prefixes

import numpy as num
import unittest

from pyrocko import util, guts


class Slab2TestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):

        unittest.TestCase.__init__(self, *args, **kwargs)
        self.slabs = Slab2()
        self.prefixes = get_slab_prefixes()

    def test_get_slab_data(self):

        slab_data = self.slabs.get_slab_data(
            self.prefixes[0], force_rewrite=False)
        print(slab_data, slab_data.shape)

    def test_get_slab_geometry(self):
        self.slabs.get_slab_geometry(self.prefixes[0])


if __name__ == '__main__':
    util.setup_logging('test_slab2', 'debug')
    unittest.main()