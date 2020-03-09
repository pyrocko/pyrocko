from __future__ import division, print_function, absolute_import

import unittest
import numpy as num
from pyrocko import gf, util

km = 1e3


class GFSourceTypesTestCase(unittest.TestCase):

    def test_rectangular_source(self):
        # WIP
        nsrc = 5
        rect_sources = []

        for n in range(nsrc):
            src = gf.RectangularSource(
                lat=0., lon=0.,
                anchor='bottom',
                north_shift=5000., east_shift=9000., depth=4.*km,
                width=2.*km, length=8.*km,
                dip=0., rake=0., strike=(180./nsrc + 1) * n,
                slip=1.)
            rect_sources.append(src)

    @staticmethod
    def plot_rectangular_source(src, store):
        from matplotlib import pyplot as plt
        from matplotlib.patches import Polygon
        ax = plt.gca()
        ne = src.outline(cs='xy')
        p = Polygon(num.fliplr(ne), fill=False, color='r', alpha=.7)
        ax.add_artist(p)

        mt = src.discretize_basesource(store)
        ax.scatter(mt.east_shifts, mt.north_shifts, alpha=1)
        ax.scatter(src.east_shift, src.north_shift, color='r')

        plt.axis('equal')
        plt.show()


if __name__ == '__main__':
    util.setup_logging('test_gf_source_types', 'warning')
    unittest.main(defaultTest='GFSourceTypesTestCase')
