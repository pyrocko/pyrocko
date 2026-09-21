# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from pyrocko.guts import Int

from .base import DelayMethod
from ..grid.slowness import SlownessGrid
from ..grid.location import LocationGrid

guts_prefix = 'gato'


class PlaneWaveDM(DelayMethod):

    sign = Int.T(
        default=-1,
        help='Sign of the produced delays. The default, -1 produces '
             'back-azimuth pointing results. +1 is for wave propagation '
             'direction.')

    def calculate(self, source_grid, receiver_grid):

        self._check_type('source_grid', source_grid, SlownessGrid)
        self._check_type('receiver_grid', receiver_grid, LocationGrid)

        slownesses = source_grid.get_nodes('ned')
        ned = receiver_grid.get_nodes('ned')
        return slownesses @ (self.sign * ned).T


__all__ = [
    'PlaneWaveDM',
]
