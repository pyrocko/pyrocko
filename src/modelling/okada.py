# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
import numpy as num
import logging

from pyrocko.guts import Bool, Float, List, String, Timestamp
from pyrocko.gf import Cloneable, Source
from pyrocko.model import Location
from pyrocko.modelling import disloc_ext

guts_prefix = 'modelling'

logger = logging.getLogger('pyrocko.modelling.okada')

d2r = num.pi / 180.
r2d = 180. / num.pi
km = 1e3


class AnalyticalSource(Location, Cloneable):
    name = String.T(
        optional=True,
        default='')

    time = Timestamp.T(
        default=0.,
        help='source origin time',
        optional=True)

    def __init__(self, **kwargs):
        Location.__init__(self, **kwargs)

    @property
    def northing(self):
        return self.north_shift

    @property
    def easting(self):
        return self.east_shift

    update = Source.update


class AnalyticalRectangularSource(AnalyticalSource):
    '''Rectangular analytical source model
    '''

    strike = Float.T(
        default=0.0,
        help='strike direction in [deg], measured clockwise from north')

    dip = Float.T(
        default=90.0,
        help='dip angle in [deg], measured downward from horizontal')

    rake = Float.T(
        default=0.0,
        help='rake angle in [deg], '
             'measured counter-clockwise from right-horizontal '
             'in on-plane view')

    length = Float.T(
        default=0.,
        help='length of rectangular source area [m]')

    width = Float.T(
        default=0.,
        help='width of rectangular source area [m]')

    slip = Float.T(
        default=0.,
        help='Slip on the rectangular source area [m]',
        optional=True)


class OkadaSource(AnalyticalRectangularSource):
    '''Rectangular Okada source model
    '''

    opening = Float.T(
        default=0.,
        help='Opening of the plane in [m]',
        optional=True)

    nu = Float.T(
        default=0.25,
        help='Poisson\'s ratio, typically 0.25',
        optional=True)

    mu = Float.T(
        default=32e9,
        help='Shear modulus along the plane [Pa]',
        optional=True)

    @property
    def seismic_moment(self):
        '''Scalar Seismic moment
        Code copied from Kite

        Disregarding the opening (as for now)
        We assume a shear modulus of :math:`\mu = 36 \mathrm{GPa}`
        and :math:`M_0 = \mu A D`

        .. important ::

            We assume a perfect elastic solid with :math:`K=\\frac{5}{3}\\mu`

            Through :math:`\\mu = \\frac{3K(1-2\\nu)}{2(1+\\nu)}` this leads to
            :math:`\\mu = \\frac{8(1+\\nu)}{1-2\\nu}`

        :returns: Seismic moment release
        :rtype: float
        '''

        if self.nu and self.mu:
            mu = self.mu
        # elif self.nu and not self.mu:
        #     self.mu = (8. * (1 + self.nu)) / (1 - 2. * self.nu)
        elif self.mu:
            mu = self.mu
        else:
            mu = 32e9  # GPa

        A = self.length * self.width
        return mu * A * self.slip

    @property
    def moment_magnitude(self):
        '''Moment magnitude from Seismic moment
         Code copied from Kite

        We assume :math:`M_\\mathrm{w} = {\\frac{2}{3}}\\log_{10}(M_0) - 10.7`

        :returns: Moment magnitude
        :rtype: float
        '''
        return 2. / 3 * num.log10(self.seismic_moment * 1e7) - 10.7

    def disloc_source(self, dsrc=None):
        if dsrc is None:
            dsrc = num.empty(10)

        dip = self.dip
        if self.dip == 90.:
            dip -= 1e-2

        dsrc[0] = self.length
        dsrc[1] = self.width
        dsrc[2] = self.depth
        dsrc[3] = -dip
        dsrc[4] = self.strike - 180.
        dsrc[5] = self.easting
        dsrc[6] = self.northing

        ss_slip = num.cos(self.rake * d2r) * self.slip
        ds_slip = num.sin(self.rake * d2r) * self.slip
        # print '{:<13}{}\n{:<13}{}'.format(
        #     'strike_slip', ss_slip, 'dip_slip', ds_slip)
        dsrc[7] = -ss_slip  # SS Strike-Slip
        dsrc[8] = -ds_slip  # DS Dip-Slip
        dsrc[9] = self.opening  # TS Tensional-Slip

        return dsrc

    def get_parameters_array(self):
        return num.array([self.__getattribute__(p) for p in self.parameters])

    def set_parameters_array(self, parameter_arr):
        if parameter_arr.size != len(self.parameters):
            raise AttributeError('Invalid number of parameters, %s has %d'
                                 ' parameters'
                                 % self.__name__, len(self.parameters))
        for ip, param in enumerate(self.parameters):
            self.__setattr__(param, parameter_arr[ip])

    @property
    def segments(self):
        yield self


class OkadaSegment(OkadaSource):
    enabled = Bool.T(
        default=True,
        optional=True)


# class OkadaPath(AnalyticalSource):

#     depth = None
#     nu = Float.T(
#         default=0.25,
#         help='Poisson\'s ratio, typically 0.25')
#     nodes = List.T(
#         default=[],
#         optional=True,
#         help='Nodes of the segments as (easting, northing) tuple of [m]')
#     segments__ = List.T(
#         default=[],
#         optional=True,
#         help='List of all segments.')

#     def __init__(self, *args, **kwargs):
#         AnalyticalSource.__init__(self, *args, **kwargs)

#         self._segments = []

#         if not self.nodes:
#             self.nodes.append(
#                 [self.easting, self.northing])

#     @property
#     def segments(self):
#         return self._segments

#     @segments.setter
#     def segments(self, segments):
#         self._segments = segments

#     @staticmethod
#     def _new_segment(e1, n1, e2, n2, **kwargs):
#         d_e = e2 - e1
#         d_n = n2 - n1
#         length = (d_n**2 + d_e**2)**.5
#         '''Width Scaling relation after

#         Leonard, M. (2010). Earthquake fault scaling: Relating rupture length,
#             width, average displacement, and moment release, Bull. Seismol.
#             Soc. Am. 100, no. 5, 1971-1988.
#         '''
#         segment = {
#             'northing': n1 + d_n / 2,
#             'easting': e1 + d_e / 2,
#             'depth': 0.,
#             'length': length,
#             'width': 15. * length**.66,
#             'strike': num.arccos(d_n / length) * r2d,
#             'slip': 45.,
#             'rake': 90.,
#         }
#         segment.update(kwargs)
#         return OkadaSegment(**segment)

#     def _move_segment(self, pos, e1, n1, e2, n2):
#         d_e = e2 - e1
#         d_n = n2 - n1
#         length = (d_n**2 + d_e**2)**.5

#         segment_update = {
#             'northing': n1 + d_n / 2,
#             'easting': e1 + d_e / 2,
#             'length': length,
#             'width': 15. * length**.66,
#             'strike': num.arccos(d_n / length) * r2d,
#         }

#         segment = self.segments[pos]
#         for attr, val in segment_update.items():
#             segment.__setattr__(attr, val)

#     def add_node(self, easting, northing):
#         self.nodes.append([easting, northing])
#         self.segments.append(
#             self._newSegment(
#                 e1=self.nodes[-2][0],
#                 n1=self.nodes[-2][1],
#                 e2=self.nodes[-1][0],
#                 n2=self.nodes[-1][1]))

#     def insert_node(self, pos, easting, northing):
#         self.nodes.insert(pos, [easting, northing])
#         self.segments.append(
#             self._newSegment(
#                 e1=self.nodes[pos][0],
#                 n1=self.nodes[pos][1],
#                 e2=self.nodes[pos + 1][0],
#                 n2=self.nodes[pos + 1][1]))
#         self._moveSegment(
#             pos - 1,
#             e1=self.nodes[pos - 1][0],
#             n1=self.nodes[pos - 1][1],
#             e2=self.nodes[pos][0],
#             n2=self.nodes[pos][1])

#     def move_node(self, pos, easting, northing):
#         self.nodes[pos] = [easting, northing]
#         if pos < len(self):
#             self._moveSegment(
#                 pos,
#                 e1=self.nodes[pos][0],
#                 n1=self.nodes[pos][1],
#                 e2=self.nodes[pos + 1][0],
#                 n2=self.nodes[pos + 1][1])
#         if pos != 0:
#             self._moveSegment(
#                 pos,
#                 e1=self.nodes[pos - 1][0],
#                 n1=self.nodes[pos - 1][1],
#                 e2=self.nodes[pos][0],
#                 n2=self.nodes[pos][1])

#     def __len__(self):
#         return len(self.segments)

#     def disloc_source(self):
#         return num.array([seg.disloc_source() for seg in self.segments
#                           if seg.enabled])


class ProcessorProfile(dict):
    pass


class AnalyticalSourceProcessor(object):
    pass


class DislocProcessor(AnalyticalSourceProcessor):

    @staticmethod
    def process(sources, coords, nthreads=0):
        result = {
            'processor_profile': dict(),
            'displacement.n': num.zeros((coords.shape[0])),
            'displacement.e': num.zeros((coords.shape[0])),
            'displacement.d': num.zeros((coords.shape[0])),
        }

        src_nu = set(src.nu for src in sources)

        for nu in src_nu:
            src_arr = num.vstack([src.disloc_source() for src in sources
                                  if src.nu == nu])
            res = disloc_ext.disloc(src_arr, coords, nu, nthreads)

            result['displacement.e'] += res[:, 0]
            result['displacement.n'] += res[:, 1]
            result['displacement.d'] += -res[:, 2]

        return result


__all__ = [
    'AnalyticalSourceProcessor',
    'DislocProcessor',
    'AnalyticalSource',
    'AnalyticalRectangularSource',
    'OkadaSource']
