# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
import numpy as num
import logging

from pyrocko.guts import Bool, Float, Object, String, Timestamp
from pyrocko.guts_array import Array
from pyrocko.gf import Cloneable, Source
from pyrocko.model import Location
from pyrocko.modelling import disloc_ext, okada_ext

guts_prefix = 'modelling'

logger = logging.getLogger('pyrocko.modelling.okada')

d2r = num.pi / 180.
r2d = 180. / num.pi
km = 1e3


class CrackSolutions(Object):
    pass


class GriffithCrack(CrackSolutions):
    width = Float.T(
        help='Width equals to 2*a',
        default=1.)

    poisson = Float.T(
        help='Poisson ratio',
        default=.25)

    shear_mod = Float.T(
        help='Shear modulus [Pa]',
        default=1.e9)

    stressdrop = Array.T(
        help='Stress drop array:'
             '[sig12_r - sig12_c, sig13_r - sig13_c, sig11_r - sig11_c]'
             '[dsig_Strike, dsig_Dip, dsig_Tensile]',
        default=num.array([0., 0., 0.]))

    @property
    def a(self):
        return self.width / 2.

    def disloc_modeI(self, x_obs):
        if type(x_obs) is not num.ndarray:
            x_obs = num.array(x_obs)

        disl = num.zeros((x_obs.shape[0], 3))
        disl[:, 2] = \
            self.stressdrop[2] * num.sqrt(self.a**2 - x_obs**2) * (
            2 * (1 - self.poisson)) / self.shear_mod

        return disl


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

    al1 = Float.T(
        default=0.,
        help='Distance "left" side to source point [m]')

    al2 = Float.T(
        default=0.,
        help='Distance "right" side to source point [m]')

    aw1 = Float.T(
        default=0.,
        help='Distance "lower" side to source point [m]')

    aw2 = Float.T(
        default=0.,
        help='Distance "upper" side to source point [m]')

    slip = Float.T(
        default=0.,
        help='Slip on the rectangular source area [m]',
        optional=True)

    @property
    def length(self):
        return num.abs(self.al1) + num.abs(self.al2)

    @property
    def width(self):
        return num.abs(self.aw1) + num.abs(self.aw2)


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
    def lamb(self):
        return (2 * self.nu * self.mu) / (1 - 2 * self.nu)

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

    def source_patch(self, source_patch=None):
        if source_patch is None:
            source_patch = num.empty(9)

        source_patch[0] = self.northing
        source_patch[1] = self.easting
        source_patch[2] = self.depth
        source_patch[3] = self.strike
        source_patch[4] = self.dip
        source_patch[5] = self.al1
        source_patch[6] = self.al2
        source_patch[7] = self.aw1
        source_patch[8] = self.aw2

        return source_patch

    def source_disloc(self, source_disl=None):
        if source_disl is None:
            source_disl = num.empty(3)

        source_disl[0] = num.cos(self.rake * d2r) * self.slip
        source_disl[1] = num.sin(self.rake * d2r) * self.slip
        source_disl[2] = self.opening

        return source_disl

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


class DislocationInverter(object):
    @staticmethod
    def get_coef_mat(source_patches_list, pure_shear=False):
        source_patches = num.array([
            src.source_patch() for src in source_patches_list])
        receiver_coords = source_patches[:, :3].copy()

        npoints = len(source_patches_list)

        if pure_shear:
            n_eq = 2
        else:
            n_eq = 3

        coefmat = num.zeros((npoints * n_eq, npoints * n_eq))

        def get_normal(strike, dip):
            return num.array([
                -num.sin(strike * d2r) * num.sin(dip * d2r),
                num.cos(strike * d2r) * num.sin(dip * d2r),
                -num.cos(dip * d2r)])

        unit_disl = 1.
        disl_cases = {
            'strikeslip': {
                'slip': unit_disl,
                'opening': 0.,
                'rake': 0.},
            'dipslip': {
                'slip': unit_disl,
                'opening': 0.,
                'rake': 90.},
            'tensileslip': {
                'slip': 0.,
                'opening': unit_disl,
                'rake': 0.}
        }

        for idisl, case_type in enumerate([
                'strikeslip', 'dipslip', 'tensileslip'][:n_eq]):
            case = disl_cases[case_type]
            source_disl = num.array([
                case['slip'] * num.cos(case['rake'] * d2r),
                case['slip'] * num.sin(case['rake'] * d2r),
                case['opening']])

            for isource, source in enumerate(source_patches):
                results = okada_ext.okada(
                    source[num.newaxis, :],
                    source_disl[num.newaxis, :],
                    receiver_coords,
                    source_patches_list[isource].nu,
                    0)

                for irec in range(receiver_coords.shape[0]):
                    eps = num.zeros((3, 3))
                    for m in range(3):
                        for n in range(3):
                            eps[m, n] = 0.5 * (
                                results[irec][m * 3 + n + 3] +
                                results[irec][n * 3 + m + 3])

                    stress_tens = num.zeros((3, 3))
                    delta = num.sum([eps[i, i] for i in range(3)])

                    for m, n in zip([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]):
                        if m == n:
                            stress_tens[m, n] = \
                                source_patches_list[isource].lamb * delta + \
                                2. * source_patches_list[isource].mu * \
                                eps[m, n]

                        else:
                            stress_tens[m, n] = \
                                2. * source_patches_list[isource].mu * \
                                eps[m, n]
                            stress_tens[n, m] = stress_tens[m, n]

                    normal = get_normal(
                        source_patches_list[isource].strike,
                        source_patches_list[isource].dip)

                    for isig in range(n_eq):
                        tension = num.sum(stress_tens[isig, :] * normal)
                        coefmat[irec * n_eq + isig, isource * n_eq + idisl] = \
                            tension / unit_disl

        return num.matrix(coefmat)

    @staticmethod
    def get_disloc_lsq(
            stress_field, coef_mat=None, source_list=None, **kwargs):

        if source_list and not coef_mat:
            coef_mat = DislocationInverter.get_coef_mat(
                source_list, **kwargs)

        if not (coef_mat is None):
            if stress_field.shape[0] == coef_mat.shape[0]:
                coef_mat = num.matrix(coef_mat)

                return num.linalg.inv(
                    coef_mat.T * coef_mat) * coef_mat.T * stress_field


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
    'OkadaSource',
    'DislocationInverter',
    'GriffithCrack']
