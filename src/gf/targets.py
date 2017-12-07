# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
from __future__ import absolute_import, division
from builtins import range

import numpy as num
import math

from . import meta
from pyrocko.guts import Timestamp, Tuple, String, Float, Object, StringChoice
from pyrocko.guts_array import Array
from pyrocko.model import gnss

d2r = num.pi / 180.


class BadTarget(Exception):
    pass


class Filter(Object):
    pass


class OptimizationMethod(StringChoice):
    choices = ['enable', 'disable']


def component_orientation(source, target, component):
    '''
    Get component and azimuth for standard components R, T, Z, N, and E.

    :param source: :py:class:`pyrocko.gf.Location` object
    :param target: :py:class:`pyrocko.gf.Location` object
    :param component: string ``'R'``, ``'T'``, ``'Z'``, ``'N'`` or ``'E'``
    '''

    _, bazi = source.azibazi_to(target)

    azi, dip = {
        'T': (bazi + 270., 0.),
        'R': (bazi + 180., 0.),
        'N': (0., 0.),
        'E': (90., 0.),
        'Z': (0., -90.)}[component]


class Target(meta.Receiver):
    '''
    A seismogram computation request for a single component, including
    its post-processing parmeters.
    '''

    quantity = meta.QuantityType.T(
        optional=True,
        help='Measurement quantity type (e.g. "displacement", "pressure", ...)'
             'If not given, it is guessed from the channel code.')

    codes = Tuple.T(
        4, String.T(), default=('', 'STA', '', 'Z'),
        help='network, station, location and channel codes to be set on '
             'the response trace.')

    elevation = Float.T(
        default=0.0,
        help='station surface elevation in [m]')

    store_id = meta.StringID.T(
        optional=True,
        help='ID of Green\'s function store to use for the computation. '
             'If not given, the processor may use a system default.')

    sample_rate = Float.T(
        optional=True,
        help='sample rate to produce. '
             'If not given the GF store\'s default sample rate is used. '
             'GF store specific restrictions may apply.')

    interpolation = meta.InterpolationMethod.T(
        default='nearest_neighbor',
        help='Interpolation method between Green\'s functions. Supported are'
             ' ``nearest_neighbor`` and ``multilinear``')

    optimization = OptimizationMethod.T(
        default='enable',
        optional=True,
        help='disable/enable optimizations in weight-delay-and-sum operation')

    tmin = Timestamp.T(
        optional=True,
        help='time of first sample to request in [s]. '
             'If not given, it is determined from the Green\'s functions.')

    tmax = Timestamp.T(
        optional=True,
        help='time of last sample to request in [s]. '
             'If not given, it is determined from the Green\'s functions.')

    azimuth = Float.T(
        optional=True,
        help='azimuth of sensor component in [deg], clockwise from north. '
             'If not given, it is guessed from the channel code.')

    dip = Float.T(
        optional=True,
        help='dip of sensor component in [deg], '
             'measured downward from horizontal. '
             'If not given, it is guessed from the channel code.')

    filter = Filter.T(
        optional=True,
        help='frequency response filter.')

    def base_key(self):
        return (self.store_id, self.sample_rate, self.interpolation,
                self.optimization,
                self.tmin, self.tmax,
                self.elevation, self.depth, self.north_shift, self.east_shift,
                self.lat, self.lon)

    def effective_quantity(self):
        if self.quantity is not None:
            return self.quantity

        # guess from channel code
        cha = self.codes[-1].upper()
        if len(cha) == 3:
            # use most common SEED conventions here, however units have to be
            # guessed, because they are not uniquely defined by the conventions
            if cha[-2] in 'HL':  # high gain, low gain seismometer
                return 'velocity'
            if cha[-2] == 'N':   # accelerometer
                return 'acceleration'
            if cha[-2] == 'D':   # hydrophone, barometer, ...
                return 'pressure'
            if cha[-2] == 'A':   # tiltmeter
                return 'tilt'
        elif len(cha) == 2:
            if cha[-2] == 'U':
                return 'displacement'
            if cha[-2] == 'V':
                return 'velocity'
        elif len(cha) == 1:
            if cha in 'NEZ':
                return 'displacement'
            if cha == 'P':
                return 'pressure'

        raise BadTarget('cannot guess measurement quantity type from channel '
                        'code "%s"' % cha)

    def receiver(self, store):
        rec = meta.Receiver(**dict(meta.Receiver.T.inamevals(self)))
        return rec

    def component_code(self):
        cha = self.codes[-1].upper()
        if cha:
            return cha[-1]
        else:
            return ' '

    def effective_azimuth(self):
        if self.azimuth is not None:
            return self.azimuth
        elif self.component_code() in 'NEZ':
            return {'N': 0., 'E': 90., 'Z': 0.}[self.component_code()]

        raise BadTarget('cannot determine sensor component azimuth for '
                        '%s.%s.%s.%s' % self.codes)

    def effective_dip(self):
        if self.dip is not None:
            return self.dip
        elif self.component_code() in 'NEZ':
            return {'N': 0., 'E': 0., 'Z': -90.}[self.component_code()]

        raise BadTarget('cannot determine sensor component dip')

    def get_sin_cos_factors(self):
        azi = self.effective_azimuth()
        dip = self.effective_dip()
        sa = math.sin(azi*d2r)
        ca = math.cos(azi*d2r)
        sd = math.sin(dip*d2r)
        cd = math.cos(dip*d2r)
        return sa, ca, sd, cd

    def get_factor(self):
        return 1.0

    def post_process(self, engine, source, tr):
        return meta.Result(trace=tr)


class StaticTarget(meta.MultiLocation):
    '''
    A computation request for a spatial multi-location target of
    static/geodetic quantities.
    '''
    quantity = meta.QuantityType.T(
        optional=True,
        default='displacement',
        help='Measurement quantity type, for now only `displacement` is'
             'supported.')

    interpolation = meta.InterpolationMethod.T(
        default='nearest_neighbor',
        help='Interpolation method between Green\'s functions. Supported are'
             ' ``nearest_neighbor`` and ``multilinear``')

    tsnapshot = Timestamp.T(
        optional=True,
        help='time of the desired snapshot in [s], '
             'If not given, the first sample is taken. If the desired sample'
             ' exceeds the length of the Green\'s function store,'
             ' the last sample is taken.')

    store_id = meta.StringID.T(
        optional=True,
        help='ID of Green\'s function store to use for the computation. '
             'If not given, the processor may use a system default.')

    def base_key(self):
        return (self.store_id,
                self.coords5.shape,
                self.quantity,
                self.tsnapshot,
                self.interpolation)

    @property
    def ntargets(self):
        ''' Number of targets held by instance. '''
        return self.ncoords

    def get_targets(self):
        ''' Discretizes the multilocation target into a list of
        :class:`Target:`

        :returns: :class:`Target`
        :rtype: list
        '''
        targets = []
        for i in range(self.ntargets):
            targets.append(
                Target(
                    lat=float(self.coords5[i, 0]),
                    lon=float(self.coords5[i, 1]),
                    north_shift=float(self.coords5[i, 2]),
                    east_shift=float(self.coords5[i, 3]),
                    elevation=float(self.coords5[i, 4])))
        return targets

    def post_process(self, engine, source, statics):
        return meta.StaticResult(result=statics)


class SatelliteTarget(StaticTarget):
    '''
    A computation request for a spatial multi-location target of
    static/geodetic quantities measured from a satellite instrument.
    The line of sight angles are provided and projecting
    post-processing is applied.
    '''
    theta = Array.T(
        shape=(None,), dtype=num.float,
        help='Horizontal angle towards satellite\'s line of sight in radians.'
             '\n\n        .. important::\n\n'
             '            :math:`0` is **east** and'
             ' :math:`\\frac{\\pi}{2}` is **north**.\n\n')

    phi = Array.T(
        shape=(None,), dtype=num.float,
        help='Theta is look vector elevation angle towards satellite from'
             ' horizon in radians. Matrix of theta towards satellite\'s'
             ' line of sight.'
             '\n\n        .. important::\n\n'
             '            :math:`-\\frac{\\pi}{2}` is **down** and'
             ' :math:`\\frac{\\pi}{2}` is **up**.\n\n')

    def __init__(self, *args, **kwargs):
        StaticTarget.__init__(self, *args, **kwargs)
        self._los_factors = None

    def get_los_factors(self):
        if (self.theta.size != self.phi.size != self.lats.size):
            raise AttributeError('LOS angles inconsistent with provided'
                                 ' coordinate shape.')
        if self._los_factors is None:
            self._los_factors = num.empty((self.theta.shape[0], 3))
            self._los_factors[:, 0] = num.sin(self.theta)
            self._los_factors[:, 1] = num.cos(self.theta) * num.cos(self.phi)
            self._los_factors[:, 2] = num.cos(self.theta) * num.sin(self.phi)
        return self._los_factors


class GNSSCampaignTarget(StaticTarget):

    def post_process(self, engine, source, statics):
        campaign = gnss.GNSSCampaign()

        for ista in range(self.ntargets):
            north = gnss.GNSSComponent(
                shift=float(statics['displacement.n'][ista]))
            east = gnss.GNSSComponent(
                shift=float(statics['displacement.e'][ista]))
            up = gnss.GNSSComponent(
                shift=-float(statics['displacement.d'][ista]))

            coords = self.coords5
            station = gnss.GNSSStation(
                lat=float(coords[ista, 0]),
                lon=float(coords[ista, 1]),
                east_shift=float(coords[ista, 2]),
                north_shift=float(coords[ista, 3]),
                elevation=float(coords[ista, 4]),
                north=north,
                east=east,
                up=up)

            campaign.add_station(station)

        return meta.GNSSCampaignResult(result=statics, campaign=campaign)
