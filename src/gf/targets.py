#!/bin/python
import numpy as num
import math

from pyrocko.gf import meta
from pyrocko.guts import Timestamp, Tuple, String, Float, Object, StringChoice
from pyrocko.guts_array import Array
from pyrocko.gf.meta import InterpolationMethod

d2r = num.pi / 180.


class BadTarget(Exception):
    pass


class Filter(Object):
    pass


class OptimizationMethod(StringChoice):
    choices = ['enable', 'disable']


class Target(meta.Receiver):
    '''
    A single channel of a computation request including post-processing params.
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
             'If not given the GF store\'s default sample rate is used.'
             'GF store specific restrictions may apply.')

    interpolation = InterpolationMethod.T(
        default='nearest_neighbor',
        help='interpolation method to use')

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
    Multilocation spatial target for static offsets
    '''
    quantity = meta.QuantityType.T(
        optional=True,
        default='displacement',
        help='Measurement quantity type (e.g. "displacement", "pressure", ...)'
             'If not given, it is guessed from the channel code.')

    interpolation = InterpolationMethod.T(
        default='nearest_neighbor',
        help='interpolation method to use')

    tsnapshot = Timestamp.T(
        optional=False,
        default=1,
        help='time of the desired snapshot, '
             'by default first snapshot is taken')

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
        return self.ncoords

    def get_targets(self):
        targets = []
        for i in xrange(self.ntargets):
            targets.append(
                Target(
                    lat=self.coords5[i, 0],
                    lon=self.coords5[i, 1],
                    north_shift=self.coords5[i, 2],
                    east_shift=self.coords5[i, 3],
                    elevation=self.coords5[i, 4]))
        return targets

    def post_process(self, engine, source, statics):
        return meta.StaticResult(result=statics)


class SatelliteTarget(StaticTarget):
    theta = Array.T(
        shape=(None, 1), dtype=num.float,
        help='Line-of-sight incident angle for each location in `coords5`.')

    phi = Array.T(
        shape=(None, 1), dtype=num.float,
        help='Line-of-sight incident angle for each location in `coords5`.')

    def post_process(self, engine, source, statics):
        statics['displacement.los'] =\
            (num.sin(self.theta) * -statics['displacement.d'] +
             num.cos(self.phi) * statics['displacement.e'] +
             num.sin(self.phi) * statics['displacement.n'])
        return meta.StaticResult(result=statics)
