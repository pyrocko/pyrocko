from collections import defaultdict
import math
import os
import re
pjoin = os.path.join

import numpy as num

from pyrocko.guts import Object, Float, String, StringChoice, List, Tuple, \
    Timestamp, Int, SObject

from pyrocko.guts_array import Array

from pyrocko import moment_tensor as mt
from pyrocko import trace, model
from pyrocko.gf import meta, store
from pyrocko import trace, model
from pyrocko.gf import meta, store, ws

d2r = math.pi/180.


class BadRequest(Exception):
    pass


class DuplicateStoreId(Exception):
    pass


class NoDefaultStoreSet(Exception):
    pass


class NoSuchStore(BadRequest):
    def __init__(self, store_id):
        Exception.__init__(self)
        self.store_id = store_id

    def __str__(self):
        return 'no GF store with id "%s" found.' % self.store_id


class Filter(Object):
    pass


class Taper(Object):
    pass


def ufloat(s):
    units = {
        'k': 1e3,
        'M': 1e6,
    }

    factor = 1.0
    if s and s[-1] in units:
        factor = units[s[-1]]
        s = s[:-1]
        if not s:
            raise ValueError('unit without a number: \'%s\'' % s)

    return float(s) * factor


def ufloat_or_none(s):
    if s:
        return ufloat(s)
    else:
        return None


def int_or_none(s):
    if s:
        return int(s)
    else:
        return None


class InvalidGridSpec(Exception):
    pass


class RS(SObject):
    '''Convenient range specification

    Equivalent ways to sepecify the range [ 0., 1000., ... 10000. ]:
    
      RS('0 .. 10k : 1k')
      RS(start=0., stop=10*km, step=1*km)
      RS(0, 10e3, 1e3)
      RS('10k .. 100k @ 10')
      RS(start=0., stop=10*km, n=11)
      RS(0, 10e3, n=11)
      RS(values=[ x*1e3 for x in range(11) ])

    Depending on the use context, it can be possible to omit any part of the
    specification. E.g. in the context of extracting a subset of an already
    existing range, the existing range's specification values would be filled
    in where missing. 
    
    The values are distributed with equal spacing, unless the *spacing* argument
    is modified.  The values can be created offset or relative to an external
    base value with the *relative* argument if the use context supports this.

    The range specification can be expressed with a short string representation:

        'start .. stop @ num | spacing, relative'
        'start .. stop : step | spacing, relative'

    most parts of the expression can be omitted if not needed. Whitespace is
    allowed for readability but can also be omitted.
    '''

    start = Float.T(optional=True)
    stop = Float.T(optional=True)
    step = Float.T(optional=True)
    n = Int.T(optional=True)
    values = Array.T(optional=True, dtype=num.float, shape=(None,))
    spacing = StringChoice.T(choices=['lin', 'log', 'symlog'], default='lin', optional=True)
    relative = StringChoice.T(choices=['', 'add', 'mult'], default='', optional=True)

    pattern = re.compile(r'^((?P<start>.*)\.\.(?P<stop>[^@|:]*))?(@(?P<n>[^|]+)|:(?P<step>[^|]+))?(\|(?P<stuff>.+))?$')

    def __init__(self, *args, **kwargs):
        
        d = {}
        if len(args) == 1:
            d = self.parse(args[0])
        elif len(args) in (2,3):
            d['start'], d['stop'] = [float(x) for x in args[:2]]
            if len(args) == 3:
                d['step'] = float(args[2])

        for k,v in kwargs:
            if k in d:
                raise guts.ArgumentError('%s specified more than once' % k) 

            d[k] = v

        SObject.__init__(self, **d)

    def __str__(self):
        def sfloat(x):
            if x is not None:
                return '%g' % x
            else:
                return ''

        if self.values:
            return ','.join('%g' % x for x in self.values)

        if self.start is None and self.stop is None:
            s0 = ''
        else:
            s0 = '%s .. %s' % (sfloat(self.start), sfloat(self.stop))

        s1 = ''
        if self.step is not None:
            s1 = [' : %g',':%g'][s0 == ''] % self.step
        elif self.n is not None:
            s1 = [' @ %i', '@%i'][s0 == ''] % self.n

        if self.spacing == 'lin' and self.relative == '':
            s2 = ''
        else:
            x = []
            if self.spacing != 'lin':
                x.append(self.spacing)

            if self.relative != '':
                x.append(self.relative)

            s2 = ' | %s' % ','.join(x)

        return s0 + s1 + s2

    @classmethod
    def parse(cls, s):
        s = re.sub('\s+', '', s)
        m = cls.pattern.match(s)
        if not m:
            try:
                vals = [ ufloat(x) for x in s.split(',') ]
            except:
                raise InvalidGridSpec('"%s" is not a valid range specification' % s)

            return dict(values=num.array(vals, dtype=num.float))

        d = m.groupdict()
        try:
            start = ufloat_or_none(d['start'])
            stop = ufloat_or_none(d['stop'])
            step = ufloat_or_none(d['step'])
            n = int_or_none(d['n'])
        except:
            raise InvalidGridSpec('"%s" is not a valid range specification' % s)

        spacing = 'lin'
        relative = ''

        if d['stuff'] is not None:
            t = d['stuff'].split(',')
            for x in t:
                if x in cls.spacing.choices:
                    spacing = x
                elif x and x in cls.relative.choices:
                    relative = x
                else:
                    raise InvalidGridSpec('"%s" is not a valid range specification' % s)

        return dict(start=start, stop=stop, step=step, n=n, spacing=spacing, relative=relative)

    def make(self, mi=None, ma=None, inc=None, base=None, eps=1e-5):
        if self.values:
            return self.values

        start = self.start
        stop = self.stop
        step = self.step
        n = self.n

        swap = step is not None and step < 0.
        if start is None:
            start = [mi, ma][swap]
        if stop is None:
            stop = [ma, mi][swap]
        if step is None and inc is not None:
            step = [inc, -inc][ma < mi]

        if start is None or stop is None:
            raise InvalidGridSpec('Cannot use range specification "%s" without start and stop in this context' % self)

        if step is None and n is None:
            step = stop - start

        if n is None:
            if (step < 0) != (stop-start < 0):
                raise InvalidGridSpec('Range specification "%s" has inconsistent ordering (step < 0 => stop > start)' % self)

            n = int(round((stop-start)/step))+1
            stop2 = start + (n-1)*step
            if abs(stop-stop2) > eps:
                n = int(math.floor((stop-start)/step))+1
                stop = start + (n-1)*step
            else:
                stop = stop2

        if start == stop:
            n = 1

        if self.spacing == 'lin':
            vals = num.linspace(start, stop, n)

        elif self.spacing in ('log', 'symlog'):
            if start > 0. and stop > 0.:
                vals = num.exp(num.linspace(num.log(start), num.log(stop), n))
            elif start < 0. and stop < 0.:
                vals = -num.exp(num.linspace(num.log(-start), num.log(-stop), n))
            else:
                raise InvalidGridSpec('log ranges should not include or cross zero (in range specification "%s")' % self)

            if self.spacing == 'symlog':
                nvals = - vals
                vals = num.concatenate((nvals[::-1], vals))

        if self.relative in ('add', 'mult') and base is None:
            raise InvalidGridSpec('cannot use relative range specification in this context')

        if self.relative == 'add':
            vals += base

        if self.relative == 'mult':
            vals *= base

        return vals


class GSE(SObject):
    param = meta.StringID.T()
    rs = RS.T()

    def __init__(self, s=None, **kwargs):
        if s is not None:
            t = s.split('=')
            if len(t) != 2:
                raise InvalidGridSpec('invalid grid specification element: %s' % s)

            sp, sr = t[0].strip(), t[1].strip()

            kwargs['param'] = sp
            kwargs['rs'] = RS(sr)

        SObject.__init__(self, **kwargs)

    def __str__(self):
        return self.param + ' = ' + str(self.rs)


class GS(SObject):
    elements = List.T(GSE.T())

    def __init__(self, s=None, **kwargs):
        if s is not None:
            t = s.splitlines()
            tt = []
            for x in t:
                x = x.strip()
                if x:
                    tt.extend(x.split(';'))

            elements = []
            for se in tt:
                elements.append(GSE(se))

            kwargs['elements'] = elements

        SObject.__init__(self, **kwargs)

    def __str__(self):
        return '; '.join(str(x) for x in self.elements)

class SGrid(object):
    def __init__(self, base, order=None, **params):
        propnames = [prop.name for prop in base.T.properties]

        if order is None:
            orderednames = propnames
        else:
            orderednames = order

        ordered = []
        for k in orderednames:
            if k in params:
                ordered.append((k, params.pop(k)))

        if params:
            for k in sorted(params.keys()):
                ordered.append((k, params.pop(k)))

        self.coords = [(k,v.make(base=getattr(base, k))) for 
                        (k,v) in ordered if isinstance(v, RS)]

        self.fixed = [(k,v) for (k,v) in ordered if not isinstance(v, RS)]
        self.base = base

    def isources(self):
        d = self.base.dict()
        d.update( dict((k,v) for (k,v) in self.fixed))

        for params in permudef(self.coords):
            d.update(dict((k,v) for (k,v) in params))
            s = type(self.base)(**d)
            s.regularize()
            yield s

def permudef(l, j=0):
    if j < len(l):
        k, v = l[j]
        for y in v:
            l[j] = k, y
            for s in permudef(l, j+1):
                yield s

        l[j] = k, v
        return
    else:
        yield l


arr = num.atleast_1d


class Source(meta.Location):
    '''
    Base class for all source models
    '''

    name = String.T(optional=True, default='')
    time = Timestamp.T(default=0.)

    stf = Filter.T(
        optional=True,
        help='source time function as spectral response')

    def grid(self, order=None, **params):
        return SGrid(base=self, order=order, **params)

    def base_key(self):
        return (self.depth, self.lat, self.north_shift,
                self.lon, self.east_shift, type(self))

    def _dparams_base(self):
        return dict(times=arr(0.),
                    lat=self.lat, lon=self.lon,
                    north_shifts=arr(self.north_shift),
                    east_shifts=arr(self.east_shift),
                    depths=arr(self.depth))

    @classmethod
    def provided_components(cls, component_scheme):
        cls = cls.discretized_source_class
        return cls.provided_components(component_scheme)

    def pyrocko_event(self, **kwargs):
        lat, lon = self.effective_latlon
        return model.Event(
            lat=lat,
            lon=lon,
            time=self.time,
            name=self.name,
            depth=self.depth,
            **kwargs)

    @classmethod
    def from_pyrocko_event(cls, ev, **kwargs):
        d = dict(
            name=ev.name,
            time=ev.time,
            lat=ev.lat,
            lon=ev.lon,
            depth=ev.depth)
        d.update(kwargs)
        return cls(**d)

    def dict(self):
        return dict((k,v) for (k,v) in self.T.inamevals(self))


class SourceWithMagnitude(Source):
    '''
    Base class for sources containing a moment magnitude
    '''

    magnitude = Float.T(
        default=6.0,
        help='moment magnitude Mw as in [Hanks and Kanamori, 1979]')

    def __init__(self, **kwargs):
        if 'moment' in kwargs:
            kwargs['magnitude'] = mt.moment_to_magnitude(kwargs.pop('moment'))

        Source.__init__(self, **kwargs)

    @property
    def moment(self):
        return mt.magnitude_to_moment(self.magnitude)

    @moment.setter
    def moment(self, value):
        self.magnitude = mt.moment_to_magnitude(value)

    def pyrocko_event(self, **kwargs):
        return Source.pyrocko_event(
            self,
            magnitude=self.magnitude,
            **kwargs)

    @classmethod
    def from_pyrocko_event(cls, ev, **kwargs):
        d = {}
        if ev.magnitude:
            d.update(magnitude=ev.magnitude)

        d.update(kwargs)
        return super(SourceWithMagnitude, cls).from_pyrocko_event(ev, **d)


class ExplosionSource(SourceWithMagnitude):
    '''
    An isotropic explosion point source.
    '''

    discretized_source_class = meta.DiscretizedExplosionSource

    def base_key(self):
        return Source.base_key(self)

    def get_factor(self):
        return mt.magnitude_to_moment(self.magnitude)

    def discretize_basesource(self, store):
        return meta.DiscretizedExplosionSource(m0s=arr(1.0),
                                               **self._dparams_base())

    def pyrocko_moment_tensor(self):
        m0 = self.moment
        return mt.MomentTensor(m=mt.symmat6(m0, m0, m0, 0., 0., 0.))

    def pyrocko_event(self, **kwargs):
        return SourceWithMagnitude.pyrocko_event(
            self,
            moment_tensor=self.pyrocko_moment_tensor(),
            **kwargs)


class DCSource(SourceWithMagnitude):
    '''
    A double-couple point source.
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

    discretized_source_class = meta.DiscretizedMTSource

    def base_key(self):
        return Source.base_key(self) + (self.strike, self.dip, self.rake)

    def get_factor(self):
        return mt.magnitude_to_moment(self.magnitude)

    def discretize_basesource(self, store):
        mot = mt.MomentTensor(strike=self.strike, dip=self.dip, rake=self.rake)

        ds = meta.DiscretizedMTSource(
            m6s=mot.m6()[num.newaxis, :], **self._dparams_base())

        return ds

    def pyrocko_moment_tensor(self):
        return mt.MomentTensor(
            strike=self.strike,
            dip=self.dip,
            rake=self.rake,
            scalar_moment=self.moment)

    def pyrocko_event(self, **kwargs):
        return SourceWithMagnitude.pyrocko_event(
            self,
            moment_tensor=self.pyrocko_moment_tensor(),
            **kwargs)

    @classmethod
    def from_pyrocko_event(cls, ev, **kwargs):
        d = {}
        mt = ev.moment_tensor
        if mt:
            (strike, dip, rake), _ = mt.both_strike_dip_rake()
            d.update(
                strike=float(strike),
                dip=float(dip),
                rake=float(rake),
                magnitude=float(mt.moment_magnitude()))

        d.update(kwargs)
        return super(DCSource, cls).from_pyrocko_event(ev, **d)


class MTSource(Source):
    '''
    A moment tensor point source.
    '''

    discretized_source_class = meta.DiscretizedMTSource

    mnn = Float.T(
        default=1.,
        help='north-north component of moment tensor in [Nm]')

    mee = Float.T(
        default=1.,
        help='east-east component of moment tensor in [Nm]')

    mdd = Float.T(
        default=1.,
        help='down-down component of moment tensor in [Nm]')

    mne = Float.T(
        default=0.,
        help='north-east component of moment tensor in [Nm]')

    mnd = Float.T(
        default=0.,
        help='north-down component of moment tensor in [Nm]')

    med = Float.T(
        default=0.,
        help='east-down component of moment tensor in [Nm]')

    def __init__(self, **kwargs):
        if 'm6' in kwargs:
            for (k, v) in zip('mnn mee mdd mne mnd med'.split(),
                              kwargs.pop('m6')):
                kwargs[k] = float(v)

        Source.__init__(self, **kwargs)

    @property
    def m6(self):
        return num.array(self.m6_astuple)

    @property
    def m6_astuple(self):
        return (self.mnn, self.mee, self.mdd, self.mne, self.mnd, self.med)

    @m6.setter
    def m6(self, value):
        self.mnn, self.mee, self.mdd, self.mne, self.mnd, self.med = value

    def base_key(self):
        return Source.base_key(self) + self.m6_astuple

    def get_factor(self):
        return 1.0

    def discretize_basesource(self, store):
        return meta.DiscretizedMTSource(m6s=self.m6[num.newaxis, :],
                                        **self._dparams_base())

    def pyrocko_moment_tensor(self):
        return mt.MomentTensor(m=mt.symmat6(*self.m6_astuple))

    def pyrocko_event(self, **kwargs):
        return Source.pyrocko_event(
            self,
            moment_tensor=self.pyrocko_moment_tensor(),
            **kwargs)

    @classmethod
    def from_pyrocko_event(cls, ev, **kwargs):
        d = {}
        mt = ev.moment_tensor
        if mt:
            d.update(m6=map(float, mt.m6()))

        d.update(kwargs)
        return super(MTSource, cls).from_pyrocko_event(ev, **d)


class RingfaultSource(SourceWithMagnitude):
    '''A ring fault with vertical doublecouples.'''

    diameter = Float.T(
        default=1.0,
        help='diameter of the ring in [m]')

    sign = Float.T(
        default=1.0,
        help='inside of the ring moves up (+1) or down (-1)')

    strike = Float.T(
        default=0.0,
        help='strike direction of the ring plane, clockwise from north,'
             ' in [deg]')

    dip = Float.T(
        default=0.0,
        help='dip angle of the ring plane from horizontal in [deg]')

    npointsources = Int.T(
        default=360,
        help='number of point sources to use')

    discretized_source_class = meta.DiscretizedMTSource

    def base_key(self):
        return Source.base_key(self) + (self.strike, self.dip, self.diameter)

    def get_factor(self):
        return self.sign * self.moment

    def discretize_basesource(self, store=None):
        n = self.npointsources
        phi = num.linspace(0, 2.0*num.pi, n, endpoint=False)

        points = num.zeros((n, 3))
        points[:, 0] = num.cos(phi) * 0.5 * self.diameter
        points[:, 1] = num.sin(phi) * 0.5 * self.diameter

        rotmat = num.array(mt.euler_to_matrix(
            self.dip*d2r, self.strike*d2r, 0.0))
        points = num.dot(rotmat.T, points.T).T  # !!! ?

        points[:, 0] += self.north_shift
        points[:, 1] += self.east_shift
        points[:, 2] += self.depth

        m = num.array(mt.MomentTensor(strike=90., dip=90., rake=-90.,
                                      scalar_moment=1.0/n).m())

        rotmats = num.transpose(
            [[num.cos(phi), num.sin(phi), num.zeros(n)],
             [-num.sin(phi), num.cos(phi), num.zeros(n)],
             [num.zeros(n), num.zeros(n), num.ones(n)]], (2, 0, 1))

        ms = num.zeros((n, 3, 3))
        for i in xrange(n):
            mtemp = num.dot(rotmats[i].T, num.dot(m, rotmats[i]))
            ms[i, :, :] = num.dot(rotmat.T, num.dot(mtemp, rotmat))

        m6s = num.vstack((ms[:, 0, 0], ms[:, 1, 1], ms[:, 2, 2],
                          ms[:, 0, 1], ms[:, 0, 2], ms[:, 1, 2])).T

        return meta.DiscretizedMTSource(
            times=num.zeros(n),
            lat=self.lat,
            lon=self.lon,
            north_shifts=points[:, 0],
            east_shifts=points[:, 1],
            depths=points[:, 2],
            m6s=m6s)


class PorePressurePointSource(Source):
    '''
    Excess pore pressure point source

    For poro-elastic initial value problem where an excess pore pressure is
    brought into a small source volume.
    '''

    discretized_source_class = meta.DiscretizedPorePressureSource

    pp = Float.T(
        default=1.0,
        help='initial excess pore pressure in [Pa]')

    def base_key(self):
        return Source.base_key(self)

    def get_factor(self):
        return self.pp

    def discretize_basesource(self, store):
        return meta.DiscretizedPorePressureSource(pp=arr(1.0),
                                                  **self._dparams_base())


class PorePressureLineSource(Source):
    '''
    Excess pore pressure line source

    The line source is centered at (north_shift, east_shift, depth).
    '''

    discretized_source_class = meta.DiscretizedPorePressureSource

    pp = Float.T(
        default=1.0,
        help='initial excess pore pressure in [Pa]')

    length = Float.T(
        default=0.0,
        help='length of the line source [m]')

    azimuth = Float.T(
        default=0.0,
        help='azimuth direction, clockwise from north [deg]')

    dip = Float.T(
        default=90.,
        help='dip direction, downward from horizontal [deg]')

    def base_key(self):
        return Source.base_key(self) + (self.azimuth, self.dip, self.length)

    def get_factor(self):
        return self.pp

    def discretize_basesource(self, store):

        n = 2 * num.ceil(self.length / min(store.config.deltas)) + 1

        a = num.linspace(-0.5*self.length, 0.5*self.length, n)

        sa = math.sin(self.azimuth*d2r)
        ca = math.cos(self.azimuth*d2r)
        sd = math.sin(self.dip*d2r)
        cd = math.cos(self.dip*d2r)

        points = num.zeros((n, 3))
        points[:, 0] = self.north_shift + a * ca * cd
        points[:, 1] = self.east_shift + a * sa * cd
        points[:, 2] = self.depth + a * sd

        return meta.DiscretizedPorePressureSource(
            times=num.zeros(n),
            lat=self.lat,
            lon=self.lon,
            north_shifts=points[:, 0],
            east_shifts=points[:, 1],
            depths=points[:, 2],
            pp=num.ones(n)/n)


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

    pre_taper = Taper.T(
        optional=True,
        help='time domain taper applied to the trace before filtering.')

    filter = Filter.T(
        optional=True,
        help='frequency response filter.')

    post_taper = Taper.T(
        optional=True,
        help='time domain taper applied to the trace after filtering.')

    def base_key(self):
        return (self.store_id, self.sample_rate, self.tmin, self.tmax,
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

        raise BadRequest('cannot guess measurement quantity type from channel '
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

        raise BadRequest('cannot determine sensor component azimuth')

    def effective_dip(self):
        if self.dip is not None:
            return self.dip
        elif self.component_code() in 'NEZ':
            return {'N': 0., 'E': 0., 'Z': -90.}[self.component_code()]

        raise BadRequest('cannot determine sensor component dip')

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


class Reduction(StringChoice):
    choices = ['sum', 'minimum', 'maximum', 'mean', 'variance']


class Request(Object):
    '''
    Synthetic seismogram computation request.

        Request(**kwargs)
        Request(sources, targets, **kwargs)
        Request(sources, targets, reductions, **kwargs)
    '''

    sources = List.T(
        Source.T(),
        help='list of sources for which to produce synthetics.')

    targets = List.T(
        Target.T(),
        help='list of targets for which to produce synthetics.')

    reductions = List.T(
        Reduction.T(),
        help='list of reductions to be applied '
             'target-wise to the synthetics')

    @classmethod
    def args2kwargs(cls, args):
        if len(args) not in (0, 2, 3):
            raise BadRequest('invalid arguments')

        if len(args) == 2:
            return dict(sources=args[0], targets=args[1])
        elif len(args) == 3:
            return dict(sources=args[0], targets=args[1], reductions=args[2])
        else:
            return {}

    def __init__(self, *args, **kwargs):
        kwargs.update(self.args2kwargs(args))
        sources = kwargs.pop('sources', [])
        targets = kwargs.pop('targets', [])
        reductions = kwargs.pop('reductions', [])

        if isinstance(sources, Source):
            sources = [sources]

        if isinstance(targets, Target):
            targets = [targets]

        if isinstance(reductions, Reduction):
            reductions = [reductions]

        Object.__init__(self, sources=sources, targets=targets,
                        reductions=reductions, **kwargs)

    def subsources_map(self):
        m = defaultdict(list)
        for source in self.sources:
            m[source.base_key()].append(source)

        return m

    def subtargets_map(self):
        m = defaultdict(list)
        for target in self.targets:
            m[target.base_key()].append(target)

        return m

    def subrequest_map(self):
        ms = self.subsources_map()
        mt = self.subtargets_map()
        m = {}
        for (ks, ls) in ms.iteritems():
            for (kt, lt) in mt.iteritems():
                m[ks, kt] = (ls, lt)

        return m


class SeismosizerTrace(Object):

    codes = Tuple.T(
        4, String.T(),
        default=('', 'STA', '', 'Z'),
        help='network, station, location and channel codes')

    data = Array.T(
        shape=(None,),
        dtype=num.float32,
        serialize_as='base64',
        serialize_dtype=num.dtype('<f4'),
        help='numpy array with data samples')

    deltat = Float.T(
        default=1.0,
        help='sampling interval [s]')

    tmin = Timestamp.T(
        default=0.0,
        help='time of first sample as a system timestamp [s]')

    def pyrocko_trace(self):
        c = self.codes
        return trace.Trace(c[0], c[1], c[2], c[3],
                           ydata=self.data,
                           deltat=self.deltat,
                           tmin=self.tmin)


class Response(Object):
    '''
    Resonse object to a synthetic seismogram computation request.
    '''

    request = Request.T()
    traces_list = List.T(List.T(SeismosizerTrace.T()))

    def pyrocko_traces(self):
        traces = []
        for trs in self.traces_list:
            for tr in trs:
                traces.append(tr.pyrocko_trace())

        return traces

    def iter_results(self):
        for isource, source in enumerate(self.request.sources):
            for itarget, target in enumerate(self.request.targets):
                yield source, target, \
                    self.traces_list[isource][itarget].pyrocko_trace()

    def snuffle(self):
        trace.snuffle(self.pyrocko_traces())


class Engine(Object):
    '''
    Base class for synthetic seismogram calculators.
    '''

    def get_store_ids(self):
        '''Get list of available GF store IDs'''
        return []


class Rule(object):
    pass


def nonzero(x, eps=1e-15):
    return abs(x) > eps


class VectorRule(Rule):
    def __init__(self, quantity, differentiate=0, integrate=0):
        self.components = [quantity + '.' + c for c in 'ned']
        self.differentiate = differentiate
        self.integrate = integrate

    def required_components(self, target):
        n, e, d = self.components
        sa, ca, sd, cd = target.get_sin_cos_factors()

        comps = []
        if nonzero(ca*cd):
            comps.append(n)

        if nonzero(sa*cd):
            comps.append(e)

        if nonzero(sd):
            comps.append(d)

        return tuple(comps)

    def apply_(self, target, base_seismogram):
        n, e, d = self.components
        sa, ca, sd, cd = target.get_sin_cos_factors()

        if nonzero(ca*cd):
            data = base_seismogram[n].data * (ca*cd)
        else:
            data = 0.0

        if nonzero(sa*cd):
            data = data + base_seismogram[e].data * (sa*cd)

        if nonzero(sd):
            data = data + base_seismogram[d].data * sd

        return data


class HorizontalVectorRule(Rule):
    def __init__(self, quantity, differentiate=0, integrate=0):
        self.components = [quantity + '.' + c for c in 'ne']
        self.differentiate = differentiate
        self.integrate = integrate

    def required_components(self, target):
        n, e = self.components
        sa, ca, _, _ = target.get_sin_cos_factors()

        comps = []
        if nonzero(ca):
            comps.append(n)

        if nonzero(sa):
            comps.append(e)

        return tuple(comps)

    def apply_(self, target, base_seismogram):
        n, e = self.components
        sa, ca, _, _ = target.get_sin_cos_factors()

        if nonzero(ca):
            data = base_seismogram[n].data * ca
        else:
            data = 0.0

        if nonzero(sa):
            data = data + base_seismogram[e].data * sa

        return data


class ScalarRule(object):
    def __init__(self, quantity, differentiate=0):
        self.c = quantity

    def required_components(self, target):
        return (self.c, )

    def apply_(self, target, base_seismogram):
        return base_seismogram[self.c].data.copy()


channel_rules = {
    'displacement': [VectorRule('displacement')],
    'velocity': [VectorRule('displacement', differentiate=1)],
    'pore_pressure': [ScalarRule('pore_pressure')],
    'vertical_tilt': [HorizontalVectorRule('vertical_tilt')],
    'darcy_velocity': [VectorRule('darcy_velocity')]}


class LocalEngine(Engine):
    '''
    Offline synthetic seismogram calculator.

    :param use_env: if ``True``, fill :py:attr:`store_superdirs` and
    :py:attr:`store_dirs` with paths set in environment variables
    GF_STORE_SUPERDIRS AND GF_STORE_DIRS
    '''

    store_superdirs = List.T(
        String.T(),
        help='directories which are searched for Green\'s function stores')

    store_dirs = List.T(
        String.T(),
        help='additional individual Green\'s function store directories')

    default_store_id = String.T(
        optional=True,
        help='default store ID to be used when a request does not provide '
             'one')

    def __init__(self, **kwargs):
        use_env = kwargs.pop('use_env', False)
        Engine.__init__(self, **kwargs)
        if use_env:
            env_store_superdirs = os.environ.get('GF_STORE_SUPERDIRS', '')
            env_store_dirs = os.environ.get('GF_STORE_DIRS', '')
            if env_store_superdirs:
                self.store_superdirs.extend(env_store_superdirs.split(':'))

            if env_store_dirs:
                self.store_dirs.extend(env_store_dirs.split(':'))

        self._id_to_store_dir = {}
        self._open_stores = {}
        self._effective_default_store_id = None

    def _get_store_id(self, store_dir):
        store_ = store.Store(store_dir)
        store_id = store_.config.id
        store_.close()
        return store_id

    def _looks_like_store_dir(self, store_dir):
        return os.path.isdir(store_dir) and \
            all(os.path.isfile(pjoin(store_dir, x)) for x in
                ('index', 'traces', 'config'))

    def iter_store_dirs(self):
        for d in self.store_superdirs:
            for entry in os.listdir(d):
                store_dir = pjoin(d, entry)
                if self._looks_like_store_dir(store_dir):
                    yield store_dir

        for store_dir in self.store_dirs:
            yield store_dir

    def _scan_stores(self):
        for store_dir in self.iter_store_dirs():
            store_id = self._get_store_id(store_dir)
            if store_id not in self._id_to_store_dir:
                self._id_to_store_dir[store_id] = store_dir
            else:
                if store_dir != self._id_to_store_dir[store_id]:
                    raise DuplicateStoreId(
                        'GF store ID %s is used in (at least) two '
                        'different stores. Locations are: %s and %s' %
                        (store_id, self._id_to_store_dir[store_id], store_dir))

    def get_store_dir(self, store_id):
        '''
        Lookup directory given a GF store ID.
        '''

        if store_id not in self._id_to_store_dir:
            self._scan_stores()

        if store_id not in self._id_to_store_dir:
            raise NoSuchStore(store_id)

        return self._id_to_store_dir[store_id]

    def get_store_ids(self):
        '''Get list of available store IDs.'''

        self._scan_stores()
        return sorted(self._id_to_store_dir.keys())

    def effective_default_store_id(self):
        if self._effective_default_store_id is None:
            if self.default_store_id is None:
                store_ids = self.get_store_ids()
                if len(store_ids) == 1:
                    self._effective_default_store_id = self.get_store_ids()[0]
                else:
                    raise NoDefaultStoreSet()
            else:
                self._effective_default_store_id = self.default_store_id

        return self._effective_default_store_id

    def get_store(self, store_id):
        if store_id is None:
            store_id = self.effective_default_store_id()

        if store_id not in self._open_stores:
            store_dir = self.get_store_dir(store_id)
            self._open_stores[store_id] = store.Store(store_dir)

        return self._open_stores[store_id]

    def get_store_config(self, store_id):
        store = self.get_store(store_id)
        return store.config

    def get_store_extra(self, store_id, key):
        store = self.get_store(store_id)
        return store.get_extra(key)

    def channel_rule(self, source, target):
        store_ = self.get_store(target.store_id)
        cprovided = source.provided_components(store_.config.component_scheme)
        quantity = target.effective_quantity()
        try:
            for rule in channel_rules[quantity]:
                cneeded = rule.required_components(target)
                if all(c in cprovided for c in cneeded):
                    return rule

        except KeyError:
            pass

        raise BadRequest(
            'no rule to calculate "%s" with GFs from store "%s" '
            'for source model "%s"' % (
                target.effective_quantity(),
                target.store_id,
                source.__class__.__name__))

    def base_seismogram(self, source, target, components):
        store_ = self.get_store(target.store_id)
        receiver = target.receiver(store_)
        base_source = source.discretize_basesource(store_)
        base_seismogram = store_.seismogram(base_source, receiver, components)
        return store.make_same_span(base_seismogram)

    def _post_process(self, base_seismogram, source, target):
        deltat = base_seismogram.values()[0].deltat

        rule = self.channel_rule(source, target)
        data = rule.apply_(target, base_seismogram)

        factor = source.get_factor() * target.get_factor()
        if factor != 1.0:
            data *= factor

        tr = SeismosizerTrace(
            codes=target.codes,
            data=data,
            deltat=deltat,
            tmin=base_seismogram.values()[0].itmin * deltat + source.time)

        return tr

    def process(self, *args, **kwargs):
        '''Process a request.

            process(**kwargs)
            process(request, **kwargs)
            process(sources, targets, **kwargs)
            process(sources, targets, reductions, **kwargs)

        The request can be given a a :py:class:`Request` object, or such an
        object is created using ``Request(**kwargs)`` for convenience.
        '''

        if len(args) not in (0, 1, 2, 3):
            raise BadRequest('invalid arguments')

        if len(args) == 1:
            kwargs['request'] = args[0]

        elif len(args) >= 2:
            kwargs.update(Request.args2kwargs(args))

        request = kwargs.pop('request', None)
        status_callback = kwargs.pop('status_callback', None)

        if request is None:
            request = Request(**kwargs)

        source_index = dict((x, i) for (i, x) in enumerate(request.sources))
        target_index = dict((x, i) for (i, x) in enumerate(request.targets))

        m = request.subrequest_map()
        skeys = sorted(m.keys())
        traces = []
        for i in xrange(len(request.sources)):
            traces.append([None] * len(request.targets))

        n = len(skeys)
        for i, k in enumerate(skeys):
            if status_callback:
                status_callback(i, n)

            sources, targets = m[k]
            components = set()
            for target in targets:
                rule = self.channel_rule(sources[0], target)
                components.update(rule.required_components(target))

            base_seismogram = self.base_seismogram(
                sources[0],
                targets[0],
                components)

            for source in sources:
                for target in targets:
                    tr = self._post_process(base_seismogram, source, target)
                    traces[source_index[source]][target_index[target]] = tr

        if status_callback:
            status_callback(n, n)

        return Response(request=request, traces_list=traces)


class RemoteEngine(Engine):
    '''
    Client for remote synthetic seismogram calculator.
    '''

    site = String.T(default=ws.g_default_site, optional=True)
    url = String.T(default=ws.g_url, optional=True)

    def process(self, request=None, status_callback=None, **kwargs):

        if request is None:
            request = Request(**kwargs)

        return ws.seismosizer(url=self.url, site=self.site, request=request)
        


def get_engine():
    return LocalEngine(use_env=True)

source_classes = [
    Source,
    SourceWithMagnitude,
    ExplosionSource,
    DCSource,
    MTSource,
    RingfaultSource,
    PorePressurePointSource,
    PorePressureLineSource
]

__all__ = '''
BadRequest
NoSuchStore
Filter
Taper
'''.split() + [S.__name__ for S in source_classes] + '''
Target
Reduction
Request
SeismosizerTrace
Response
Engine
LocalEngine
RemoteEngine
source_classes
get_engine
RS
SGrid
'''.split()
