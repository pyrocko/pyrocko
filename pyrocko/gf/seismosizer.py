from collections import defaultdict
import math
import os
pjoin = os.path.join

import numpy as num

from guts import Object, Float, String, StringChoice, List, Tuple, Timestamp, \
    Int

from guts_array import Array

from pyrocko import moment_tensor as mt
from pyrocko import trace
from pyrocko.gf import meta, store

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


arr = num.atleast_1d


class Source(meta.Location):
    '''
    Base class for all source models
    '''

    time = Timestamp.T(default=0.)

    stf = Filter.T(
        optional=True,
        help='source time function as spectral response')

    def base_key(self):
        return (self.depth, self.lat, self.north_shift,
                self.lon, self.east_shift, type(self))

    def _dparams_base(self):
        return dict(times=arr(0.),
                    lat=self.lat, lon=self.lon,
                    north_shifts=arr(self.north_shift),
                    east_shifts=arr(self.east_shift),
                    depths=arr(self.depth))


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


class ExplosionSource(SourceWithMagnitude):
    '''
    An isotropic explosion point source.
    '''

    def base_key(self):
        return Source.base_key(self)

    def get_factor(self):
        return mt.magnitude_to_moment(self.magnitude)

    def discretize_basesource(self, store):
        return meta.DiscretizedExplosionSource(m0s=arr(1.0),
                                               **self._dparams_base())


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

    def base_key(self):
        return Source.base_key(self) + (self.strike, self.dip, self.rake)

    def get_factor(self):
        return mt.magnitude_to_moment(self.magnitude)

    def discretize_basesource(self, store):
        mot = mt.MomentTensor(strike=self.strike, dip=self.dip, rake=self.rake)

        ds = meta.DiscretizedMTSource(
            m6s=mot.m6()[num.newaxis, :], **self._dparams_base())

        return ds


class MTSource(Source):
    '''
    A moment tensor point source.
    '''

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


class Target(meta.Receiver):
    '''
    A single channel of a computation request including post-processing params.
    '''

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

    def component_code(self):
        if self.codes[-1]:
            return self.codes[-1][-1].upper()

        raise BadRequest('cannot get component code')

    def receiver(self, store):
        rec = meta.Receiver(**dict(meta.Receiver.T.inamevals(self)))
        return rec

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

    def get_factor(self):
        return 1.0


class Reduction(StringChoice):
    choices = ['sum', 'minimum', 'maximum', 'mean', 'variance']


class Request(Object):
    '''
    Synthetic seismogram computation request.
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


class Engine(Object):
    '''
    Base class for synthetic seismogram calculators.
    '''

    def get_store_ids(self):
        '''Get list of available GF store IDs'''
        return []


class LocalEngine(Engine):
    '''
    Offline synthetic seismogram calculator.
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
        Engine.__init__(self, **kwargs)
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

    def base_seismogram(self, source, target):
        store_ = self.get_store(target.store_id)
        receiver = target.receiver(store_)
        base_source = source.discretize_basesource(store_)
        return store.make_same_span(store_.seismogram(base_source, receiver))

    def _post_process(self, base_seismogram, source, target):
        deltat = base_seismogram[0].deltat
        if len(base_seismogram) == 3:
            ndata, edata, zdata = [x.data for x in base_seismogram]
            azi = target.effective_azimuth()
            dip = target.effective_dip()
            if (azi, dip) == (0.0, 0.0):
                data = ndata.copy()
            elif (azi, dip) == (90.0, 0.0):
                data = edata.copy()
            elif (azi, dip) == (0.0, -90):
                data = zdata.copy()
            else:
                data = \
                    ndata * (math.cos(azi*d2r) * math.cos(dip*d2r)) + \
                    edata * (math.sin(azi*d2r) * math.cos(dip*d2r)) + \
                    zdata * math.sin(dip*d2r)
        else:
            data = base_seismogram[0].data.copy()

        factor = source.get_factor() * target.get_factor()
        if factor != 1.0:
            data *= factor

        tr = SeismosizerTrace(
            codes=target.codes,
            data=data,
            deltat=deltat,
            tmin=base_seismogram[0].itmin * deltat + source.time)

        return tr

    def process(self, request=None, status_callback=None, **kwargs):
        '''Process a request.

        The request can be given a a :py:class:`Request` object, or such an
        object is created using ``Request(**kwargs)`` for convenience.
        '''

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
            base_seismogram = self.base_seismogram(sources[0], targets[0])
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

    url = String.T(default='http://kinherd.org/gf/seismosizer')

__all__ = '''
BadRequest
NoSuchStore
Filter
Taper
Source
SourceWithMagnitude
ExplosionSource
DCSource
MTSource
RingfaultSource
Target
Reduction
Request
SeismosizerTrace
Response
Engine
LocalEngine
RemoteEngine
'''.split()
