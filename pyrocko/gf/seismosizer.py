from guts import Object, Float, String, List, Timestamp, load
from guts_array import Array
from pyrocko import gf, orthodrome, trace
import math, os, urllib

d2r = math.pi / 180.

class NotImplemented(Exception):
    pass

class NoSuchStore(Exception):
    def __init__(self, store_id):
        Exception.__init__(self)
        self.store_id = store_id

    def __str__(self):
        return 'no GF store with id "%s" found.' % self.store_id

class BadRequest(Exception):
    pass

def make_weights(azi, bazi, m6):

    sa = math.sin(azi*d2r)
    ca = math.cos(azi*d2r)
    s2a = math.sin(2.*azi*d2r)
    c2a = math.cos(2.*azi*d2r)
    sb = math.sin(bazi*d2r+math.pi)
    cb = math.cos(bazi*d2r+math.pi)

    f0 = m6[0]*ca**2 + m6[1]*sa**2 + m6[3]*s2a
    f1 = m6[4]*ca + m6[5]*sa
    f2 = m6[2]
    f3 = 0.5*(m6[1]-m6[0])*s2a + m6[3]*c2a
    f4 = m6[5]*ca - m6[4]*sa
    f5 = m6[0]*sa**2 + m6[1]*ca**2 - m6[3]*s2a

    w_n = (cb*f0, cb*f1, cb*f2, cb*f5, -sb*f3, -sb*f4)
    g_n = (0, 1, 2, 8, 3, 4)
    w_e = (sb*f0, sb*f1, sb*f2, sb*f5, cb*f3, cb*f4)
    g_e = (0, 1, 2, 8, 3, 4)
    w_d = (f0, f1, f2, f5)
    g_d = (0, 1, 2, 9)

    return (('N', w_n, g_n), ('E', w_e, g_e), ('Z', w_d, g_d))


class SeismosizerConfig(Object):
    stores_dir = String.T()

    def get_store(self, store_id):
        store_dir = os.path.join(self.stores_dir, store_id)
        if not os.path.isdir(store_dir):
            raise NoSuchStore(store_id)

        store = gf.store.Store(store_dir)
        return store


class SeismosizerRequest(Object):
    '''Get seismogram for a moment tensor point source.'''

    store_id = gf.meta.StringID.T(help='identifier of Green\'s function store')
    source_lat = Float.T()
    source_lon = Float.T()
    source_depth = Float.T(default=0.,
                           help='source depth in [m]')

    source_time = Timestamp.T(default=0.)

    receiver_lat = Float.T()
    receiver_lon = Float.T()
    receiver_depth = Float.T(optional=True,
                             help='receiver depth in [m]')

    mnn = Float.T(default=1., 
                  help='north-north component of moment tensor in [Nm]')
    mee = Float.T(default=1., 
                  help='east-east component of moment tensor in [Nm]')
    mdd = Float.T(default=1., 
                  help='down-down component of moment tensor in [Nm]')
    mne = Float.T(default=0.,
                  help='north-east component of moment tensor in [Nm]')
    mnd = Float.T(default=0.,
                  help='north-down component of moment tensor in [Nm]')
    med = Float.T(default=0.,
                  help='east-down component of moment tensor in [Nm]')

    net_code = String.T(default='',
                  help='network code to set on returned traces')
    sta_code = String.T(default='STA',
                  help='station code to set on returned traces')
    loc_code = String.T(default='SY',
                  help='location code to set on returned traces')

    def sparams(self):
        return urllib.urlencode([(name, val) for (name, val) in self.T.inamevals_to_save(self)])


class SeismosizerTrace(Object):
    net_code = String.T(default='')
    sta_code = String.T(default='STA')
    loc_code = String.T(default='')
    cha_code = String.T(default='Z')
    data = Array.T(shape=(None,))
    deltat = Float.T(default=1.0)
    tmin = Timestamp.T(default=0.0)

    def pyrocko_trace(self):
        return trace.Trace(self.net_code, self.sta_code, 
                self.loc_code, self.cha_code,
                ydata = self.data,
                deltat = self.deltat,
                tmin = self.tmin)


class SeismosizerResponse(Object):
    request = SeismosizerRequest.T()
    traces_list = List.T(SeismosizerTrace.T())

    @property
    def traces(self):
        return [ tr.pyrocko_trace() for tr in self.traces_list ]


def make_seismogram(req, config):
    source = orthodrome.Loc(lat=req.source_lat, lon=req.source_lon)
    receiver = orthodrome.Loc(lat=req.receiver_lat, lon=req.receiver_lon)

    azi = orthodrome.azimuth(source, receiver)
    bazi = orthodrome.azimuth(receiver, source)
    distance = orthodrome.distance_accurate50m(source, receiver)

    m = (req.mnn, req.mee, req.mdd, req.mne, req.mnd, req.med)

    traces = []
    store = config.get_store(req.store_id)
    if not isinstance(store.config, gf.meta.ConfigTypeA):
        raise NotImplemented('unsupported GF store type.')

    if req.receiver_depth is not None:
        if (abs(store.config.receiver_depth - req.receiver_depth) > store.config.source_depth_delta*0.001):
            raise BadRequest('this GF store has a fixed receiver depth of %g m' % store.config.receiver_depth)

    for channel, w, gc in make_weights(azi, bazi, m):

        depths = [ req.source_depth ] * len(gc)
        distances = [ distance ] * len(gc)
        delays = [ 0. ] * len(gc)
        
        gtr = store.sum(args = (depths, distances, gc), 
                   delays = delays,
                   weights = w)

        tr = SeismosizerTrace(
                net_code = req.net_code,
                sta_code = req.sta_code,
                loc_code = req.loc_code,
                cha_code = channel,
                data = gtr.data,
                deltat = gtr.deltat,
                tmin = gtr.deltat * gtr.itmin + req.source_time)

        traces.append(tr)

    store.close()

    return SeismosizerResponse(request=req, traces_list=traces)

class StoresResponse(Object):
    store_configs = List.T(gf.meta.Config.T())

def request_seismogram(req, baseurl='http://localhost:8000/seismosizer/'):
    f = urllib.urlopen('%s?%s' % (baseurl, req.sparams()))
    resp = load(stream=f)
    f.close()
    return resp


