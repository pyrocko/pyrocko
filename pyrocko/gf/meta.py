
import math
import re
import fnmatch
import numpy as num
from guts import Object, SObject, String, StringChoice, StringPattern, \
    Unicode, Float, Bool, Int, TBase, List, ValidationError
from guts import dump, load  # noqa

from guts_array import literal, Array
from pyrocko import cake, orthodrome, spit

d2r = math.pi / 180.
r2d = 1.0 / d2r
km = 1000.


class Earthmodel1D(Object):
    dummy_for = cake.LayeredModel

    class __T(TBase):
        def regularize_extra(self, val):
            if isinstance(val, basestring):
                val = cake.LayeredModel.from_scanlines(
                    cake.read_nd_model_str(val))

            return val

        def to_save(self, val):
            return literal(cake.write_nd_model_str(val))


class StringID(StringPattern):
    pattern = r'^[A-Za-z][A-Za-z0-9._]{0,64}$'


class ScopeType(StringChoice):
    choices = [
        'global',
        'regional',
        'local',
    ]


class WaveformType(StringChoice):
    choices = [
        'full waveform',
        'bodywave',
        'P wave',
        'S wave',
        'surface wave',
    ]


class NearfieldTermsType(StringChoice):
    choices = [
        'complete',
        'incomplete',
        'missing',
    ]


class Reference(Object):
    id = StringID.T()
    type = String.T()
    title = Unicode.T()
    journal = Unicode.T(optional=True)
    volume = Unicode.T(optional=True)
    number = Unicode.T(optional=True)
    pages = Unicode.T(optional=True)
    year = String.T()
    note = Unicode.T(optional=True)
    issn = String.T(optional=True)
    doi = String.T(optional=True)
    url = String.T(optional=True)
    eprint = String.T(optional=True)
    authors = List.T(Unicode.T())
    publisher = Unicode.T(optional=True)
    keywords = Unicode.T(optional=True)
    note = Unicode.T(optional=True)
    abstract = Unicode.T(optional=True)

    @classmethod
    def from_bibtex(cls, filename=None, stream=None):

        from pybtex.database.input import bibtex

        parser = bibtex.Parser()

        if filename is not None:
            bib_data = parser.parse_file(filename)
        elif stream is not None:
            bib_data = parser.parse_stream(stream)

        references = []

        for id_, entry in bib_data.entries.iteritems():
            d = {}
            avail = entry.fields.keys()
            for prop in cls.T.properties:
                if prop.name == 'authors' or prop.name not in avail:
                    continue

                d[prop.name] = entry.fields[prop.name]

            if 'author' in entry.persons:
                d['authors'] = []
                for person in entry.persons['author']:
                    d['authors'].append(unicode(person))

            c = Reference(id=id_, type=entry.type, **d)
            references.append(c)

        return references


_fpat = r'[+-](\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?'
_spat = StringID.pattern[1:-1]

timing_regex = re.compile(
    r'^((first|last)?\((' + _spat + r'(\|' + _spat + r')*)\)|(' +
    _spat + r'))?(' + _fpat + ')?$')


class PhaseSelect(StringChoice):
    choices = ['', 'first', 'last']


class InvalidTimingSpecification(ValidationError):
    pass


class Timing(SObject):
    '''
    Definition of a time instant relative to one or more named phase arrivals

    Instances of this class can be used e.g. in cutting and tapering
    operations. They can hold an absolute time or an offset to a named phase
    arrival or group of such arrivals.

    Timings can be instantiated from a simple string defintion i.e. with
    ``Timing(str)`` where ``str`` is something like
    ``'SELECT(PHASE_IDS)[+-]OFFSET'`` where ``'SELECT'`` is ``'first'``,
    ``'last'`` or empty, ``'PHASE_IDS'`` is a ``'|'``-separated list of phase
    names, and ``'OFFSET'`` is the time offset in seconds.

    **Examples:**

    * ``'100'`` : absolute time; 100 s.
    * ``'P-100'`` : 100 s before arrival of P phase.
    * ``'(A|B)'`` : time instant of phase arrival A, or B if A is undefined for
      a given geometry.
    * ``'first(A|B)'`` : as above, but the temporally first arrival of A and B
      is chosen, if both phases are defined for a given geometry.
    * ``'last(A|B)'`` : as above but the temporally last arrival is chosen.
    * ``'first(A|B|C)-100'`` : 100 s before first out of arrivals A, B, and C.
    '''

    def __init__(self, s=None, **kwargs):

        if s is not None:
            m = timing_regex.match(s)
            if m:
                if m.group(3):
                    phase_ids = m.group(3).split('|')
                elif m.group(5):
                    phase_ids = [m.group(5)]
                else:
                    phase_ids = []

                select = m.group(2) or ''

                offset = 0.0
                if m.group(6):
                    offset = float(m.group(6))

                kwargs = dict(
                    phase_ids=phase_ids,
                    select=select,
                    offset=offset)

            else:
                raise InvalidTimingSpecification(s)

        SObject.__init__(self, **kwargs)

    def __str__(self):
        l = []
        if self.phase_ids:
            sphases = '|'.join(self.phase_ids)
            if len(self.phase_ids) > 1 or self.select:
                sphases = '(%s)' % sphases

            if self.select:
                sphases = self.select + sphases

            l.append(sphases)

        if self.offset != 0.0 or not self.phase_ids:
            l.append('%+g' % self.offset)

        return ''.join(l)

    def evaluate(self, get_phase, args):
        try:
            phases = [get_phase(phase_id) for phase_id in self.phase_ids]
            times = [phase(args) for phase in phases]
            times = [t+self.offset for t in times if t is not None]
            if not times:
                return None
            elif self.select == 'first':
                return min(times)
            elif self.select == 'last':
                return max(times)
            else:
                return times[0]
        except spit.OutOfBounds:
            raise OutOfBounds(args)

    phase_ids = List.T(StringID.T())
    offset = Float.T(default=0.0)
    select = PhaseSelect.T(
        default='',
        help=('Can be either ``\'%s\'``, ``\'%s\'``, or ``\'%s\'``. ' %
              tuple(PhaseSelect.choices)))


def mkdefs(s):
    defs = []
    for x in s.split(','):
        try:
            defs.append(float(x))
        except ValueError:
            if x.startswith('!'):
                defs.extend(cake.PhaseDef.classic(x[1:]))
            else:
                defs.append(cake.PhaseDef(x))

    return defs


class TPDef(Object):
    '''Maps an arrival phase identifier to an arrival phase definition'''
    id = StringID.T(
        help='name used to identify the phase')
    definition = String.T(
        help='definition of the phase in either cake syntax as defined in '
             ':py:class:`pyrocko.cake.PhaseDef`, or, if prepended with an '
             '``!``, as a *classic phase name*, or, if it is a simple '
             'number, as a constant horizontal velocity.')

    @property
    def phases(self):
        return [x for x in mkdefs(self.definition)
                if isinstance(x, cake.PhaseDef)]

    @property
    def horizontal_velocities(self):
        return [x for x in mkdefs(self.definition) if isinstance(x, float)]


class OutOfBounds(Exception):
    def __init__(self, values=None):
        Exception.__init__(self)
        self.values = values

    def __str__(self):
        if self.values:
            return 'out of bounds: (%s)' % ','.join('%g' % x
                                                    for x in self.values)
        else:
            return 'out of bounds'


class Location(Object):
    '''
    Geographical location

    The location is given by a reference point at the earth's surface
    (:py:attr:`lat`, :py:attr:`lon`) and a cartesian offset from this point
    (:py:attr:`north_shift`, :py:attr:`east_shift`, :py:attr:`depth`). The
    offset corrected lat/lon coordinates of the location can be accessed though
    the :py:attr:`effective_latlon`, :py:attr:`effective_lat`, and
    :py:attr:`effective_lon` properties.
    '''

    lat = Float.T(
        default=0.0,
        optional=True,
        help='latitude of reference point [deg]')

    lon = Float.T(
        default=0.0,
        optional=True,
        help='longitude of reference point [deg]')

    north_shift = Float.T(
        default=0.,
        optional=True,
        help='northward cartesian offset from reference point [m]')

    east_shift = Float.T(
        default=0.,
        optional=True,
        help='eastward cartesian offset from reference point [m]')

    depth = Float.T(
        default=0.0,
        help='depth [m]')

    def __init__(self, **kwargs):
        Object.__init__(self, **kwargs)
        self._latlon = None

    def __setattr__(self, name, value):
        if name in ('lat', 'lon', 'north_shift', 'east_shift'):
            self.__dict__['_latlon'] = None

        Object.__setattr__(self, name, value)

    @property
    def effective_latlon(self):
        '''
        Property holding the offset-corrected lat/lon pair of the location.
        '''

        if self._latlon is None:
            if self.north_shift == 0.0 and self.east_shift == 0.0:
                self._latlon = self.lat, self.lon
            else:
                self._latlon = orthodrome.ne_to_latlon(
                    self.lat, self.lon, self.north_shift, self.east_shift)

        return self._latlon

    @property
    def effective_lat(self):
        '''
        Property holding the offset-corrected latitude of the location.
        '''

        return self.effective_latlon[0]

    @property
    def effective_lon(self):
        '''
        Property holding the offset-corrected longitude of the location.
        '''

        return self.effective_latlon[1]

    def same_origin(self, other):
        '''
        Check whether other location object has the same reference location.
        '''

        return self.lat == other.lat and self.lon == other.lon

    def distance_to(self, other):
        '''
        Compute distance [m] to other location object.
        '''

        if self.same_origin(other):
            return math.sqrt((self.north_shift - other.north_shift)**2 +
                            (self.east_shift - other.east_shift)**2)

        else:
            slat, slon = self.effective_latlon
            rlat, rlon = other.effective_latlon
            return orthodrome.distance_accurate50m_numpy(slat, slon,
                                                         rlat, rlon)

    def azibazi_to(self, other):
        '''
        Compute azimuth and backazimuth to and from other location object.
        '''

        if self.same_origin(other):
            azi = r2d * math.atan2(other.east_shift - self.east_shift,
                                   other.north_shift - self.north_shift)
            bazi = azi + 180.
        else:
            slat, slon = self.effective_latlon
            rlat, rlon = other.effective_latlon
            azi = orthodrome.azimuth_numpy(slat, slon, rlat, rlon)
            bazi = orthodrome.azimuth_numpy(rlat, rlon, slat, slon)

        return azi, bazi


class Receiver(Location):
    pass


def g(x, d):
    if x is None:
        return d
    else:
        return x


class DiscretizedSource(Object):
    times = Array.T(shape=(None,), dtype=num.float)
    lats = Array.T(shape=(None,), dtype=num.float, optional=True)
    lons = Array.T(shape=(None,), dtype=num.float, optional=True)
    lat = Float.T(optional=True)
    lon = Float.T(optional=True)
    north_shifts = Array.T(shape=(None,), dtype=num.float, optional=True)
    east_shifts = Array.T(shape=(None,), dtype=num.float, optional=True)
    depths = Array.T(shape=(None,), dtype=num.float)

    def __init__(self, **kwargs):
        Object.__init__(self, **kwargs)
        self._latlons = None

    def __setattr__(self, name, value):
        if name in ('lat', 'lon', 'north_shifts', 'east_shifts',
                    'lats', 'lons'):
            self.__dict__['_latlons'] = None

        Object.__setattr__(self, name, value)

    @property
    def effective_latlons(self):
        if self._latlons is None:
            if self.lats is not None and self.lons is not None:
                if (self.north_shifts is not None and
                        self.east_shifts is not None):
                    self._latlons = orthodrome.ne_to_latlon(
                        self.lats, self.lons,
                        self.north_shifts, self.east_shifts)
                else:
                    self._latlons = self.lats, self.lons
            else:
                lat = g(self.lat, 0.0)
                lon = g(self.lon, 0.0)
                self._latlons = orthodrome.ne_to_latlon(
                    lat, lon, self.north_shifts, self.east_shifts)

        return self._latlons

    def same_origin(self, receiver):
        return (g(self.lat, 0.0) == receiver.lat and
                g(self.lon, 0.0) == receiver.lon and
                self.lats is None and self.lons is None)

    def azibazis_to(self, receiver):
        if self.same_origin(receiver):
            azis = r2d * num.arctan2(receiver.east_shift - self.east_shifts,
                                     receiver.north_shift - self.north_shifts)
            bazis = azis + 180.
        else:
            slats, slons = self.effective_latlons
            rlat, rlon = receiver.effective_latlon
            azis = orthodrome.azimuth_numpy(slats, slons, rlat, rlon)
            bazis = orthodrome.azimuth_numpy(rlat, rlon, slats, slons)

        return azis, bazis

    def distances_to(self, receiver):
        if self.same_origin(receiver):
            return num.sqrt((self.north_shifts - receiver.north_shift)**2 +
                            (self.east_shifts - receiver.east_shift)**2)

        else:
            slats, slons = self.effective_latlons
            rlat, rlon = receiver.effective_latlon
            return orthodrome.distance_accurate50m_numpy(slats, slons,
                                                         rlat, rlon)

    def element_coords(self, i):
        if self.lats is not None and self.lons is not None:
            lat = float(self.lats[i])
            lon = float(self.lons[i])
        else:
            lat = self.lat
            lon = self.lon

        if self.north_shifts is not None and self.east_shifts is not None:
            north_shift = float(self.north_shifts[i])
            east_shift = float(self.east_shifts[i])

        else:
            north_shift = east_shift = None

        return lat, lon, north_shift, east_shift

    @property
    def nelements(self):
        return self.times.size


class DiscretizedExplosionSource(DiscretizedSource):
    m0s = Array.T(shape=(None,), dtype=num.float)

    def make_weights(self, receiver, scheme):
        azis, bazis = self.azibazis_to(receiver)

        sb = num.sin(bazis*d2r-num.pi)
        cb = num.cos(bazis*d2r-num.pi)

        m0s = self.m0s
        n = azis.size

        if scheme == 'elastic2':
            w_n = (cb*m0s,)
            g_n = num.repeat((0,), n)
            w_e = (sb*m0s,)
            g_e = num.repeat((0,), n)
            w_u = (-m0s,)
            g_u = num.repeat((1,), n)

        elif scheme == 'elastic10':
            w_n = num.concatenate((cb*m0s, cb*m0s, cb*m0s))
            g_n = num.repeat((0, 2, 8), n)
            w_e = num.concatenate((sb*m0s, sb*m0s, sb*m0s))
            g_e = num.repeat((0, 2, 8), n)
            w_u = num.concatenate((-m0s, -m0s, -m0s))
            g_u = num.repeat((5, 7, 9), n)

        else:
            assert False

        return (('N', w_n, g_n), ('E', w_e, g_e), ('Z', w_u, g_u))

    def split(self):
        from pyrocko.gf.seismosizer import ExplosionSource
        sources = []
        for i in xrange(self.nelements):
            lat, lon, north_shift, east_shift = self.element_coords(i)
            sources.append(ExplosionSource(
                time=float(self.times[i]),
                lat=lat,
                lon=lon,
                north_shift=north_shift,
                east_shift=east_shift,
                depth=float(self.depths[i]),
                moment=float(self.m0s[i])))

        return sources


class DiscretizedMTSource(DiscretizedSource):
    m6s = Array.T(shape=(None, 6), dtype=num.float)

    def make_weights(self, receiver, scheme):

        if scheme != 'elastic10':
            assert False

        azis, bazis = self.azibazis_to(receiver)

        sa = num.sin(azis*d2r)
        ca = num.cos(azis*d2r)
        s2a = num.sin(2.*azis*d2r)
        c2a = num.cos(2.*azis*d2r)
        sb = num.sin(bazis*d2r-num.pi)
        cb = num.cos(bazis*d2r-num.pi)

        m6s = self.m6s

        f0 = m6s[:, 0]*ca**2 + m6s[:, 1]*sa**2 + m6s[:, 3]*s2a
        f1 = m6s[:, 4]*ca + m6s[:, 5]*sa
        f2 = m6s[:, 2]
        f3 = 0.5*(m6s[:, 1]-m6s[:, 0])*s2a + m6s[:, 3]*c2a
        f4 = m6s[:, 5]*ca - m6s[:, 4]*sa
        f5 = m6s[:, 0]*sa**2 + m6s[:, 1]*ca**2 - m6s[:, 3]*s2a

        n = azis.size

        w_n = num.concatenate((cb*f0, cb*f1, cb*f2, cb*f5, -sb*f3, -sb*f4))
        g_n = num.repeat((0, 1, 2, 8, 3, 4), n)
        w_e = num.concatenate((sb*f0, sb*f1, sb*f2, sb*f5, cb*f3, cb*f4))
        g_e = num.repeat((0, 1, 2, 8, 3, 4), n)
        w_u = num.concatenate((-f0, -f1, -f2, -f5))
        g_u = num.repeat((5, 6, 7, 9), n)

        return (('N', w_n, g_n), ('E', w_e, g_e), ('Z', w_u, g_u))

    def split(self):
        from pyrocko.gf.seismosizer import MTSource
        sources = []
        for i in xrange(self.nelements):
            lat, lon, north_shift, east_shift = self.element_coords(i)
            sources.append(MTSource(
                time=float(self.times[i]),
                lat=lat,
                lon=lon,
                north_shift=north_shift,
                east_shift=east_shift,
                depth=float(self.depths[i]),
                m6=self.m6s[i]))

        return sources


class ComponentSchemes(StringChoice):
    choices = ('elastic10', 'elastic2')


class Config(Object):
    '''Greens function store meta information.'''

    id = StringID.T()

    derived_from_id = StringID.T(optional=True)
    version = String.T(default='1.0', optional=True)
    modelling_code_id = StringID.T(optional=True)
    author = Unicode.T(optional=True)
    author_email = String.T(optional=True)
    scope_type = ScopeType.T(optional=True)
    waveform_type = WaveformType.T(optional=True)
    nearfield_terms = NearfieldTermsType.T(optional=True)
    description = String.T(default='', optional=True)
    reference_ids = List.T(StringID.T())
    size = Int.T(optional=True)

    earthmodel_1d = Earthmodel1D.T(optional=True)

    can_interpolate_source = Bool.T(optional=True)
    can_interpolate_receiver = Bool.T(optional=True)
    frequency_min = Float.T(optional=True)
    frequency_max = Float.T(optional=True)
    sample_rate = Float.T(optional=True)
    ncomponents = Int.T(default=1)
    component_scheme = ComponentSchemes.T(default='elastic10')
    tabulated_phases = List.T(TPDef.T())

    def __init__(self, **kwargs):
        self._do_auto_updates = False
        Object.__init__(self, **kwargs)
        self._index_function = None
        self._indices_function = None
        self._vicinity_function = None
        self._do_auto_updates = True
        self.update()

    def __setattr__(self, name, value):
        Object.__setattr__(self, name, value)
        try:
            self.T.get_property(name)
            if self._do_auto_updates:
                self.update()

        except ValueError:
            pass

    def update(self):
        self._update()
        self._make_index_functions()

    def irecord(self, *args):
        return self._index_function(*args)

    def irecords(self, *args):
        return self._indices_function(*args)

    def vicinity(self, *args):
        return self._vicinity_function(*args)

    def iter_nodes(self, level=None):
        return nditer_outer(self.coords[:level])

    def iter_extraction(self, gdef):
        i = 0
        arrs = []
        ntotal = 1
        for mi, ma, inc in zip(self.mins, self.maxs, self.deltas):
            if gdef and len(gdef) > i:
                sssn = gdef[i]
            else:
                sssn = (None,)*4

            arr = num.linspace(*start_stop_num(*(sssn + (mi, ma, inc))))
            ntotal *= len(arr)

            arrs.append(arr)
            i += 1

        arrs.append(self.coords[-1])
        return nditer_outer(arrs)

    def make_sum_params(self, source, receiver):
        out = []
        delays = source.times
        for comp, weights, icomponents in source.make_weights(
                receiver,
                self.component_scheme):

            args = self.make_indexing_args(source, receiver, icomponents)
            delays_expanded = num.tile(delays, icomponents.size/delays.size)
            out.append((comp, args, delays_expanded, weights))

        return out

    def short_info(self):
        raise NotImplemented('should be implemented in subclass')

class ConfigTypeA(Config):
    '''Cylindrical symmetry, fixed receiver depth

Index variables are (source_depth, distance, component).'''

    receiver_depth = Float.T(default=0.0)
    source_depth_min = Float.T()
    source_depth_max = Float.T()
    source_depth_delta = Float.T()
    distance_min = Float.T()
    distance_max = Float.T()
    distance_delta = Float.T()

    short_type = 'A'

    def get_distance(self, args):
        return args[1]

    def get_source_depth(self, args):
        return args[0]

    def get_receiver_depth(self, args):
        return self.receiver_depth

    def _update(self):
        self.mins = num.array([self.source_depth_min, self.distance_min])
        self.maxs = num.array([self.source_depth_max, self.distance_max])
        self.deltas = num.array([self.source_depth_delta, self.distance_delta])
        self.ns = num.round((self.maxs - self.mins) /
                            self.deltas).astype(num.int) + 1
        self.deltat = 1.0/self.sample_rate
        self.nrecords = num.product(self.ns) * self.ncomponents
        self.coords = tuple(num.linspace(mi, ma, n)
                            for (mi, ma, n)
                            in zip(self.mins, self.maxs, self.ns)) + \
            (num.arange(self.ncomponents),)

        self.nsource_depths, self.ndistances = self.ns

    def _make_index_functions(self):

        amin, bmin = self.mins
        da, db = self.deltas
        na, nb = self.ns

        ng = self.ncomponents

        def index_function(a, b, ig):
            ia = int(round((a - amin) / da))
            ib = int(round((b - bmin) / db))
            try:
                return num.ravel_multi_index((ia, ib, ig), (na, nb, ng))
            except ValueError:
                raise OutOfBounds()

        def indices_function(a, b, ig):
            ia = num.round((a - amin) / da).astype(int)
            ib = num.round((b - bmin) / db).astype(int)
            try:
                return num.ravel_multi_index((ia, ib, ig), (na, nb, ng))
            except ValueError:
                for ia_, ib_, ig_ in zip(ia, ib, ig):
                    try:
                        num.ravel_multi_index((ia_, ib_, ig_), (na, nb, ng))
                    except ValueError:
                        raise OutOfBounds()

        def vicinity_function(a, b, ig):
            ias = indi12((a - amin) / da, na)
            ibs = indi12((b - bmin) / db, nb)

            if not (0 <= ig < ng):
                raise OutOfBounds()

            indis = []
            for ia, va in ias:
                iia = ia*nb*ng
                for ib, vb in ibs:
                    indis.append((iia + ib*ng + ig, va*vb))

            return indis

        self._index_function = index_function
        self._indices_function = indices_function
        self._vicinity_function = vicinity_function

    def make_indexing_args(self, source, receiver, icomponents):
        nc = icomponents.size
        dists = source.distances_to(receiver)
        n = dists.size
        return (num.tile(source.depths, nc/n),
                num.tile(dists, nc/n),
                icomponents)

    @property
    def short_extent(self):
        return '%g:%g:%g x %g:%g:%g' % (
            self.source_depth_min/km,
            self.source_depth_max/km,
            self.source_depth_delta/km,
            self.distance_min/km,
            self.distance_max/km,
            self.distance_delta/km)


class ConfigTypeB(Config):
    '''Cylindrical symmetry

Index variables are (receiver_depth, source_depth, distance, component).'''


    receiver_depth_min = Float.T()
    receiver_depth_max = Float.T()
    receiver_depth_delta = Float.T()
    source_depth_min = Float.T()
    source_depth_max = Float.T()
    source_depth_delta = Float.T()
    distance_min = Float.T()
    distance_max = Float.T()
    distance_delta = Float.T()

    short_type = 'B'

    def get_distance(self, args):
        return args[2]

    def get_source_depth(self, args):
        return args[1]

    def get_receiver_depth(self, args):
        return args[0]

    def _update(self):
        self.mins = num.array([
            self.receiver_depth_min,
            self.source_depth_min,
            self.distance_min])

        self.maxs = num.array([
            self.receiver_depth_max,
            self.source_depth_max,
            self.distance_max])

        self.deltas = num.array([
            self.receiver_depth_delta,
            self.source_depth_delta,
            self.distance_delta])

        self.ns = num.round((self.maxs - self.mins) /
                            self.deltas).astype(num.int) + 1
        self.deltat = 1.0/self.sample_rate
        self.nrecords = num.product(self.ns) * self.ncomponents
        self.coords = tuple(num.linspace(mi, ma, n) for
                            (mi, ma, n) in
                            zip(self.mins, self.maxs, self.ns)) + \
            (num.arange(self.ncomponents),)
        self.nreceiver_depths, self.nsource_depths, self.ndistances = self.ns

    def _make_index_functions(self):

        amin, bmin, cmin = self.mins
        da, db, dc = self.deltas
        na, nb, nc = self.ns
        ng = self.ncomponents

        def index_function(a, b, c, ig):
            ia = int(round((a - amin) / da))
            ib = int(round((b - bmin) / db))
            ic = int(round((c - cmin) / dc))
            try:
                return num.ravel_multi_index((ia, ib, ic, ig),
                                             (na, nb, nc, ng))
            except ValueError:
                raise OutOfBounds()

        def indices_function(a, b, c, ig):
            ia = num.round((a - amin) / da).astype(int)
            ib = num.round((b - bmin) / db).astype(int)
            ic = num.round((c - cmin) / dc).astype(int)
            try:
                return num.ravel_multi_index((ia, ib, ic, ig),
                                             (na, nb, nc, ng))
            except ValueError:
                raise OutOfBounds()

        def vicinity_function(a, b, c, ig):
            ias = indi12((a - amin) / da, na)
            ibs = indi12((b - bmin) / db, nb)
            ics = indi12((c - cmin) / dc, nc)

            if not (0 <= ig < ng):
                raise OutOfBounds()

            indis = []
            for ia, va in ias:
                iia = ia*nb*nc*ng
                for ib, vb in ibs:
                    iib = ib*nc*ng
                    for ic, vc in ics:
                        indis.append((iia + iib + ic*ng + ig, va*vb*vc))

            return indis

        self._index_function = index_function
        self._indices_function = indices_function
        self._vicinity_function = vicinity_function

    def make_indexing_args(self, source, receiver, icomponents):
        nc = icomponents.size
        dists = source.distances_to(receiver)
        n = dists.size
        receiver_depths = num.empty(nc)
        receiver_depths.fill(receiver.depth)
        return (receiver_depths,
                num.tile(source.depths, nc/n),
                num.tile(dists, nc/n),
                icomponents)

    @property
    def short_extent(self):
        return '%g:%g:%g x %g:%g:%g x %g:%g:%g' % (
            self.receiver_depth_min/km,
            self.receiver_depth_max/km,
            self.receiver_depth_delta/km,
            self.source_depth_min/km,
            self.source_depth_max/km,
            self.source_depth_delta/km,
            self.distance_min/km,
            self.distance_max/km,
            self.distance_delta/km)


class Weighting(Object):
    factor = Float.T(default=1.0)


class Taper(Object):
    tmin = Timing.T()
    tmax = Timing.T()
    tfade = Float.T(default=0.0)
    shape = StringChoice.T(
        choices=['cos', 'linear'],
        default='cos',
        optional=True)


class SimplePattern(SObject):

    _pool = {}

    def __init__(self, pattern):
        self._pattern = pattern
        SObject.__init__(self)

    def __str__(self):
        return self._pattern

    @property
    def regex(self):
        pool = SimplePattern._pool
        if self.pattern not in pool:
            rpat = '|'.join(fnmatch.translate(x) for
                            x in self.pattern.split('|'))
            pool[self.pattern] = re.compile(rpat, re.I)

        return pool[self.pattern]

    def match(self, s):
        return self.regex.match(s)


class WaveformType(StringChoice):
    choices = ['dis', 'vel', 'acc',
               'amp_spec_dis', 'amp_spec_vel', 'amp_spec_acc',
               'envelope_dis', 'envelope_vel', 'envelope_acc']


class ChannelSelection(Object):
    pattern = SimplePattern.T(optional=True)
    min_sample_rate = Float.T(optional=True)
    max_sample_rate = Float.T(optional=True)


class StationSelection(Object):
    includes = SimplePattern.T()
    excludes = SimplePattern.T()
    distance_min = Float.T(optional=True)
    distance_max = Float.T(optional=True)
    azimuth_min = Float.T(optional=True)
    azimuth_max = Float.T(optional=True)


class WaveformSelection(Object):
    channel_selection = ChannelSelection.T(optional=True)
    station_selection = StationSelection.T(optional=True)
    taper = Taper.T()
    #filter = FrequencyResponse.T()
    waveform_type = WaveformType.T(default='dis')
    weighting = Weighting.T(optional=True)
    sample_rate = Float.T(optional=True)
    gf_store_id = StringID.T(optional=True)


vicinity_eps = 1e-5


def indi12(x, n):
    r = round(x)
    if abs(r - x) < vicinity_eps:
        i = int(r)
        if not (0 <= i < n):
            raise OutOfBounds()

        return ((int(r), 1.),)
    else:
        f = math.floor(x)
        i = int(f)
        if not (0 <= i < n-1):
            raise OutOfBounds()

        v = x-f
        return ((i, 1.-v), (i + 1, v))


def float_or_none(s):
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

    if s:
        return float(s) * factor
    else:
        return None


class GridSpecError(Exception):
    def __init__(self, s):
        Exception.__init__(self, 'invalid grid specification: %s' % s)


def parse_grid_spec(spec):
    try:
        result = []
        for dspec in spec.split(','):
            t = dspec.split('@')
            num = start = stop = step = None
            if len(t) == 2:
                num = int(t[1])
                if num <= 0:
                    raise GridSpecError(spec)

            elif len(t) > 2:
                raise GridSpecError(spec)

            s = t[0]
            v = [float_or_none(x) for x in s.split(':')]
            if len(v) == 1:
                start = stop = v[0]
            if len(v) >= 2:
                start, stop = v[0:2]
            if len(v) == 3:
                step = v[2]

            if len(v) > 3 or (len(v) > 2 and num is not None):
                raise GridSpecError(spec)

            if step == 0.0:
                raise GridSpecError(spec)

            result.append((start, stop, step, num))

    except ValueError:
        raise GridSpecError(spec)

    return result


def start_stop_num(start, stop, step, num, mi, ma, inc, eps=1e-5):
    swap = step is not None and step < 0.
    if start is None:
        start = [mi, ma][swap]
    if stop is None:
        stop = [ma, mi][swap]
    if step is None:
        step = [inc, -inc][ma < mi]
    if num is None:
        if (step < 0) != (stop-start < 0):
            raise GridSpecError()

        num = int(round((stop-start)/step))+1
        stop2 = start + (num-1)*step
        if abs(stop-stop2) > eps:
            num = int(math.floor((stop-start)/step))+1
            stop = start + (num-1)*step
        else:
            stop = stop2

    if start == stop:
        num = 1

    return start, stop, num


def nditer_outer(x):
    return num.nditer(
        x, op_axes=(num.identity(len(x), dtype=num.int)-1).tolist())

__all__ = '''
Earthmodel1D
StringID
ScopeType
WaveformType
NearfieldTermsType
Reference
PhaseSelect
InvalidTimingSpecification
Timing
TPDef
OutOfBounds
Location
Receiver
DiscretizedSource
DiscretizedExplosionSource
DiscretizedMTSource
ComponentSchemes
Config
ConfigTypeA
ConfigTypeB
GridSpecError
Weighting
Taper
SimplePattern
WaveformType
ChannelSelection
StationSelection
WaveformSelection
dump
load
'''.split()
