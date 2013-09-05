
import math, re
import numpy as num
from guts import *
from guts_array import literal
from pyrocko import cake


class CakeNDModel(Object):
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
            'undefined',
        ]


class WaveformType(StringChoice):
    choices = [
            'full waveform',
            'bodywave', 
            'P wave', 
            'S wave', 
            'surface wave',
            'undefined',
        ]
    

class NearfieldTermsType(StringChoice):
    choices = [
            'complete',
            'incomplete',
            'missing',
            'undefined',
        ]


class GFType(StringChoice):
    choices = [
            'Pyrocko',
            'Kiwi-HDF',
        ]


class Citation(Object):
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

        citations = []

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
            
            c = Citation(id=id_, type=entry.type, **d)
            citations.append(c)

        return citations

_fpat = r'[+-](\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?'
_spat = StringID.pattern[1:-1]
pat = r'^((first|last)?\((' + _spat + r'(\|' + _spat + r')*)\)|(' +  \
                 _spat + r'))?(' + _fpat + ')?$'
timing_regex = re.compile( pat )

class PhaseSelect(StringChoice):
    choices = [ '', 'first', 'last' ]

class InvalidTimingSpecification(ValidationError):
    pass

class Timing(SObject):

    def __init__(self, s=None, **kwargs):
        
        if s is not None:
            m = timing_regex.match(s)
            if m:
                if m.group(3):
                    phase_ids = m.group(3).split('|')
                elif m.group(5):
                    phase_ids = [ m.group(5) ]
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
        phases = [ get_phase(phase_id) for phase_id in self.phase_ids ]
        times = [ phase(args) for phase in phases ]
        times = [ t+self.offset for t in times if t is not None ]
        if not times:
            return None
        elif self.select == 'first':
            return min(times)
        elif self.select == 'last':
            return max(times)
        else:
            return times[0]

    phase_ids = List.T(String.T())
    offset = Float.T(default=0.0)
    select = PhaseSelect.T(default='')

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

class PhaseTabDef(Object):
    id = StringID.T()
    definition = String.T()

    @property
    def phases(self):
        return [ x for x in mkdefs(self.definition) if isinstance(x, cake.PhaseDef) ]

    @property
    def horizontal_velocities(self):
        return [ x for x in mkdefs(self.definition) if isinstance(x, float) ]

class EarthModel(Object):
    id = StringID.T()
    region = String.T(optional=True)
    description = String.T(optional=True)
    citation_ids = List.T(StringID.T())


class ModellingCode(Object):
    id = StringID.T()
    name = String.T(optional=True)
    version = String.T(optional=True)
    method = String.T(optional=True)
    author = Unicode.T(optional=True)
    author_email = String.T(optional=True)
    citation_ids = List.T(StringID.T())

class OutOfBounds(Exception):
    def __init__(self, values=None):
        Exception.__init__(self)
        self.values = values

    def __str__(self):
        if self.values:
            return 'out of bounds: (%s)' % ','.join('%g' % x for x in self.values)
        else:
            return 'out of bounds'

class GFSet(Object):
    id = StringID.T()
    derived_from_id = StringID.T(optional=True)
    version = String.T(default='1.0', optional=True)
    author = Unicode.T(optional=True)
    author_email = String.T(optional=True)
    type = GFType.T(default='Pyrocko', optional=True)
    modelling_code_id = StringID.T(optional=True)
    scope_type = ScopeType.T(default='undefined')
    waveform_type = WaveformType.T(default='undefined')
    nearfield_terms = NearfieldTermsType.T(default='undefined')
    can_interpolate_source = Bool.T(default=False)
    can_interpolate_receiver = Bool.T(default=False)
    frequency_min = Float.T(optional=True)
    frequency_max = Float.T(optional=True)
    sample_rate = Float.T(optional=True)
    size = Int.T(optional=True)
    citation_ids = List.T(StringID.T())
    description = String.T(default='', optional=True)
    ncomponents = Int.T(default=1)
    earthmodel_cake = CakeNDModel.T(optional=True)
    phase_tab_defs = List.T(PhaseTabDef.T())

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
            t = self.T.get_property(name)
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

            arr =  num.linspace(*start_stop_num(*(sssn + (mi,ma,inc)))) 
            ntotal *= len(arr)

            arrs.append(arr)
            i += 1

        arrs.append(self.coords[-1])
        return nditer_outer(arrs)

class GFSetTypeA(GFSet):
    '''Rotational symmetry, fixed receiver depth
    
    Index variables are (source_depth, distance, component).'''

    earth_model_id = StringID.T(optional=True)
    receiver_depth = Float.T(default=0.0)
    source_depth_min = Float.T()
    source_depth_max = Float.T()
    source_depth_delta = Float.T()
    distance_min = Float.T()
    distance_max = Float.T()
    distance_delta = Float.T()

    def get_distance(self, args):
        return args[1]

    def get_source_depth(self, args):
        return args[0]

    def get_receiver_depth(self, args):
        return self.receiver_depth

    def _update(self):
        self.mins = num.array([self.source_depth_min, self.distance_min])
        self.maxs =  num.array([self.source_depth_max, self.distance_max])
        self.deltas = num.array([self.source_depth_delta, self.distance_delta])
        self.ns = num.round((self.maxs - self.mins) / self.deltas).astype(num.int) + 1 
        self.deltat =  1.0/self.sample_rate
        self.nrecords = num.product(self.ns) * self.ncomponents
        self.coords = tuple( num.linspace(mi,ma,n) for 
                (mi,ma,n) in zip(self.mins, self.maxs, self.ns) ) + \
                    ( num.arange(self.ncomponents), )
        self.nsource_depths, self.ndistances = self.ns

    def _make_index_functions(self):

        amin, bmin = self.mins
        da, db = self.deltas
        na,nb = self.ns

        ng = self.ncomponents

        def index_function(a,b, ig):
            ia = int(round((a - amin) / da))
            ib = int(round((b - bmin) / db))
            try:
                return num.ravel_multi_index((ia,ib,ig), (na,nb,ng))
            except ValueError:
                raise OutOfBounds()

        def indices_function(a,b, ig):
            ia = num.round((a - amin) / da).astype(int)
            ib = num.round((b - bmin) / db).astype(int)
            try:
                return num.ravel_multi_index((ia,ib,ig), (na,nb,ng))
            except ValueError:
                for ia_, ib_, ig_ in zip(ia,ib,ig):
                    try:
                        num.ravel_multi_index((ia_,ib_,ig_), (na,nb,ng))
                    except ValueError:
                        raise OutOfBounds()

        def vicinity_function(a,b, ig):
            ias = indi12((a - amin) / da, na)
            ibs = indi12((b - bmin) / db, nb)

            if not (0 <= ig < ng):
                raise OutOfBounds()

            indis = []
            for ia, va in ias:
                iia = ia*nb*ng
                for ib, vb in ibs:
                    indis.append( ( iia + ib*ng + ig, va*vb ) )
            
            return indis

        self._index_function = index_function
        self._indices_function = indices_function
        self._vicinity_function = vicinity_function

class GFSetTypeB(GFSet):
    '''Rotational symmetry

    Index variables are (receiver_depth, source_depth, distance, component).'''

    earth_model_id = StringID.T(optional=True)
    receiver_depth_min = Float.T()
    receiver_depth_max = Float.T()
    receiver_depth_delta = Float.T()
    source_depth_min = Float.T()
    source_depth_max = Float.T()
    source_depth_delta = Float.T()
    distance_min = Float.T()
    distance_max = Float.T()
    distance_delta = Float.T()

    def get_distance(self, args):
        return args[2]

    def get_source_depth(self, args):
        return args[1]

    def get_receiver_depth(self, args):
        return args[0]

    def _update(self):
        self.mins = num.array([self.receiver_depth_min, self.source_depth_min, self.distance_min])
        self.maxs =  num.array([self.receiver_depth_max, self.source_depth_max, self.distance_max])
        self.deltas = num.array([self.receiver_depth_delta, self.source_depth_delta, self.distance_delta])
        self.ns = num.round((self.maxs - self.mins) / self.deltas).astype(num.int) + 1 
        self.deltat =  1.0/self.sample_rate
        self.nrecords = num.product(self.ns) * self.ncomponents
        self.coords = tuple( num.linspace(mi,ma,n) for 
                (mi,ma,n) in zip(self.mins, self.maxs, self.ns) ) + \
                    ( num.arange(self.ncomponents), )
        self.nreceiver_depths, self.nsource_depths, self.ndistances = self.ns

    def _make_index_functions(self):

        amin, bmin, cmin = self.mins
        da, db, dc = self.deltas
        na,nb,nc = self.ns
        ng = self.ncomponents

        def index_function(a,b,c, ig):
            ia = int(round((a - amin) / da))
            ib = int(round((b - bmin) / db))
            ic = int(round((c - cmin) / dc))
            try:
                return num.ravel_multi_index((ia,ib,ic,ig), (na,nb,nc,ng))
            except ValueError:
                raise OutOfBounds()

        def indices_function(a,b,c, ig):
            ia = num.round((a - amin) / da).astype(int)
            ib = num.round((b - bmin) / db).astype(int)
            ic = num.round((c - cmin) / dc).astype(int)
            try:
                return num.ravel_multi_index((ia,ib,ic,ig), (na,nb,nc,ng))
            except ValueError:
                raise OutOfBounds()

        def vicinity_function(a,b,c, ig):
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
                        indis.append( ( iia + iib + ic*ng + ig, va*vb*vc ) )
            
            return indis

        self._index_function = index_function
        self._indices_function = indices_function
        self._vicinity_function = vicinity_function



class Inventory(Object):
    citations = List.T(Citation.T())
    earth_models = List.T(EarthModel.T())
    modelling_codes = List.T(ModellingCode.T())
    gf_sets = List.T(GFSet.T())

vicinity_eps = 1e-5

def indi12(x, n):
    r = round(x)
    if abs(r - x) < vicinity_eps:
        i = int(r)
        if not (0 <= i < n):
            raise OutOfBounds()

        return ( (int(r), 1.), )
    else:
        f = math.floor(x)
        i = int(f)
        if not (0 <= i < n-1):
            raise OutOfBounds()

        v = x-f
        return ( (i, 1.-v), (i + 1, v) )

def float_or_none(s):
    units = {
            'k' : 1e3,
            'M' : 1e6,
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
            v = [ float_or_none(x) for x in s.split(':') ]
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
        start = [mi,ma][swap]
    if stop is None:
        stop = [ma,mi][swap]
    if step is None:
        step = [inc,-inc][ma<mi]
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
    return num.nditer(x, 
            op_axes=(num.identity(len(x), dtype=num.int)-1).tolist())

__all__ = 'GFSet GFSetTypeA GFSetTypeB'.split()




