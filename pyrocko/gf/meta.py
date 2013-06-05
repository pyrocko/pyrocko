
import numpy as num
from guts import *

class StringID(StringPattern):
    pattern = r'^[A-Za-z][A-Za-z0-9.-]*$'

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
            'Kiwi-HDF',
            'Pyrocko',
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


class GFSet(Object):
    id = StringID.T()
    derived_from_id = StringID.T(optional=True)
    version = String.T(default='1.0', optional=True)
    author = Unicode.T(optional=True)
    author_email = String.T(optional=True)
    type = GFType.T(default='Kiwi-HDF', optional=True)
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

class GFSetTypeA(GFSet):
    '''Rotational symmetry, fixed receiver depth'''

    earth_model_id = StringID.T(optional=True)
    receiver_depth = Float.T(default=0.0)
    source_depth_min = Float.T()
    source_depth_max = Float.T()
    source_depth_delta = Float.T()
    distance_min = Float.T()
    distance_max = Float.T()
    distance_delta = Float.T()

class GFSetTypeB(GFSet):
    '''Rotational symmetry, variable receiver depth'''

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

    def __init__(self, **kwargs):
        GFSet.__init__(self, **kwargs)
        self._index_function = None
        self._indices_function = None

    @property
    def deltas(self):
        return num.array([self.receiver_depth_delta, self.source_depth_delta, self.distance_delta])

    @property
    def mins(self):
        return num.array([self.receiver_depth_min, self.source_depth_min, self.distance_min])
        
    @property
    def maxs(self):
        return num.array([self.receiver_depth_max, self.source_depth_max, self.distance_max])

    @property
    def ns(self):
        return num.round((self.maxs - self.mins) / self.deltas).astype(num.int) + 1 

    @property
    def deltat(self):
        return 1.0/self.sample_rate
    
    @property
    def nreceiver_depths(self):
        return int(round((self.receiver_depth_max - self.receiver_depth_min) / self.receiver_depth_delta)) + 1

    @property
    def nsource_depths(self):
        return int(round((self.source_depth_max - self.source_depth_min) / self.source_depth_delta)) + 1

    @property
    def ndistances(self):
        return int(round((self.distance_max - self.distance_min) / self.distance_delta)) + 1

    @property
    def nrecords(self):
        return self.ndistances * self.nreceiver_depths * self.nsource_depths * self.ncomponents

    def irecord(self, receiver_depth, source_depth, distance, icomponent):

        if self._index_function is None:
            self._make_index_functions()

        return self._index_function(receiver_depth, source_depth, distance, icomponent)

    def irecords(self, receiver_depths, source_depths, distances, icomponents):

        if self._indices_function is None:
            self._make_index_functions()

        return self._indices_function(receiver_depths, source_depths, distances, icomponents)

    def iter_nodes(self):
        for ia in xrange(self.nreceiver_depths):
            receiver_depth = self.receiver_depth_min + ia * self.receiver_depth_delta
            for ib in xrange(self.nsource_depths):
                source_depth = self.source_depth_min + ib * self.source_depth_delta
                for ic in xrange(self.ndistances):
                    distance = self.distance_min + ic * self.distance_delta
                    yield receiver_depth, source_depth, distance

    def iter_nodes_components(self):
        for args in self.iter_nodes():
            for icomponent in xrange(self.ncomponents):
                yield args + (icomponent,)

    def _make_index_functions(self):

        amin = self.receiver_depth_min
        bmin = self.source_depth_min
        cmin = self.distance_min

        da = self.receiver_depth_delta
        db = self.source_depth_delta
        dc = self.distance_delta

        na = self.nreceiver_depths
        nb = self.nsource_depths
        nc = self.ndistances
        ng = self.ncomponents

        def index_function(a,b,c, ig):
            ia = int(round((a - amin) / da))
            ib = int(round((b - bmin) / db))
            ic = int(round((c - cmin) / dc))
            
            assert (0 <= ia < na)
            assert (0 <= ib < nb)
            assert (0 <= ic < nc)
            assert (0 <= ig < ng)

            return ia*nb*nc*ng + ib*nc*ng + ic*ng + ig

        def indices_function(a,b,c, ig):
            ia = num.round((a - amin) / da).astype(int)
            ib = num.round((b - bmin) / db).astype(int)
            ic = num.round((c - cmin) / dc).astype(int)
            
            assert num.all(0 <= ia) and num.all(ia < na)
            assert num.all(0 <= ib) and num.all(ib < nb)
            assert num.all(0 <= ic) and num.all(ic < nc)
            assert num.all(0 <= ig) and num.all(ig < ng)

            return ia*nb*nc*ng + ib*nc*ng + ic*ng + ig

        self._index_function = index_function
        self._indices_function = indices_function


class Inventory(Object):
    citations = List.T(Citation.T())
    earth_models = List.T(EarthModel.T())
    modelling_codes = List.T(ModellingCode.T())
    gf_sets = List.T(GFSet.T())



