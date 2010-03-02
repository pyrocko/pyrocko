import trace, io, util, config

import numpy as num
import os, pickle, logging, time
pjoin = os.path.join
logger = logging.getLogger('pyrocko.pile')

from util import reuse
from trace import degapper


class TracesFileCache(object):

    caches = {}

    def __init__(self, cachedir):
        self.cachedir = cachedir
        self.dircaches = {}
        self.modified = set()
        util.ensuredir(self.cachedir)
        
    def get(self, abspath):
        '''Try to get an item from the cache.'''
        
        dircache = self._get_dircache_for(abspath)
        if abspath in dircache:
            return dircache[abspath]
        return None

    def put(self, abspath, tfile):
        '''Put an item into the cache.'''
        
        cachepath = self._dircachepath(abspath)
        # get lock on cachepath here
        dircache = self._get_dircache(cachepath)
        dircache[abspath] = tfile
        self.modified.add(cachepath)

    def dump_modified(self):
        '''Save any modifications to disk.'''

        for cachepath in self.modified:
            self._dump_dircache(self.dircaches[cachepath], cachepath)
            # unlock 
            
        self.modified = set()

    def clean(self):
        '''Weed out missing files from the disk caches.'''
        
        self.dump_modified()
        
        for fn in os.listdir(self.cachedir):
            try:
                i = int(fn) # valid filenames are integers
                cache = self._load_dircache(pjoin(self.cachedir, fn))
                self._dump_dircache(cache, pjoin(self.cachedir, fn))
                
            except ValueError:
                pass

    def _get_dircache_for(self, abspath):
        return self._get_dircache(self._dircachepath(abspath))
    
    def _get_dircache(self, cachepath):
        if cachepath not in self.dircaches:
            if os.path.isfile(cachepath):
                self.dircaches[cachepath] = self._load_dircache(cachepath)
            else:
                self.dircaches[cachepath] = {}
                
        return self.dircaches[cachepath]
       
    def _dircachepath(self, abspath):
        cachefn = "%i" % abs(hash(os.path.dirname(abspath)))
        return  pjoin(self.cachedir, cachefn)
            
    def _load_dircache(self, cachefilename):
        
        f = open(cachefilename,'r')
        cache = pickle.load(f)
        f.close()
        
        # weed out files which no longer exist
        for fn in cache.keys():
            if not os.path.isfile(fn):
                del cache[fn]
        return cache
        
    def _dump_dircache(self, cache, cachefilename):
        if not cache:
            if os.path.exists(cachefilename):
                os.remove(cachefilename)
            return            
        tmpfn = cachefilename+'.%i.tmp' % os.getpid()
        f = open(tmpfn, 'w')
        pickle.dump(cache, f)
        f.close()
        os.rename(tmpfn, cachefilename)


def get_cache(cachedir):
    if cachedir not in TracesFileCache.caches:
        TracesFileCache.caches[cachedir] = TracesFileCache(cachedir)
        
    return TracesFileCache.caches[cachedir]
    
def loader(filenames, fileformat, cache, filename_attributes):
            
    if config.show_progress:
        widgets = ['Scanning files', ' ',
                progressbar.Bar(marker='-',left='[',right=']'), ' ',
                progressbar.Percentage(), ' ',]
        
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(filenames)).start()
    
    regex = None
    if filename_attributes:
        regex = re.compile(filename_attributes)
    
    failures = []
    for ifile, filename in enumerate(filenames):
        try:
            abspath = os.path.abspath(filename)
            
            substitutions = None
            if regex:
                m = regex.search(filename)
                if not m: raise FilenameAttributeError(
                    "Cannot get attributes with pattern '%s' from path '%s'" 
                        % (filename_attributes, filename))
                substitutions = m.groupdict()
                
            
            mtime = os.stat(filename)[8]
            tfile = None
            if cache:
                tfile = cache.get(abspath)
            
            if not tfile or tfile.mtime != mtime or substitutions:
                tfile = TracesFile(abspath, fileformat, substitutions=substitutions, mtime=mtime)
                if cache and not substitutions:
                    cache.put(abspath, tfile)
                
        except (io.FileLoadError, OSError, FilenameAttributeError), xerror:
            failures.append(abspath)
            logging.warn(xerror)
        else:
            yield tfile
        
        if config.show_progress: pbar.update(ifile+1)
    
    if config.show_progress: pbar.finish()
    if failures:
        logging.warn('The following file%s caused problems and will be ignored:\n' % plural_s(len(failures)) + '\n'.join(failures))
    
    if cache:
        cache.dump_modified()

class TracesGroup(object):
    def __init__(self):
        self.empty()
    
    def empty(self):
        self.networks, self.stations, self.locations, self.channels, self.nslc_ids = [ set() for x in range(5) ]
        self.tmin, self.tmax = num.inf, -num.inf
        self.have_tuples = False
    
    def update(self, content, empty=True):
        if empty:
            self.empty()
        else:
            if self.have_tuples:
                self._convert_tuples_to_sets()
            
        for c in content:
        
            if isinstance(c, TracesGroup):
                self.networks.update( c.networks )
                self.stations.update( c.stations )
                self.locations.update( c.locations )
                self.channels.update( c.channels )
                self.nslc_ids.update( c.nslc_ids )
                
            elif isinstance(c, trace.Trace):
                self.networks.add(c.network)
                self.stations.add(c.station)
                self.locations.add(c.location)
                self.channels.add(c.channel)
                self.nslc_ids.add(c.nslc_id)
                
            self.tmin = min(self.tmin, c.tmin)
            self.tmax = max(self.tmax, c.tmax)
        
        if empty:    
            self._convert_small_sets_to_tuples()
        
           
    def overlaps(self, tmin,tmax):
        return not (tmax < self.tmin or self.tmax < tmin)
    
    def is_relevant(self, tmin, tmax, selector=None):
        return  not (tmax <= self.tmin or self.tmax < tmin) and (selector is None or selector(self))

    def _convert_tuples_to_sets(self):
        if not isinstance(self.networks, set):
            self.networks = set(self.networks)
        if not isinstance(self.stations, set):
            self.stations = set(self.stations)
        if not isinstance(self.locations, set):
            self.locations = set(self.locations)
        if not isinstance(self.channels, set):
            self.channels = set(self.channels)
        if not isinstance(self.nslc_ids, set):
            self.nslc_ids = set(self.nslc_ids)
        self.have_tuples = False

    def _convert_small_sets_to_tuples(self):
        if len(self.networks) < 32:
            self.networks = reuse(tuple(self.networks))
            self.have_tuples = True
        if len(self.stations) < 32:
            self.stations = reuse(tuple(self.stations))
            self.have_tuples = True
        if len(self.locations) < 32:
            self.locations = reuse(tuple(self.locations))
            self.have_tuples = True
        if len(self.channels) < 32:
            self.channels = reuse(tuple(self.channels))
            self.have_tuples = True
        if len(self.nslc_ids) < 32:
            self.nslc_ids = reuse(tuple(self.nslc_ids))
            self.have_tuples = True
            
class TracesFile(TracesGroup):
    def __init__(self, abspath, format, substitutions=None, mtime=None):
        self.abspath = abspath
        self.format = format
        self.traces = []
        self.data_loaded = 0
        self.substitutions = substitutions
        self.load_headers(mtime=mtime)
        self.update(self.traces)
        self.mtime = mtime
        
    def load_headers(self, mtime=None):
        logger.debug('loading headers from file: %s' % self.abspath)
        if mtime is None:
            self.mtime = os.stat(self.abspath)[8]

        for tr in io.load(self.abspath, format=self.format, getdata=False, substitutions=self.substitutions):
            self.traces.append(tr)
            
        self.data_loaded = 0
        
    def load_data(self):
        if self.data_loaded == 0:
            logger.debug('loading data from file: %s' % self.abspath)
            self.traces = []
            for tr in io.load(self.abspath, format=self.format, getdata=True, substitutions=self.substitutions):
                self.traces.append(tr)
            
        self.data_loaded += 1

    def drop_data(self):
        if self.data_loaded:
            if self.data_loaded == 1:
                logger.debug('forgetting data of file: %s' % self.abspath)
                for tr in self.traces:
                    tr.drop_data()
                    
            self.data_loaded -= 1
    
    def reload_if_modified(self):
        mtime = os.stat(self.abspath)[8]
        if mtime != self.mtime:
            logger.debug('reloading file: %s' % self.abspath)
            self.mtime = mtime
            if self.data_loaded:
                self.load_data()
                self.data_loaded -= 1
            else:
                self.load_headers()
            
            self.update(self.traces)
            
            return True
            
        return False
       
    def chop(self,tmin,tmax,selector):
        chopped = []
        for tr in self.traces:
            if not selector or selector(tr):
                try:
                    chopped.append(tr.chop(tmin,tmax,inplace=False))
                except trace.NoData:
                    pass
            
        return chopped
        
    def get_deltats(self):
        deltats = set()
        for trace in self.traces:
            deltats.add(trace.deltat)
            
        return deltats
    
    def iter_traces(self):
        for trace in self.traces:
            yield trace
    
    def gather_keys(self, gather):
        keys = set()
        for trace in self.traces:
            keys.add(gather(trace))
            
        return keys
    
    def __str__(self):
        
        def sl(s):
            return sorted(list(s))
        
        s = 'TracesFile\n'
        s += 'abspath: %s\n' % self.abspath
        s += 'file mtime: %s\n' % util.gmctime(self.mtime)
        s += 'number of traces: %i\n' % len(self.traces)
        s += 'timerange: %s - %s\n' % (util.gmctime(self.tmin), util.gmctime(self.tmax))
        s += 'networks: %s\n' % ', '.join(sl(self.networks))
        s += 'stations: %s\n' % ', '.join(sl(self.stations))
        s += 'locations: %s\n' % ', '.join(sl(self.locations))
        s += 'channels: %s\n' % ', '.join(sl(self.channels))
        return s


    
class FilenameAttributeError(Exception):
    pass

class SubPile(TracesGroup):
    def __init__(self):
        self.files = []
        self.empty()
    
    def add_file(self, file):
        self.files.append(file)
        self.update((file,), empty=False)
        
    def remove_file(self, file):
        self.files.remove(file)
        self.update(self.files)
    
    def chop(self, tmin, tmax, group_selector=None, trace_selector=None):
        used_files = set()
        chopped = []
        for file in self.files:
            if file.is_relevant(tmin, tmax, group_selector):
                file.load_data()
                used_files.add(file)
                chopped.extend( file.chop(tmin, tmax, trace_selector) )
                
        return chopped, used_files
        
    def gather_keys(self, gather):
        keys = set()
        for file in self.files:
            keys |= file.gather_keys(gather)
            
        return keys

    def get_deltats(self):
        deltats = set()
        for file in self.files:
            deltats.add(file.deltat)
            
        return deltats

    def iter_traces(self, load_data=False, return_abspath=False):
        for file in self.files:
            
            must_drop = False
            if load_data:
                file.load_data()
                must_drop = True
            
            for trace in file.iter_traces():
                if return_abspath:
                    yield file.abspath, trace
                else:
                    yield trace
            
            if must_drop:
                file.drop_data()

    def reload_modified(self):
        modified = False
        for file in self.files:
            modified |= file.reload_if_modified()
        
        if modified:
            self.update(self.files)
            
        return modified
        
    def __str__(self):
    
        def sl(s):
            return sorted([ x for x in s ])

        s = 'SubPile\n'
        s += 'number of files: %i\n' % len(self.files)
        s += 'timerange: %s - %s\n' % (util.gmctime(self.tmin), util.gmctime(self.tmax))
        s += 'networks: %s\n' % ', '.join(sl(self.networks))
        s += 'stations: %s\n' % ', '.join(sl(self.stations))
        s += 'locations: %s\n' % ', '.join(sl(self.locations))
        s += 'channels: %s\n' % ', '.join(sl(self.channels))
        return s

             
class Pile(TracesGroup):
    def __init__(self, ):
        self.subpiles = {}
        self.update(self.subpiles)
        self.open_files = set()
    
    def add_files(self, files=None, filenames=None, filename_attributes=None, fileformat='mseed', cache=None):
        modified_subpiles = set()
        if filenames is not None:
            for file in loader(filenames, fileformat, cache, filename_attributes):
                subpile = self.dispatch(file)
                subpile.add_file(file)
                modified_subpiles.add(subpile)
                
        self.update(modified_subpiles, empty=False)
        
    def add_file(self, file):
        subpile = self.dispatch(file)
        subpile.add_file(file)
        self.update((file,), empty=False)
    
    def remove_file(self, file):
        subpile = self.dispatch(file)
        subpile.remove_file(file)
        self.update(self.subpiles)
        
    def dispatch_key(self, file):
        tt = time.gmtime(file.tmin)
        return (tt[0],tt[1])
    
    def dispatch(self, file):
        k = self.dispatch_key(file)
        if k not in self.subpiles:
            self.subpiles[k] = SubPile()
            
        return self.subpiles[k]
        
    def chop(self, tmin, tmax, group_selector=None, trace_selector=None):
        chopped = []
        used_files = set()
        for subpile in self.subpiles.values():
            if subpile.is_relevant(tmin,tmax, group_selector):
                _chopped, _used_files =  subpile.chop(tmin, tmax, group_selector, trace_selector)
                chopped.extend(_chopped)
                used_files.update(_used_files)
                
        return chopped, used_files

    def _process_chopped(self, chopped, degap, want_incomplete, wmax, wmin, tpad):
        chopped.sort(lambda a,b: cmp(a.full_id, b.full_id))
        if degap:
            chopped = degapper(chopped)
            
        if not want_incomplete:
            wlen = (wmax+tpad)-(wmin-tpad)
            chopped_weeded = []
            for trace in chopped:
                if abs(wlen - round(wlen/trace.deltat)*trace.deltat) > 0.001:
                    logging.warn('Selected window length (%g) not nicely divideable by sampling interval (%g).' % (wlen, trace.deltat) )
                if len(trace.ydata) == t2ind((wmax+tpad)-(wmin-tpad), trace.deltat):
                    chopped_weeded.append(trace)
            chopped = chopped_weeded
        return chopped
            
    def chopper(self, tmin=None, tmax=None, tinc=None, tpad=0., selector=None, 
                      want_incomplete=True, degap=True, keep_current_files_open=False):
        
        if tmin is None:
            tmin = self.tmin+tpad
                
        if tmax is None:
            tmax = self.tmax-tpad
            
        if tinc is None:
            tinc = tmax-tmin
        
        if not self.is_relevant(tmin-tpad,tmax+tpad,selector): return
        
        iwin = 0
        open_files = set()       
        while True:
            chopped = []
            wmin, wmax = tmin+iwin*tinc, tmin+(iwin+1)*tinc
            if wmin >= tmax: break
            chopped, used_files = self.chop(wmin-tpad, wmax+tpad, selector) 
            processed = self._process_chopped(chopped, degap, want_incomplete, wmax, wmin, tpad)
            yield processed
            unused_files = self.open_files - used_files
            while unused_files:
                file = unused_files.pop()
                file.drop_data()
                open_files.remove(file)
                
            iwin += 1
        
        while open_files:
            file = open_files.pop()
            file.drop_data()
            
        
    def all(self, *args, **kwargs):
        alltraces = []
        for traces in self.chopper( *args, **kwargs ):
            alltraces.extend( traces )
            
        return alltraces
        
    def iter_all(self, *args, **kwargs):
        for traces in self.chopper( *args, **kwargs):
            for trace in traces:
                yield trace
    
    def chopper_grouped(self, gather, *args, **kwargs):
        keys = self.gather_keys(gather)
        outer_selector = None
        if 'selector' in kwargs:
            outer_selector = kwargs['selector']
        if outer_selector is None:
            outer_selector = lambda xx: True
            
        gather_cache = {}
        
        for key in keys:
            def sel(obj):
                if isinstance(obj, trace.Trace):
                    return gather(obj) == key and outer_selector(obj)
                else:
                    if obj not in gather_cache:
                        gather_cache[obj] = obj.gather_keys(gather)
                        
                    return key in gather_cache[obj] and outer_selector(obj)
                
            kwargs['selector'] = sel
            
            for traces in self.chopper(*args, **kwargs):
                yield traces
        
    def gather_keys(self, gather):
        keys = set()
        for subpile in self.subpiles.values():
            keys |= subpile.gather_keys(gather)
            
        return sorted(keys)
    
    def get_deltats(self):
        deltats = set()
        for subpile in self.subpiles.values():
            deltats.update(subpile.get_deltats())
            
        return sorted(list(deltats))
    
    def iter_traces(self, load_data=False, return_abspath=False):
        for subpile in self.subpiles.values():
            for xx in subpile.iter_traces(load_data, return_abspath):
                yield xx
   
    def reload_modified(self):
        modified = False
        for subpile in self.subpiles.values():
            modified |= subpile.reload_modified()
        
        if modified:
            self.update(self.subpiles)
            
        return modified
            
    def __str__(self):
        
        def sl(s):
            return sorted([ x for x in s ])
        
        s = 'Pile\n'
        s += 'number of subpiles: %i\n' % len(self.subpiles)
        s += 'timerange: %s - %s\n' % (util.gmctime(self.tmin), util.gmctime(self.tmax))
        s += 'networks: %s\n' % ', '.join(sl(self.networks))
        s += 'stations: %s\n' % ', '.join(sl(self.stations))
        s += 'locations: %s\n' % ', '.join(sl(self.locations))
        s += 'channels: %s\n' % ', '.join(sl(self.channels))
        return s


