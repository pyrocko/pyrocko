import trace, io, util, config

import numpy as num
import os, pickle, logging, time

logger = logging.getLogger('pyrocko.pile')

from util import reuse
from trace import degapper


def TracesFileCache(object):

    caches = {}

    def __init__(self, cachedir):
        self.cachedir = cachedir
        self.dircaches = {}
        self.modified = set()
        ensure_dir(self.cachedir)
        
    def get(self, abspath):
        dircache = self.get_dircache_for(abspath)
        if abspath in dircache:
            return dircache[abspath]

    def put(self, abspath, tfile):
        cachepath = self.dircachepath(abspath)
        dircache = self.get_dircache(cachepath)
        dircache[abspath] = tfile
        self.modified.add(cachepath)
    
    def get_dircache(self, cachepath):
        if cachepath not in self.dircaches:
            if os.path.isfile(cachepath):
                self.dircaches[cachepath] = self.load_cache(cachepath)
            else:
                self.dircaches[cachepath] = {}
                
        return self.dircaches[cachepath]

    def get_dircache_for(self, abspath):
        return self.get_dircache(self.dircachepath(abspath))
       
            
    def dircachepath(self, abspath):
        cachefn = "%i" % abs(hash(dirname(abspath)))
        return  pjoin(self.cachedir, cachefn)
        
    def dump_modified(self):
        for cachepath in self.modified:
            self.dump_cache(self.dircaches[cachepath], cachepath)
            
        self.modified = set()
            
    def load_dircache(self, cachefilename):
        
        f = open(cachefilename,'r')
        cache = pickle.load(f)
        f.close()
        
        # weed out files which no longer exist
        for fn in cache.keys():
            if not os.path.isfile(fn):
                del cache[fn]
        return cache
        
    def dump_cache(self, cache, cachefilename):
        f = open(cachefilename+'.tmp','w')
        pickle.dump(cache, f)
        f.close()
        os.rename(cachefilename+'.tmp', cachefilename)

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
    
    def update_from_contents(self, content, flush=True):
        if flush:
            self.empty()
        
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
            
        #self._convert_small_sets_to_tuples()
           
    def overlaps(self, tmin,tmax):
        return not (tmax < self.tmin or self.tmax < tmin)
    
    def is_relevant(self, tmin, tmax, selector=None):
        return  not (tmax <= self.tmin or self.tmax < tmin) and (selector is None or selector(self))

    def _convert_small_sets_to_tuples(self):
        if len(self.networks) < 32:
            self.networks = reuse(tuple(self.networks))
        if len(self.stations) < 32:
            self.stations = reuse(tuple(self.stations))
        if len(self.locations) < 32:
            self.locations = reuse(tuple(self.locations))
        if len(self.channels) < 32:
            self.channels = reuse(tuple(self.channels))
        if len(self.nslc_ids) < 32:
            self.nslc_ids = reuse(tuple(self.nslc_ids))

class TracesFile(TracesGroup):
    def __init__(self, abspath, format, substitutions=None, mtime=None):
        self.abspath = abspath
        self.format = format
        self.traces = []
        self.data_loaded = False
        self.substitutions = substitutions
        self.load_headers(mtime=mtime)
        self.mtime = mtime
        
    def load_headers(self, mtime=None):
        logger.debug('loading headers from file: %s' % self.abspath)
        if mtime is None:
            self.mtime = os.stat(self.abspath)[8]

        for tr in io.load(self.abspath, format=self.format, getdata=False, substitutions=self.substitutions):
            self.traces.append(tr)
            
        self.data_loaded = False
        self.update_from_contents(self.traces)
        
    def load_data(self):
        logger.debug('loading data from file: %s' % self.abspath)
        self.traces = []
        for tr in io.load(self.abspath, format=self.format, getdata=True, substitutions=self.substitutions):
            self.traces.append(tr)

        self.data_loaded = True
        self.update_from_contents(self.traces)
        
    def drop_data(self):
        logger.debug('forgetting data of file: %s' % self.abspath)
        for tr in self.traces:
            tr.drop_data()
        self.data_loaded = False
    
    def reload_if_modified(self):
        mtime = os.stat(self.abspath)[8]
        if mtime != self.mtime:
            logger.debug('reloading file: %s' % self.abspath)
            self.mtime = mtime
            if self.data_loaded:
                self.load_data()
            else:
                self.load_headers()
    
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
        self.update_from_contents((file,), flush=False)
        
    def remove_file(self, file):
        self.files.remove(file)
        self.update_from_contents(self.files)
    
    def chop(self, tmin, tmax, group_selector=None, trace_selector=None):
        chopped = []
        for file in self.files:
            if file.is_relevant(tmin, tmax, group_selector):
                file.load_data()
                chopped.extend( file.chop(tmin, tmax, trace_selector) )
        return chopped

            
class Pile(TracesGroup):
    def __init__(self, ):
        self.subpiles = {}
        self.update_from_contents(self.subpiles)
        self.open_files = set()
    
    def add_files(self, files=None, filenames=None, filename_attributes=None, fileformat='mseed', cache=None):
        modified_subpiles = set()
        if filenames is not None:
            for file in loader(filenames, fileformat, cache, filename_attributes):
                subpile = self.dispatch(file)
                subpile.add_file(file)
                modified_subpiles.add(subpile)
                
        self.update_from_contents(modified_subpiles, flush=False)
        
    def add_file(self, file):
        subpile = self.dispatch(file)
        subpile.add_file(file)
        self.update_from_contents((file,), flush=False)
    
    def remove_file(self, file):
        subpile = self.dispatch(file)
        subpile.remove_file(file)
        self.update_from_contents(self.subpiles)
        
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
        for subpile in self.subpiles.values():
            if subpile.is_relevant(tmin,tmax, group_selector):
                chopped.extend( subpile.chop(tmin, tmax, group_selector, trace_selector) )
        return chopped

    def _process_chopped(chopped, degap, want_incomplete, wmax, wmin, tpad):
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
        
        if not self.is_relevant(tmin,tmax,selector): return
        
        files_match_full = [ f for f in self.msfiles if f.is_relevant( tmin-tpad, tmax+tpad, selector ) ]
        
        if not files_match_full: return
        
        ftmin = num.inf
        ftmax = -num.inf
        for f in files_match_full:
            ftmin = min(ftmin,f.tmin)
            ftmax = max(ftmax,f.tmax)
        
        iwin = max(0, int(((ftmin-tpad)-tmin)/tinc-2))
        files_match_partial = files_match_full
        
        partial_every = 50
        
        while True:
            chopped = []
            wmin, wmax = tmin+iwin*tinc, tmin+(iwin+1)*tinc
            if wmin >= ftmax or wmin >= tmax: break
                        
            if iwin%partial_every == 0:  # optimization
                swmin, swmax = tmin+iwin*tinc, tmin+(iwin+partial_every)*tinc
                files_match_partial = [ f for f in files_match_full if f.is_relevant( swmin-tpad, swmax+tpad, selector ) ]
                
            files_match_win = [ f for f in files_match_partial if f.is_relevant( wmin-tpad, wmax+tpad, selector ) ]
            
            if files_match_win:
                used_files = set()
                for file in files_match_win:
                    used_files.add(file)
                    if not file.data_loaded:
                        self.open_files.add(file)
                        file.load_data()
                    chopped.extend( file.chop(wmin-tpad, wmax+tpad, selector) )
                
                    self._process_chopped(chopped, degap, want_incomplete, wmax, wmin, tpad)
                    
                yield chopped
                
                unused_files = self.open_files - used_files
                for file in unused_files:
                    file.drop_data()
                    self.open_files.remove(file)
            
            iwin += 1
        
        if not keep_current_files_open:
            while self.open_files:
                file = self.open_files.pop()
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
        for file in self.msfiles:
            keys |= file.gather_keys(gather)
            
        return sorted(keys)
    
    def get_deltats(self):
        deltats = set()
        for file in self.msfiles:
            deltats.update(file.get_deltats())
        return sorted(list(deltats))
    
    def iter_traces(self, load_data=False, return_abspath=False):
        for file in self.msfiles:
            
            must_close = False
            if load_data and not file.data_loaded:
                file.load_data()
                must_close = True
            
            for trace in file.iter_traces():
                if return_abspath:
                    yield file.abspath, trace
                else:
                    yield trace
            
            if must_close:
                file.drop_data()
    
    def reload_modified(self):
        for file in self.msfiles:
            file.reload_if_modified()
            
        self.update_from_contents(self.msfiles)
            
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


