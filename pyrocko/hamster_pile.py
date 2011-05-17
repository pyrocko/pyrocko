import pile, trace, io, util
import os, logging

logger = logging.getLogger('pyrocko.hamster_pile')

class Processor:
    def __init__(self):
        self._buffers = {}

    def process(self, trace):
        return [ trace.copy() ]

    def get_buffer(self, trace):
        
        nslc = trace.nslc_id
        if nslc not in self._buffers:
            return None
        
        trbuf = self._buffers[nslc]
        if (abs((trbuf.tmax+trbuf.deltat) - trace.tmin) < 1.0e-1*trbuf.deltat and 
                        trbuf.ydata.dtype == trace.ydata.dtype and
                        trbuf.deltat == trace.deltat  ):
            return trbuf
        else:
            return None

    def set_buffer(self, trace):
        nslc = trace.nslc_id
        self._buffers[nslc] = trace

class Chain(Processor):
    def __init__(self, *processors):
        self._processors = processors

    def process(self, trace):
        traces = [ trace ]
        xtraces = []
        for p in self._processors:
            for tr in traces:
                xtraces.extend(p.process(traces))

        return xtraces

class DownsampleProcessor(Processor):

    def __init__(self, mapping, deltat):
        Processor.__init__(self)
        self._mapping = mapping
        self._deltat = deltat
        self._tout = {}

    def process(self, trace):
        target_id = self._mapping(trace)
        if target_id is None:
            return []

        previous = self.get_buffer(trace)
        if previous is not None:
            previous.append(trace.ydata)
            trace = previous

        ds_trace = trace.copy()
        ds_trace.downsample_to(self._deltat, snap=True, demean=False)
        ds_trace.set_codes(*target_id)

        if ds_trace.get_ydata().size == 0:
            self.set_buffer(trace)
            return []

        tpad = ((trace.tmax-trace.tmin) - (ds_trace.tmax-ds_trace.tmin)) * 3.
        print tpad 
        self.set_buffer(trace.chop(trace.tmax-tpad, trace.tmax, inplace=False))
        
        #if target_id in self._tout:
         #   tout = self._tout[target_id]
         #   ds_trace.chop(tout, ds_trace.tmax, inplace=True)

        self._tout[target_id] = ds_trace.tmax
        
        return [ ds_trace ]

class HamsterPile(pile.Pile):
    
    def __init__(self, fixation_length=None, path=None, processors=None):
        pile.Pile.__init__(self)
        self._buffers = {}          # keys: nslc,  values: MemTracesFile
        self._fixation_length = fixation_length
        self._path = path
        if processors is None:
            self._processors = [ Processor() ]
        else:
            self._processors = []
            for p in processors:
                self.add_processor(p)

    def set_fixation_length(self, l):
        '''Set length after which the fixation method is called on buffer traces.
        
        The length should be given in seconds. Give None to disable.
        '''
        self.fixate_all()
        self._fixation_length = l   # in seconds
        
    def set_save_path(self, path=
                'dump_%(network)s.%(station)s.%(location)s.%(channel)s_%(tmin)s_%(tmax)s.mseed'):
        self.fixate_all()
        self._path = path
        
    def add_processor(self, processor):
        self.fixate_all()
        self._processors.append(processor)

    def insert_trace(self, trace):
        logger.debug('Received a trace: %s' % trace)
    
        for p in self._processors:
            for tr in p.process(trace):
                self._insert_trace(tr)

    def _insert_trace(self, trace):
        buf = self._append_to_buffer(trace)
        nslc = trace.nslc_id
        
        if buf is None: # create new buffer trace
            if nslc in self._buffers:
                self._fixate(self._buffers[nslc])
                
            trbuf = trace.copy()
            buf = pile.MemTracesFile(None,[trbuf])
            self.add_file(buf)
            self._buffers[nslc] = buf
        
        buf.recursive_grow_update([trace])
        trbuf = buf.get_traces()[0]
        if self._fixation_length is not None:
            if trbuf.tmax - trbuf.tmin > self._fixation_length:
                self._fixate(buf)
                del self._buffers[nslc]

    def _append_to_buffer(self, trace):
        '''Try to append the trace to the active buffer traces.
        
        Returns the current buffer trace or None if unsuccessful.
        '''
        
        nslc = trace.nslc_id
        if nslc not in self._buffers:
            return None
        
        buf = self._buffers[nslc]
        trbuf = buf.get_traces()[0]
        if (abs((trbuf.tmax+trbuf.deltat) - trace.tmin) < 1.0e-1*trbuf.deltat and 
                        trbuf.ydata.dtype == trace.ydata.dtype and
                        trbuf.deltat == trace.deltat  ):

            trbuf.append(trace.ydata)
            return buf
        
        return None

    def fixate_all(self):
        for buf in self._buffers.values():
            self._fixate(buf)
            
        self._buffers = {}

    def _fixate(self, buf):
        if self._path:
            trbuf = buf.get_traces()[0]
            fns = io.save([trbuf], self._path)
            
            self.remove_file(buf)
            self.load_files(fns, show_progress=False)
        
    def drop_older(self, tmax, delete_disk_files=False):
        self.drop(
            condition = lambda file: file.tmax < tmax, 
            delete_disk_files = delete_disk_files)
                
    def drop(self, condition, delete_disk_files=False):
        candidates = []
        buffers = self._buffers.values()
        for file in self.iter_files():
            if condition(file) and file not in buffers:
                candidates.append(file)
        
        self.remove_files(candidates)
        if delete_disk_files:
            for file in candidates:
                if file.abspath and os.path.exists(file.abspath):
                    os.unlink(file.abspath)

                
    def __del__(self):
        self.fixate_all()
        
        
