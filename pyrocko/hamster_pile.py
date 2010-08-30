import pile, trace, io
import os, logging

logger = logging.getLogger('pyrocko.hamster_pile')

class HamsterPile(pile.Pile):
    
    def __init__(self, fixation_length=None, path=None):
        pile.Pile.__init__(self)
        self._buffers = {}          # keys: nslc,  values: MemTracesFile
        self._fixation_length = fixation_length
        self._path = path
        
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
        
    def insert_trace(self, trace):
        logger.debug('Received a trace: %s' % trace)
    
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
            

    def _fixate(self, buf):
        if self._path:
            trbuf = buf.get_traces()[0]
            fns = io.save([trbuf], self._path)
            
            self.remove_file(buf)
            self.add_files(fns, show_progress=False)
        
    def __del__(self):
        self.fixate_all()
        
        