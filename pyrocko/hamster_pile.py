import pile, trace

class HamsterPile(pile.Pile):
    
    def __init__(self):
        pile.Pile.__init__(self)
        self._buffers = {}          # keys: nslc,  values: MemTracesFile
        
    def insert_trace(self, trace):

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
        if trbuf.tmax - trbuf.tmin > 30.:
            self._fixate(buf)

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

    def _fixate(self, buf):
        pass
        #io.save([trace], pjoin(self.path, '%(network)s.%(station)s.%(location)s.%(channel)s.%(tmin)s.%(tmax)s.mseed'))
