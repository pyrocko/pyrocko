

class HamsterPile(pile.Pile):        
    
    def __init__(self):
        pile.Pile.__init__(self)
        self._traces = {}
        self._traces_to_files = {}

    def got_trace(self, trace):

        nslc = trace.nslc_id
        trnew = False
        if nslc in self._traces:
            trbuf = self._traces[nslc]

            if (abs((trbuf.tmax+trbuf.deltat) - trace.tmin) < 1.0e-1*trbuf.deltat and 
                        trbuf.ydata.dtype == trace.ydata.dtype and
                        trbuf.deltat == trace.deltat  ):
                
                trbuf.append(trace.ydata)
                trace = trbuf
            else:
                trnew = True
        else:
            trnew = True
        
        if trnew:
            if nslc in self._traces:
                self.save(self._traces[nslc])
                
            memfile = pile.MemTracesFile(None,[trace])
            self.add_file(memfile)
            self._traces_to_files[trace] = memfile
            
            self._traces[nslc] = trace
        
        self._traces_to_files[trace].recursive_grow_update([trace])

        for nslc, tr2 in self._traces.items():
            if tr2.tmax - tr2.tmin > 30.:
                self.save(tr2)
                del self._traces[nslc]


    def save(self, trace):
        io.save([trace], pjoin(self.path, '%(network)s.%(station)s.%(location)s.%(channel)s.%(tmin)s.%(tmax)s.mseed'))
