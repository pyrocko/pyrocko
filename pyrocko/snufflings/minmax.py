from pyrocko.snuffling import Snuffling, Param
from pyrocko import trace

class MinMaxSnuffling(Snuffling):
    
    def setup(self):
        self.set_name('Minimum Maximum Peak-To-Peak')
        self.tinc = None
    
    def call(self):
                
        # to select a reasonable increment for the chopping, the smallest 
        # sampling interval in the pile is looked at. this is only done, 
        # the first time the snuffling is called.
        if self.tinc is None:
            self.tinc = self.get_pile().get_deltats()[0] * 10000.
        
        # the chopper yields lists of traces but for minmax() below, an iterator
        # yielding single traces is needed. using a converter:
        def iter_single_traces():
            for traces in self.chopper_selected_traces(tinc=self.tinc, degap=False):
                for tr in traces:
                    yield tr
        
        # the function minmax() in the trace module can get minima and maxima 
        # grouped by (network,station,location,channel):
        mima = trace.minmax(iter_single_traces())
        
        for nslc in sorted(mima.keys()):
            p2p = mima[nslc][1] - mima[nslc][0]
            print '%s.%s.%s.%s: %12.5g %12.5g %12.5g' % (nslc + mima[nslc] + (p2p,))
                                            
def __snufflings__():
    return [ MinMaxSnuffling() ]

