from pyrocko.snuffling import Snuffling, Param, Choice
import numpy as num

class Save(Snuffling):
    
    '''
    Plot Amplitude Spectrum
    '''

    def setup(self):
        '''Customization of the snuffling.'''
        
        self.set_name('Plot Amplitude Spectrum')
        self.set_live_update(False)
        
    def call(self):
        '''Main work routine of the snuffling.'''
        
        all = [] 
        for traces in self.chopper_selected_traces(fallback=True):
            for trace in traces:
                all.append(trace)

        p = self.pylab() 
        extrema = []
        for tr in all:
            tr.ydata = tr.ydata.astype(num.float)
            tr.ydata -= tr.ydata.mean()
            f, a = tr.spectrum()
            absa = num.abs(a)
            labsa = num.log(absa)
            stdabsa = num.std(labsa)
            meanabsa = num.mean(labsa)
            mi, ma = meanabsa - 3*stdabsa, meanabsa + 3*stdabsa
            extrema.append(mi)
            extrema.append(ma)
            p.plot(f,num.abs(a))

        mi, ma = min(extrema), max(extrema)

        p.set_xscale('log')
        p.set_yscale('log')
        p.set_ylim(num.exp(mi),num.exp(ma))
        p.set_xlabel('Frequency [Hz]')
        p.set_ylabel('Counts')

def __snufflings__():
    '''Returns a list of snufflings to be exported by this module.'''
    
    return [ Save() ]

