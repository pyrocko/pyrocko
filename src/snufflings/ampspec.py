from pyrocko.snuffling import Snuffling, Param, Choice, Switch
import numpy as num
import matplotlib.pyplot as plt

def window(freqs, fc, b):
    
    if fc==0.:
        w = num.zeros(len(freqs))
        w[freqs==0] = 1.
        return w
    T = num.log10(freqs/fc)*b
    w = (num.sin(T)/T)**4
    w[freqs==fc] = 1.
    w[freqs==0.] = 0.
    w/=num.nansum(w)
    return w


class Save(Snuffling):
    '''
    Plot Amplitude Spectrum
    '''
    def setup(self):
        '''Customization of the snuffling.'''
        
        self.set_name('Plot Amplitude Spectrum local')
        self.add_parameter(Choice('Plot', 'want_plot', 'smoothed', ['smoothed', 'unsmoothed',
                                                        'both']))
        self.add_parameter(Param('Smoothing band width', 'b', 40., 1., 100.))
        self.set_live_update(False)
        self._wins = {}

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
            if self.want_plot in ['unsmoothed', 'both']:
                p.plot(f,num.abs(a))
            if self.want_plot in ['smoothed', 'both']:
                smoothed = self.konnoohmachi(f, num.abs(a), self.b)
                p.plot(f, num.abs(smoothed), '-')

        mi, ma = min(extrema), max(extrema)

        p.set_xscale('log')
        p.set_yscale('log')
        p.set_ylim(num.exp(mi),num.exp(ma))
        p.set_xlabel('Frequency [Hz]')
        p.set_ylabel('Counts')
    
    def konnoohmachi(self, freqs, amps, b=20):
        smooth = num.zeros(len(freqs), dtype=freqs.dtype)
        amps = num.array(amps)
        for i, fc in enumerate(freqs):
            fkey = tuple((fc, freqs[0], freqs[1], freqs[-1]))
            if fkey in self._wins.keys():
                win = self._wins[fkey]
            else:
                win = window(freqs, fc, b)
                self._wins[fkey] = win
            smooth[i] = num.nansum(win*amps)

        return smooth

def __snufflings__():
    '''Returns a list of snufflings to be exported by this module.'''
    
    return [ Save() ]

