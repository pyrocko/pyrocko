from pyrocko.snuffling import Snuffling, Param, Choice, Switch
import numpy as num
import matplotlib.pyplot as plt
from matplotlib import cm


def window(freqs, fc, b):
    if fc==0.:
        w = num.zeros(len(freqs))
        w[freqs==0] = 1.
        return w
    T = num.log10(freqs/fc)*b
    w = (num.sin(T)/T)**4
    w[freqs==fc] = 1.
    w[freqs==0.] = 0.
    w/=num.sum(w)
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
        colors = iter(cm.Accent(num.linspace(0., 1., len(all))))
        if self.want_plot=='both':
            alpha = 0.2
        else:
            alpha = 1.
        minf = 0.
        maxf = 0.
        for tr in all:
            tr.ydata = tr.ydata.astype(num.float)
            tr.ydata -= tr.ydata.mean()
            f, a = tr.spectrum()
            minf = min([f.min(), minf]) 
            maxf = max([f.max(), maxf]) 
            absa = num.abs(a)
            labsa = num.log(absa)
            stdabsa = num.std(labsa)
            meanabsa = num.mean(labsa)
            mi, ma = meanabsa - 3*stdabsa, meanabsa + 3*stdabsa
            extrema.append(mi)
            extrema.append(ma)
            c = next(colors)
            if self.want_plot in ['unsmoothed', 'both']:
                p.plot(f,num.abs(a), c=c, alpha=alpha, label='.'.join(tr.nslc_id))
            if self.want_plot in ['smoothed', 'both']:
                smoothed = self.konnoohmachi(num.abs(a), f, self.b)
                p.plot(f, num.abs(smoothed), '-', c=c, label='.'.join(tr.nslc_id))
        
        mi, ma = min(extrema), max(extrema)
        p.set_xscale('log')
        p.set_yscale('log')
        p.set_ylim(num.exp(mi), num.exp(ma))
        p.set_xlim(minf, maxf)
        p.set_xlabel('Frequency [Hz]')
        p.set_ylabel('Counts')
        handles, labels = p.get_legend_handles_labels()
        leg_dict = dict(zip(labels, handles))
        p.legend(leg_dict.values(), leg_dict.keys())
    
    def konnoohmachi(self, amps, freqs, b=20):
        smooth = num.zeros(len(freqs), dtype=freqs.dtype)
        amps = num.array(amps)
        for i, fc in enumerate(freqs):
            fkey = tuple((self.b, fc, freqs[0], freqs[1], freqs[-1]))
            if fkey in self._wins.keys():
                win = self._wins[fkey]
            else:
                win = window(freqs, fc, b)
                self._wins[fkey] = win
            smooth[i] = num.sum(win*amps)

        return smooth

def __snufflings__():
    '''Returns a list of snufflings to be exported by this module.'''
    
    return [ Save() ]

