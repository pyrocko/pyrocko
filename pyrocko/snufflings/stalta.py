from pyrocko import pile, trace, util, io
import sys, os, math, time
import numpy as num
from pyrocko.snuffling import Param, Snuffling, Switch, Choice
from pyrocko.gui_util import Marker

h = 3600.
m = 60.

scalingmethods = ('[0-1]', '[-1/ratio,1]', '[-1/ratio,1] clipped to [0,1]')
scalingmethod_map = dict([ (m,i+1) for (i,m) in enumerate(scalingmethods) ] )

class DetectorSTALTA(Snuffling):

    def setup(self):    
        '''Customization of the snuffling.'''
        
        self.set_name('STA LTA')
        self.add_parameter(Param('Highpass [Hz]', 'highpass', None, 0.001, 100., low_is_none=True))
        self.add_parameter(Param('Lowpass [Hz]', 'lowpass', None, 0.001, 100., high_is_none=True))
        self.add_parameter(Param('Short window [s]', 'swin', 30., 1, 2*h))
        self.add_parameter(Param('Ratio',  'ratio', 3., 1.1, 20.))
        self.add_parameter(Param('Level', 'level', 0.5, 0., 1.))
        self.add_parameter(Param('Processing Block length (rel. to long window)', 'block_factor', 10., 2., 100.,))
        self.add_parameter(Switch('Show trigger level traces', 'show_level_traces', False))
        self.add_parameter(Switch('Apply to full dataset', 'apply_to_all', False))
        self.add_parameter(Choice('Scaling/Normalization method', 'scalingmethod', '[0-1]', scalingmethods))
        
        self.set_live_update(False)


    def call(self):
        '''Main work routine of the snuffling.'''
        
        self.cleanup()
        
        swin, ratio = self.swin, self.ratio
        lwin = swin * ratio
        tpad = lwin/2.
        
        pile = self.get_pile()
        tmin, tmax = pile.get_tmin() + tpad, pile.get_tmax() - tpad
        
        if not self.apply_to_all:
            vtmin, vtmax = self.get_viewer().get_time_range()
            tmin = max(vtmin, tmin)
            tmax = min(vtmax, tmax)

        tinc = min(lwin * self.block_factor, tmax-tmin)
        
        show_level_traces = self.show_level_traces
        
       # if show_level_traces and tmax-tmin > lwin * 150:
       #     self.error('Processing time window is longer than 50 x LTA window. Turning off display of level traces.')
       #     show_level_traces = False
        
        markers = []
        for traces in pile.chopper(tmin=tmin, tmax=tmax, tinc=tinc, tpad=tpad, want_incomplete=False):
            sumtrace = None
            isum = 0
            for trace in traces:
                if self.lowpass is not None:
                    trace.lowpass(4, self.lowpass, nyquist_exception=True)

                if self.highpass is not None:
                    trace.highpass(4, self.highpass, nyquist_exception=True)
                
                trace.sta_lta_centered(swin, lwin, scalingmethod=scalingmethod_map[self.scalingmethod])
                trace.chop(trace.wmin, min(trace.wmax,tmax))
                trace.set_codes(location='cg')
                trace.meta = { 'tabu': True }
                
                #print trace.ydata.max()
                if sumtrace is None:
                    sumtrace = trace.copy()
                    sumtrace.set_codes(network='', station='SUM', location='cg', channel='')
                else:
                    sumtrace.add(trace)
                isum += 1
    
            if show_level_traces:
                self.add_traces(traces)
    
            if sumtrace is not None:
                tpeaks, apeaks = sumtrace.peaks(self.level*isum, swin)
    
                for t, a in zip(tpeaks, apeaks):
                    mark = Marker([  ], t, t)
                    print mark, a #'%s.%s.%s.%s' % ('', trace.station, '', trace.channel)
                    markers.append(mark)
                
                if show_level_traces:
                    self.add_trace(sumtrace)
                    
        self.add_markers(markers)
    
def __snufflings__():    
   return [ DetectorSTALTA() ]
