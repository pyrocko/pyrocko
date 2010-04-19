import trace, util, model

import logging, copy
import numpy as num
import cPickle as pickle

logger = logging.getLogger('pyrocko.eventdata')

class NoRestitution(Exception):
    pass

class Problems:
    def __init__(self):
        self._problems = {}
        
    def add(self, kind, nslct):
        if kind not in self._problems:
            self._problems[kind] = set()
        problem = self._problems[kind]
        problem.add(nslct)
        
    def dump(self, fn):
        f = open(fn, 'w')
        pickle.dump(self._problems, f)
        f.close()
        
    def load(self, fn):
        f = open(fn, 'r')
        self._problems = pickle.load(f)
        f.close()
    
    def mapped(self, mapping=lambda nslct: nslct[:3]):
        p = {}
        for kind, problem in self._problems.iteritems():            
            nsl = set()
            for nslct in problem:
                nsl.add(mapping(nslct))
            p[kind] = nsl
        
        return p
        
class EventDataAccess:
    '''Abstract base class for event data access (see rdseed.py)'''
    
    def __init__(self, datapile=None):
        
        self._pile = datapile
        self._events = None
        self._stations = None
        self._problems = Problems()
    
    def get_pile(self):
        return self._pile
    
    def get_events(self):
        if not self._events:
            self._events = self._get_events_from_file()
        return self._events
        
    def get_stations(self, relative_event=None):
        
        if not self._stations:
            self._stations = {}
            for station in self._get_stations_from_file():
                self._stations[station.network, station.station, station.location] = station
            self._insert_channel_descriptions(self._stations)
        
        stations = copy.deepcopy(self._stations)
        
        if relative_event is not None:
            for s in stations.values():
                s.set_event_relative_data(relative_event)
                
        return stations
        
    def _insert_channel_descriptions(self, stations):
        pile = self.get_pile()
        nslc_ids = pile.gather_keys( lambda tr: (tr.network, tr.station, tr.location, tr.channel) )
        
        for nslc in nslc_ids:
            sta = stations[nslc[:3]]
            sta.add_channel(  self._get_channel_description_from_file(nslc) )
   
    def _get_channel_description_from_file(self, nslc):
        return model.Channel(nslc[3], None, None, 1.)
   
    def iter_traces(self, group_selector=None, trace_selector=None):
         
         for traces in self.get_pile().chopper_grouped(
             gather=lambda tr: (tr.network, tr.station, tr.location),
             group_selector=group_selector,
             trace_selector=trace_selector):
             
             yield traces
               
    def problems(self):
        return self._problems
               
    def iter_displacement_traces( self, tfade, freqband, 
                                  deltat=None,
                                  rotate=None,
                                  project=None,
                                  maxdisplacement=None,
                                  extend=None,
                                  group_selector=None,
                                  trace_selector=None,
                                  allowed_methods=None,
                                  crop=True):
        
        stations = self.get_stations()
        
        if rotate is not None:
            angles_func, rotation_mappings = rotate
            
        for traces in self.get_pile().chopper_grouped(
                gather=lambda tr: (tr.network, tr.station, tr.location),
                group_selector=group_selector,
                trace_selector=trace_selector,
                progress='Processing traces'):
            
            traces.sort( lambda a,b: cmp(a.full_id, b.full_id) )
            
            traces = trace.degapper(traces)  # mainly to get rid if overlaps and duplicates
            if traces:
                displacements = []
                for tr in traces:
                    if deltat is not None:
                        try:
                            tr.downsample_to(deltat, snap=True)
                        except util.UnavailableDecimation, e:
                            self.problems().add('cannot_downsample', tr.full_id)
                            logger.warn( 'Cannot downsample %s.%s.%s.%s: %s' % (tr.nslc_id + (e,)))
                            continue
                        
                    try:
                        trans = self.get_restitution(tr, allowed_methods)
                    except NoRestitution, e:
                        self.problems().add('no_response', tr.full_id)
                        logger.warn( 'Cannot restitute trace %s.%s.%s.%s: %s' % (tr.nslc_id + (e,)))
                        continue
                    
                    try:
                        if extend:
                            tr.extend(tr.tmin+extend[0], tr.tmax+extend[1], fillmethod='repeat')
                            
                        displacement = tr.transfer( tfade, freqband, transfer_function=trans, cut_off_fading=crop )
                        amax = num.max(num.abs(displacement.get_ydata()))
                        if maxdisplacement is not None and amax > maxdisplacement:
                            self.problems().add('unrealistic_amplitude', tr.full_id)
                            logger.warn( 'Trace %s.%s.%s.%s has too large displacement: %g' % (tr.nslc_id + (amax,)) )
                            continue
                        
                        if not num.all(num.isfinite(displacement.get_ydata())):
                            self.problems().add('has_nan_or_inf', tr.full_id)
                            logger.warn( 'Trace %s.%s.%s.%s has NaNs or Infs' % tr.nslc_id )
                            continue
                            
                    except trace.TraceTooShort, e:
                        self.problems().add('gappy', tr.full_id)
                        logger.warn( '%s' % e )
                        continue
                    
                    displacements.append(displacement)
                
                if project:
                    station = stations[tr.network, tr.station, tr.location]
                    
                    matrix, in_channels, out_channels = project(station)
                    projected = trace.project(displacements, matrix, in_channels, out_channels)
                    displacements.extend(projected)
                
                if rotate:
                    angle = angles_func(tr)
                    for in_channels, out_channels in rotation_mappings:
                        rotated = trace.rotate(displacements, angle, in_channels, out_channels)
                        displacements.extend(rotated)
                        
                yield displacements
                
    def get_restitution(self, tr):
        return tr.IntegrationResponse()
