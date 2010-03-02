import trace, util

import logging, copy
import numpy as num

logger = logging.getLogger('pyrocko.eventdata')

class NoRestitution(Exception):
    pass


class EventDataAccess:
    '''Abstract base class for event data access (see rdseed.py)'''
    
    def __init__(self, datapile=None):
        
        self._pile = datapile
        self._events = None
        self._stations = None
    
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
                
        stations = copy.deepcopy(self._stations)
                
        if relative_event is not None:
            
            for s in stations.values():
                s.set_event_relative_data(relative_event)
        
        return stations
        
    def iter_displacement_traces(self, tfade, freqband, deltat=None, rotate=None, maxdisplacement=None):
        
        if rotate is not None:
            angles_func, rotation_mappings = rotate
                
        for traces in self.get_pile().chopper_grouped(
                gather=lambda tr: (tr.network, tr.station, tr.location)):
            
            traces.sort( lambda a,b: cmp(a.full_id, b.full_id) )
            
            traces = trace.degapper(traces)  # mainly to get rid if overlaps and duplicates
            if traces:
                displacements = []
                for tr in traces:
                    if deltat is not None:
                        try:
                            tr.downsample_to(deltat, snap=True)
                        except util.UnavailableDecimation, e:
                            logger.warn( 'Cannot downsample %s.%s.%s.%s: %s' % (tr.nslc_id + (e,)))
                            continue
                        
                    try:
                        trans = self.get_restitution(tr)
                    except NoRestitution, e:
                        logger.warn( 'Cannot restitute trace %s.%s.%s.%s: %s' % (tr.nslc_id + (e,)))
                        continue
                    try:
                        displacement = tr.transfer( tfade, freqband, transfer_function=trans )
                        tmax = num.max(num.abs(displacement.get_ydata()))
                        if maxdisplacement is not None and tmax > maxdisplacement:
                            logger.warn( 'Trace %s.%s.%s.%s has too large displacement: %g' % (tr.nslc_id + (tmax,)) )
                            continue
                        
                        if not num.all(num.isfinite(displacement.get_ydata())):
                            logger.warn( 'Trace %s.%s.%s.%s has NaNs' % tr.nslc_id )
                            continue
                            
                    except trace.TraceTooShort, e:
                        logger.warn( '%s' % e )
                        continue
                    
                    displacements.append(displacement)
                
                rotated = []
                if rotate:
                    try:
                        angle = angles_func(tr)
                    except Exception, e:
                        logger.warn( 'Cannot get station locations for rotation of traces at %s.%s.%s: %s' % (tr.nslc_id[:3]+(e,)) )
                        continue
                    
                    for in_channels, out_channels in rotation_mappings:
                        rotated = trace.rotate(displacements, angle, in_channels, out_channels)
                        displacements.extend(rotated)
                yield displacements
                
    def get_restitution(self, tr):
        return tr.IntegrationResponse()
