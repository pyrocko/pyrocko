from pyrocko import pile, trace, util, io, iris_ws
from pyrocko.gui_util import EventMarker
import sys, os, math, time, urllib2
import numpy as num
from pyrocko.snuffling import Param, Snuffling, Switch
pjoin = os.path.join

class IrisData(Snuffling):

    def setup(self):    
        '''Customization of the snuffling.'''
        
        self.set_name('Iris Data')
        self.add_parameter(Param('Min Radius [deg]', 'minradius', 0., 0., 20.))
        self.add_parameter(Param('Max Radius [deg]', 'maxradius', 5., 0., 20.))
        self.add_parameter(Param('Origin latitude [deg]', 'lat', 0, -90., 90.))
        self.add_parameter(Param('Origin longitude [deg]', 'lon', 0., -180., 180.))
        self.add_parameter(Switch('Use coordinates of selected event as origin', 'useevent', False))
        self.set_live_update(False)

    def call(self):
        '''Main work routine of the snuffling.'''
        
        self.cleanup()

        view = self.get_viewer()
        pile = self.get_pile()

        tmin, tmax = view.get_time_range()
        if self.useevent:
            markers = view.selected_markers()
            if len(markers) != 1:
                self.fail('Exactly one marker must be selected.')
            marker = markers[0]
            if not isinstance(marker, EventMarker):
                self.fail('An event marker must be selected.')

            ev = marker.get_event()
            
            lat, lon = ev.lat, ev.lon
        else:
            lat, lon = self.lat, self.lon
        
        print lat, lon, self.minradius, self.maxradius, util.time_to_str(tmin), util.time_to_str(tmax)

        data = iris_ws.ws_station(lat=lat, lon=lon, minradius=self.minradius, maxradius=self.maxradius, 
                                                     timewindow=(tmin,tmax), level='chan' )
        stations = iris_ws.grok_station_xml(data, tmin, tmax)
        networks = set( [ s.network for s in stations ] )
        
        dir = self.tempdir()
        fns = []
        for net in networks:
            nstations = [ s for s in stations if s.network == net ]
            selection = sorted(iris_ws.data_selection( nstations, tmin, tmax ))
            if selection:
                for x in selection:
                    print x
                
                try:
                    d = iris_ws.ws_bulkdataselect(selection)
                    fn = pjoin(dir,'data-%s.mseed' % net) 
                    f = open(fn, 'w')
                    f.write(d)
                    f.close()
                    fns.append(fn)
                except urllib2.HTTPError:
                    pass

        newstations = []
        for sta in stations:
            if not view.has_station(sta):
                print sta
                newstations.append(sta)

        view.add_stations(newstations)
        for fn in fns:
            traces = list(io.load(fn))
            self.add_traces(traces)

def __snufflings__():    
   return [ IrisData() ]

