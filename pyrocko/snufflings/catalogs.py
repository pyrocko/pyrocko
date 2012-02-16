from pyrocko.snuffling import Snuffling, Param, Choice
from pyrocko.gui_util import Marker, EventMarker

from pyrocko import catalog

class CatalogSearch(Snuffling):
    

    def setup(self):
        
        self.catalogs = { 'Geofon': catalog.Geofon(),
                        'USGS/NEIC PDE': catalog.USGS('PDE'), 
                        'USGS/NEIC PDE-Q': catalog.USGS('PDE-Q'),
                        'Global-CMT': catalog.GlobalCMT(), }

        catkeys = sorted(self.catalogs.keys())
        self.set_name('Catalog Search')
        self.add_parameter(Choice('Catalog', 'catalog', catkeys[0], catkeys)) 
        self.add_parameter(Param('Min Magnitude', 'magmin', 0, 0, 10))
        self.set_live_update(False)
        
        
    def call(self):
       
        viewer = self.get_viewer()
        tmin, tmax = viewer.get_time_range()
        
        cat = self.catalogs[self.catalog]
        event_names = cat.get_event_names(
            time_range=(tmin,tmax), 
            magmin=self.magmin)
            
        for event_name in event_names:
            event = cat.get_event(event_name)
            marker = EventMarker(event)
            self.add_markers([marker])
                
def __snufflings__():
    
    return [ CatalogSearch() ]
