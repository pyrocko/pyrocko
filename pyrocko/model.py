import orthodrome, config

# simple flat datatypes until I have a better idea

class Event:
    def __init__(self, lat=0., lon=0., time=0., name='', depth=None, magnitude=None, region=None):
        self.lat = lat
        self.lon = lon
        self.time = time
        self.name = name
        self.depth = depth
        self.magnitude = magnitude
        self.region = region
        
class Station:
    def __init__(self, network, station, location, lat, lon, elevation, name='', components=None):
        self.network = network
        self.station = station
        self.location = location
        self.lat = lat
        self.lon = lon
        self.elevation = elevation
        self.name = name
        if components is None:
            self.components = set()
        else:
            self.components = components
            
        self.dist_deg = None
        self.dist_m = None
        self.azimuth = None
        self.backazimuth = None

    def set_event_relative_data( self, event ):
        self.dist_m = orthodrome.distance_accurate50m( event, self )
        self.dist_deg = self.dist_m / orthodrome.earthradius_equator *orthodrome.r2d
        self.azimuth = orthodrome.azimuth(event, self)
        self.backazimuth = orthodrome.azimuth(self, event)
        
    def __str__(self):
        return '%s.%s.%s  %f %f %f  %f %f %f  %s' % (self.network, self.station, self.location, self.lat, self.lon, self.elevation, self.dist_m, self.dist_deg, self.azimuth, self.name)
