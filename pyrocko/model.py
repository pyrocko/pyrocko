import orthodrome, config
d2r = num.pi/180.

# simple flat datatypes until I have a better idea

class Event:
    def __init__(self, lat=0., lon=0., time=0., name='', depth=None, magnitude=None):
        self.lat = lat
        self.lon = lon
        self.time = time
        self.name = name
        self.depth = depth
        self.magnitude = magnitude
        
class Station:
    def __init__(self, network, station, location, lat, lon, elevation, name='', channels=None):
        self.network = network
        self.station = station
        self.location = location
        self.lat = lat
        self.lon = lon
        self.elevation = elevation
        self.name = name
        if channels is None:
            self.channels = set()
        else:
            self.channels = channels
        
        self.dist_deg = None
        self.dist_m = None
        self.azimuth = None
        self.backazimuth = None
        self.channel_map = None

    def set_event_relative_data( self, event ):
        self.dist_m = orthodrome.distance_accurate50m( event, self )
        self.dist_deg = self.dist_m / orthodrome.earthradius_equator *orthodrome.r2d
        self.azimuth = orthodrome.azimuth(event, self)
        self.backazimuth = orthodrome.azimuth(self, event)
        
    self.set_components(self, components):
        self.components = components
    
    def projection_to_enu(self):
        assert len(self.components) == 3
        return num.hstack( [component.enu for component in self.components ] )

    def projection_to_ned(self):
        assert len(self.components) == 3
        return num.hstack( [component.ned for component in self.components ] )
        
    def __str__(self):
        return '%s.%s.%s  %f %f %f  %f %f %f  %s' % (self.network, self.station, self.location, self.lat, self.lon, self.elevation, self.dist_m, self.dist_deg, self.azimuth, self.name)

class Component:
    def __init__(self, name, azimuth, dip, gain=1.0):
        self.name = name
        self.azimuth = azimuth
        self.dip = dip
        self.gain = gain
        n = math.cos(self.azimuth*d2r)*math.cos(self.dip*d2r)
        e = math.sin(self.azimuth*d2r)*math.cos(self.dip*d2r)
        d = math.sin(self.dip*d2r)
        self.ned = num.matrix( [[n],[e],[d]], dtype=num.float )
        self.enu = num.matrix( [[e],[n],[-d]], dtype=num.float )
        
        
    