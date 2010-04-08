import orthodrome, config, util, math
import numpy as num

d2r = num.pi/180.

def mkvec(x,y,z):
    return num.array( [x,y,z], dtype=num.float )

def fill_orthogonal(enus):
    
    nmiss = sum( x is None for x in enus )
    
    if nmiss == 1:
        for ic in range(len(enus)):
            if enus[ic] is None:
                enus[ic] = num.cross(enus[(ic-2)%3],enus[(ic-1)%3])
    
    if nmiss == 2:
        for ic in range(len(enus)):
            if enus[ic] is not None:
                xenu = enus[ic] + mkvec(1,1,1)
                enus[(ic+1)%3] = num.cross(enus[ic], xenu)
                enus[(ic+2)%3] = num.cross(enus[ic], enus[(ic+1)%3])

    if nmiss == 3:
        # holy camoly..
        enus[0] = mkvec(1,0,0)
        enus[1] = mkvec(0,1,0)
        enus[2] = mkvec(0,0,1)

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
        
    def __str__(self):
        return '%s %s %g %g %g %g %s' % (self.name, util.gmctime(self.time), self.magnitude, self.lat, self.lon, self.depth, self.region)
                
class Station:
    def __init__(self, network, station, location, lat, lon, elevation, depth=None, name='', channels=None):
        self.network = network
        self.station = station
        self.location = location
        self.lat = lat
        self.lon = lon
        self.elevation = elevation
        self.depth = depth
        self.name = name
        if channels is None:
            self.channels = []
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
        
    def set_channels(self, channels):
        self.channels = channels
        
    def get_channels(self):
        return self.channels
        
    def add_channel(self, channel):
        self.channels.append(channel)
            
    def get_channel(self, name):
        for channel in self.channels:
            if channel.name == name:
                return channel
            
        return None
    
    
    def _projection_to(self, to, in_channels, out_channels, divide_by_gains=False):
        channels = [ self.get_channel(name) for name in in_channels ]
        
        # create orthogonal vectors for missing components, such that this 
        # won't break projections when components are missing.
        
        vecs = []
        for ch in channels:
            if ch is None: 
                vecs.append(None)
            else:
                vec = getattr(ch,to)
                if divide_by_gains:
                    vec /= ch.gain
                vecs.append(vec)
                
        fill_orthogonal(vecs)
        
        m = num.hstack([ vec[:,num.newaxis] for vec in vecs ])
        
        m = num.where(num.abs(m) < num.max(num.abs(m))*1e-16, 0., m)
        return m, in_channels, out_channels
    
    def projection_to_enu(self, in_channels, out_channels=('E', 'N', 'U'), **kwargs):
        return self._projection_to('enu', in_channels, out_channels, **kwargs)

    def projection_to_ned(self, in_channels, out_channels=('N', 'E', 'D'), **kwargs):
        return self._projection_to('ned', in_channels, out_channels, **kwargs)
        
    def __str__(self):
        return '%s.%s.%s  %f %f %f  %f %f %f  %s' % (self.network, self.station, self.location, self.lat, self.lon, self.elevation, self.dist_m, self.dist_deg, self.azimuth, self.name)

class Channel:
    def __init__(self, name, azimuth, dip, gain=1.0):
        self.name = name
        self.azimuth = azimuth
        self.dip = dip
        self.gain = gain
        n = math.cos(self.azimuth*d2r)*math.cos(self.dip*d2r)
        e = math.sin(self.azimuth*d2r)*math.cos(self.dip*d2r)
        d = math.sin(self.dip*d2r)
        self.ned = mkvec(n,e,d)
        self.enu = mkvec(e,n,-d)

def load_kps_event_list(filename):
    elist =[]
    f = open(filename, 'r')
    for line in f:
        toks = line.split()
        if len(toks) < 7: continue
        
        tim = util.ctimegm(toks[0]+' '+toks[1])
        lat, lon, depth, magnitude = [ float(x) for x in toks[2:6] ]
        region = toks[-1]
        name = util.gmctime_fn(tim)
        e = Event(lat, lon, tim, name, depth, magnitude)
        
        elist.append(e)
        
    f.close()
    return elist
