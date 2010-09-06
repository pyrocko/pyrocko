import catalog, orthodrome, config, util, moment_tensor
import math
import numpy as num

from orthodrome import wrap

d2r = num.pi/180.

def guess_azimuth_from_name(channel_name):
    if channel_name.endswith('N'):
        return 0.
    elif channel_name.endswith('E'):
        return 90.
    elif channel_name.endswith('Z'):
        return 0.
    
    return None

def guess_dip_from_name(channel_name):
    if channel_name.endswith('N'):
        return 0.
    elif channel_name.endswith('E'):
        return 0.
    elif channel_name.endswith('Z'):
        return -90.
    
    return None

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

class FileParseError(Exception):
    pass

class Event:
    def __init__(self, lat=0., lon=0., time=0., name='', depth=None, magnitude=None, region=None, load=None):
        if load is not None:
            self.load(load)
        else:
            self.lat = lat
            self.lon = lon
            self.time = time
            self.name = name
            self.depth = depth
            self.magnitude = magnitude
            self.region = region
            
    def time_as_string(self):
        return util.gmctime(self.time)
    
    def set_name(self, name):
        self.name = name
        
    def __str__(self):
        return '%s %s %s %g %g %s %s' % (self.name, util.gmctime(self.time), self.magnitude, self.lat, self.lon, self.depth, self.region)
                
    def dump(self, filename):
        f = open(filename, 'w')
        f.write('name = %s\n' % self.name)
        f.write('time = %s\n' % util.gmctime(self.time))
        f.write('latitude = %g\n' % self.lat)
        f.write('longitude = %g\n' % self.lon)
        if self.magnitude is not None:
            f.write('magnitude = %g\n' % self.magnitude)
            f.write('moment = %g\n' % moment_tensor.magnitude_to_moment(self.magnitude))
        if self.depth is not None:
            f.write('depth = %g\n' % self.depth)
        if self.region is not None:
            f.write('region = %s\n' % self.region)
        f.close()
        
    def load(self, filename):
        f = open(filename, 'r')
        d = {}
        try:
            try:
                for line in f:
                    toks = line.split(' = ',1)
                    if len(toks) == 2:
                        k,v = toks[0].strip(), toks[1].strip()
                        if k in ('name', 'region'):
                            d[k] = v
                        if k in ('latitude', 'longitude', 'magnitude', 'depth'):
                            d[k] = float(v)
                        if k == 'time':
                            d[k] = util.ctimegm(v[:19])
                        
            except Exception, e:
                raise FileParseError(e)
        finally:
            f.close()
            
        self.lat = d.get('latitude', 0.0)
        self.lon = d.get('longitude', 0.0)
        self.time = d.get('time', 0.0)
        self.name = d.get('name', '')
        self.depth = d.get('depth', None)
        self.magnitude = d.get('magnitude', None)
        self.region = d.get('region', None)
        

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
        self.channels = {}
        if channels is not None:
            self.set_channels(channels)
        
        self.dist_deg = None
        self.dist_m = None
        self.azimuth = None
        self.backazimuth = None

    def set_event_relative_data( self, event ):
        self.dist_m = orthodrome.distance_accurate50m( event, self )
        self.dist_deg = self.dist_m / orthodrome.earthradius_equator *orthodrome.r2d
        self.azimuth = orthodrome.azimuth(event, self)
        self.backazimuth = orthodrome.azimuth(self, event)
        
    def set_channels(self, channels):
        self.channels = {}
        for ch in channels:
            self.channels[ch.name] = ch
        
    def get_channels(self):
        return [ self.channels[k] for k in sorted(self.channels) ]
        
    def add_channel(self, channel):
        self.channels[channel.name] = channel
            
    def get_channel(self, name):
        if name in self.channels:
            return self.channels[name]
        
        return None
    
    def rotation_ne_to_rt(self, in_channel_names, out_channel_names):
        
        angle = wrap(self.backazimuth + 180., -180., 180.)
        in_channels = [ self.get_channel(name) for name in in_channel_names ]
        out_channels = [
            Channel(out_channel_names[0], wrap(self.backazimuth+180., -180.,180.),  0., 1.),
            Channel(out_channel_names[1], wrap(self.backazimuth+270., -180.,180.),  0., 1.) ]
        return angle, in_channels, out_channels
    
    def _projection_to(self, to, in_channel_names, out_channel_names, use_gains=False):
        in_channels = [ self.get_channel(name) for name in in_channel_names ]
        
        # create orthogonal vectors for missing components, such that this 
        # won't break projections when components are missing.
        
        vecs = []
        for ch in in_channels:
            if ch is None: 
                vecs.append(None)
            else:
                vec = getattr(ch,to)
                if use_gains:
                    vec /= ch.gain
                vecs.append(vec)
                
        fill_orthogonal(vecs)
        
        m = num.hstack([ vec[:,num.newaxis] for vec in vecs ])
        
        m = num.where(num.abs(m) < num.max(num.abs(m))*1e-16, 0., m)
        
        if to == 'ned': 
            out_channels = [
                Channel(out_channel_names[0], 0.,   0., 1.),
                Channel(out_channel_names[1], 90.,  0., 1.),
                Channel(out_channel_names[2], 0.,  90., 1.) ]
                
        elif to == 'enu':
            out_channels = [
                Channel(out_channel_names[0], 90.,  0., 1.),
                Channel(out_channel_names[1], 0.,   0., 1.),
                Channel(out_channel_names[2], 0., -90., 1.) ]
        
        return m, in_channels, out_channels
    
    def projection_to_enu(self, in_channels, out_channels=('E', 'N', 'U'), **kwargs):
        return self._projection_to('enu', in_channels, out_channels, **kwargs)

    def projection_to_ned(self, in_channels, out_channels=('N', 'E', 'D'), **kwargs):
        return self._projection_to('ned', in_channels, out_channels, **kwargs)
        
    def projection_from_enu(self, in_channels=('E','N','U'), out_channels=('X','Y','Z'), **kwargs):
        m, out_channels, in_channels = self._projection_to('enu', out_channels,in_channels, **kwargs)
        return num.linalg.inv(m), in_channels, out_channels
    
    def projection_from_ned(self, in_channels=('N','E','D'), out_channels=('X','Y','Z'), **kwargs):
        m, out_channels, in_channels = self._projection_to('enu', out_channels,in_channels, **kwargs)
        return num.linalg.inv(m), in_channels, out_channels
        
    def nsl_string(self):
        return '.'.join((self.network, self.station, self.location))
        
    def __str__(self):
        sta = self
        elevation = 0.0
        if sta.elevation is not None:
            elevation = sta.elevation
            
        depth = 0.0
        if sta.depth is not None:
            depth = sta.depth
        nsl = '%s.%s.%s' % (sta.network, sta.station, sta.location)
        s = '%-15s  %14g %14g %14g %14g %s' % (nsl, sta.lat, sta.lon, elevation, depth, sta.name)
        return s

def dump_stations(stations, filename):
    f = open(filename, 'w')
    for sta in stations:
        f.write(str(sta)+'\n')
        for cha in sta.get_channels():
            azimuth = 'NaN'
            if cha.azimuth is not None:
                azimuth = '%7g' % cha.azimuth
                
            dip = 'NaN'
            if cha.dip is not None:
                dip = '%7g' % cha.dip
            
            f.write( '%5s %14s %14s %14g\n' % (cha.name, azimuth, dip, cha.gain) )
            
    f.close()
    
def float_or_none(s):
    if s.lower() == 'nan':
        return None
    else:
        return float(s)
    
def load_stations(filename):
    stations = []
    f = open(filename, 'r')
    station = None
    for line in f:
        toks = line.split(None, 5)
        if len(toks) == 5 or len(toks) == 6:
            
            net, sta, loc = toks[0].split('.')
            lat, lon, elevation, depth = [ float(x) for x in toks[1:5] ]
            if len(toks) == 5:
                name = ''
            else:
                name =toks[5]
            station = Station(net, sta, loc, lat, lon, elevation=elevation, depth=depth, name=name)
            stations.append(station)
        elif len(toks) == 4 and station is not None:
            name, azi, dip, gain = toks[0], float_or_none(toks[1]), float_or_none(toks[2]), float(toks[3])
            channel = Channel(name, azimuth=azi, dip=dip, gain=gain)
            station.add_channel(channel)
            
    f.close()
    return stations

class Channel:
    def __init__(self, name, azimuth=None, dip=None, gain=1.0):
        self.name = name
        if azimuth is None:
            azimuth = guess_azimuth_from_name(name)
        if dip is None:
            dip = guess_dip_from_name(name)
        
        self.azimuth = azimuth
        self.dip = dip
        self.gain = gain
        self.ned = None
        self.enu = None
        if azimuth is not None and dip is not None:
            n = math.cos(self.azimuth*d2r)*math.cos(self.dip*d2r)
            e = math.sin(self.azimuth*d2r)*math.cos(self.dip*d2r)
            d = math.sin(self.dip*d2r)
        
            self.ned = mkvec(n,e,d)
            self.enu = mkvec(e,n,-d)
        
    def __str__(self):
        return '%s %f %f %g' % (self.name, self.azimuth, self.dip, self.gain)

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
        
def load_gfz_event_list(filename):
    cat = catalog.Geofon()
    
    elist =[]
    f = open(filename, 'r')
    for line in f:
        e = cat.get_event(line.strip())
        elist.append(e)
        
    f.close()
    return elist

    
