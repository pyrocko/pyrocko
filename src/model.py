from pyrocko import orthodrome, util, moment_tensor
import math, copy, logging
import numpy as num

from pyrocko.orthodrome import wrap
from pyrocko.guts import Object, Float, String, Timestamp, List

logger = logging.getLogger('pyrocko.model')

guts_prefix = 'pf'

d2r = num.pi/180.

class ChannelsNotOrthogonal(Exception):
    pass

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

def guess_azimuth_dip_from_name(channel_name):
    return guess_azimuth_from_name(channel_name), \
           guess_dip_from_name(channel_name)

def mkvec(x,y,z):
    return num.array( [x,y,z], dtype=num.float )

def are_orthogonal(enus, eps=0.05):
    return all(abs(x) < eps for x in [
        num.dot(enus[0], enus[1]),
        num.dot(enus[1], enus[2]),
        num.dot(enus[2], enus[0])])

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


class FileParseError(Exception):
    pass

class EOF(Exception):
    pass

class EmptyEvent(Exception):
    pass

class Event(Object):
    '''Seismic event representation

    :param lat: latitude of hypocenter (default 0.0)
    :param lon: longitude of hypocenter (default 0.0) 
    :param time: origin time as float in seconds after '1970-01-01 00:00:00
    :param name: event identifier as string (optional)
    :param depth: source depth (optional)
    :param magnitude: magnitude of event (optional)
    :param region: source region (optional)
    :param catalog: name of catalog that lists this event (optional)
    :param moment_tensor: moment tensor as :py:class:`moment_tensor.MomentTensor`
                            instance (optional)
    :param duration: source duration as float (optional)
    '''

    lat = Float.T(default=0.0)
    lon = Float.T(default=0.0)
    time = Timestamp.T(default=util.str_to_time('1970-01-01 00:00:00'))
    name = String.T(default='', optional=True)
    depth = Float.T(optional=True)
    magnitude = Float.T(optional=True)
    magnitude_type = String.T(optional=True)
    region = String.T(optional=True)
    catalog = String.T(optional=True)
    moment_tensor = moment_tensor.MomentTensor.T(optional=True)
    duration = Float.T(optional=True)

    def __init__(self, lat=0., lon=0., time=0., name='', depth=None,
            magnitude=None, magnitude_type=None, region=None, load=None,
            loadf=None, catalog=None, moment_tensor=None, duration=None):

        vals = None
        if load is not None:
            vals = Event.oldload(load)
        elif loadf is not None:
            vals = Event.oldloadf(loadf)

        if vals:
            lat, lon, time, name, depth, magnitude, magnitude_type, region, \
                catalog, moment_tensor, duration = vals
            
        Object.__init__(self, lat=lat, lon=lon, time=time, name=name, depth=depth,
                        magnitude=magnitude, magnitude_type=magnitude_type,
                        region=region, catalog=catalog,
                        moment_tensor=moment_tensor, duration=duration)
            
    def time_as_string(self):
        return util.time_to_str(self.time)
    
    def set_name(self, name):
        self.name = name
        
    #def __str__(self):
    #    return '%s %s %s %g %g %s %s' % (self.name, util.time_to_str(self.time), self.magnitude, self.lat, self.lon, self.depth, self.region)
                
    def olddump(self, filename):
        file = open(filename, 'w')
        self.olddumpf(file)
        file.close()
        
    def olddumpf(self, file):
        file.write('name = %s\n' % self.name)
        file.write('time = %s\n' % util.time_to_str(self.time))
        if self.lat is not None:
            file.write('latitude = %.12g\n' % self.lat)
        if self.lon is not None:
            file.write('longitude = %.12g\n' % self.lon)
        if self.magnitude is not None:
            file.write('magnitude = %g\n' % self.magnitude)
            file.write('moment = %g\n' % moment_tensor.magnitude_to_moment(self.magnitude))
        if self.magnitude_type is not None:
            file.write('magnitude_type = %s\n' % self.magnitude_type)
        if self.depth is not None:
            file.write('depth = %.10g\n' % self.depth)
        if self.region is not None:
            file.write('region = %s\n' % self.region)
        if self.catalog is not None:
            file.write('catalog = %s\n' % self.catalog)
        if self.moment_tensor is not None:
            m = self.moment_tensor.m()
            sdr1, sdr2 = self.moment_tensor.both_strike_dip_rake()
            file.write(('mnn = %g\nmee = %g\nmdd = %g\nmne = %g\nmnd = %g\nmed = %g\n'+
                        'strike1 = %g\ndip1 = %g\nrake1 = %g\nstrike2 = %g\ndip2 = %g\nrake2 = %g\n')
                               % ((m[0,0],m[1,1],m[2,2],m[0,1],m[0,2],m[1,2]) + sdr1 + sdr2) )
        if self.duration is not None:
            file.write('duration = %g\n' % self.duration)

    @staticmethod
    def unique(events, deltat=10., group_cmp=(lambda a,b: cmp(a.catalog, b.catalog))):
        groups = Event.grouped(events, deltat)
        
        events = []
        for group in groups:
            if group:
                group.sort(group_cmp)
                events.append(group[-1]) 
        
        return events

    @staticmethod
    def grouped(events, deltat=10.):
        events = list(events)
        groups = []
        for ia,a in enumerate(events):
            groups.append([])
            haveit = False
            for ib,b in enumerate(events[:ia]):
                if abs(b.time - a.time) < deltat:
                    groups[ib].append(a)
                    haveit = True
                    break

            if not haveit:
                groups[ia].append(a)

        
        groups = [ g for g in groups if g ]
        groups.sort( key=lambda g: sum(e.time for e in g)/len(g) )
        return groups

    @staticmethod
    def dump_catalog(events, filename=None, stream=None):
        if filename is not None:
            file = open(filename, 'w')
        else:
            file = stream
        try:
            i = 0
            for ev in events:

                ev.olddumpf(file)

                file.write('--------------------------------------------\n')
                i += 1

        finally: 
            if filename is not None:
                file.close()
    
    @staticmethod
    def oldload(filename):
        with open(filename, 'r') as file:
            return Event.oldloadf(file)
    
    @staticmethod
    def oldloadf(file):
        d = {}
        try:
            for line in file:
                if line.lstrip().startswith('#'):
                    continue

                toks = line.split(' = ',1)
                if len(toks) == 2:
                    k,v = toks[0].strip(), toks[1].strip()
                    if k in ('name', 'region', 'catalog', 'magnitude_type'):
                        d[k] = v
                    if k in ('latitude longitude magnitude depth duration mnn mee mdd mne mnd med strike1 dip1 rake1 strike2 dip2 rake2 duration'.split()):
                        d[k] = float(v)
                    if k == 'time':
                        d[k] = util.str_to_time(v)
            
                if line.startswith('---'):
                    d['have_separator'] = True
                    break

        except Exception, e:
            raise FileParseError(e)

        if not d:
            raise EOF()
        
        if 'have_separator' in d and len(d) == 1:
            raise EmptyEvent()

        mt = None
        m6 = [ d[x] for x in 'mnn mee mdd mne mnd med'.split() if x in d ]
        if len(m6) == 6:
            mt = moment_tensor.MomentTensor( m = moment_tensor.symmat6(*m6) )
        else:
            sdr = [ d[x] for x in 'strike1 dip1 rake1'.split() if x in d ]
            if len(sdr) == 3:
                moment = 1.0
                if 'moment' in d:
                    moment = d['moment']
                elif 'magnitude' in d:
                    moment = moment_tensor.magnitude_to_moment(d['magnitude'])

                mt = moment_tensor.MomentTensor(strike=sdr[0], dip=sdr[1], rake=sdr[2], scalar_moment=moment)
                
        return (
            d.get('latitude', 0.0),
            d.get('longitude', 0.0),
            d.get('time', 0.0),
            d.get('name', ''),
            d.get('depth', None),
            d.get('magnitude', None),
            d.get('magnitude_type', None),
            d.get('region', None),
            d.get('catalog', None),
            mt,
            d.get('duration', None))

    @staticmethod
    def load_catalog(filename):

        file = open(filename, 'r')
        
        try:
            while True:
                try:
                    ev = Event(loadf=file)
                    yield ev
                except EmptyEvent:
                    pass

        except EOF:
            pass
        
        file.close()

    def get_hash(self):
        e = self
        return util.base36encode(abs(hash((util.time_to_str(e.time), str(e.lat), str(e.lon), str(e.depth), str(e.magnitude), e.catalog, e.name, e.region)))).lower()

    def human_str(self):
        s = [
            'Latitude [deg]: %g' % self.lat,
            'Longitude [deg]: %g' % self.lon,
            'Time [UTC]: %s' % util.time_to_str(self.time)]

        if self.name:
            s.append('Name: %s' % self.name)

        if self.depth is not None:
            s.append('Depth [km]: %g' % (self.depth/1000.))

        if self.magnitude is not None:
            s.append('Magnitude [%s]: %3.1f' % (
                self.magnitude_type or 'M?', self.magnitude))

        if self.region:
            s.append('Region: %s' % self.region)

        if self.catalog:
            s.append('Catalog: %s' % self.catalog)

        if self.moment_tensor:
            s.append(str(self.moment_tensor))

        return '\n'.join(s)


def load_events(filename):
    '''Read events file.

    :param filename: name of file as str
    :returns: list of :py:class:`Event` objects
    '''
    return list(Event.load_catalog(filename))

def load_one_event(filename):
    l = Event.load_catalog(filename)
    return l.next()

def dump_events(events, filename=None, stream=None):
    '''Write events file.

    :param events: list of :py:class:`Event` objects
    :param filename: name of file as str
    '''
    Event.dump_catalog(events, filename=filename, stream=stream)


class Channel(Object):
    name = String.T()
    azimuth = Float.T(optional=True)
    dip = Float.T(optional=True)
    gain = Float.T(default=1.0)

    def __init__(self, name, azimuth=None, dip=None, gain=1.0):
        if azimuth is None:
            azimuth = guess_azimuth_from_name(name)
        if dip is None:
            dip = guess_dip_from_name(name)

        Object.__init__(
            self,
            name=name,
            azimuth=float_or_none(azimuth),
            dip=float_or_none(dip),
            gain=float(gain))

    @property
    def ned(self):
        if None in (self.azimuth, self.dip):
            return None

        n = math.cos(self.azimuth*d2r)*math.cos(self.dip*d2r)
        e = math.sin(self.azimuth*d2r)*math.cos(self.dip*d2r)
        d = math.sin(self.dip*d2r)
        return mkvec(n, e, d)
    
    @property
    def enu(self):
        if None in (self.azimuth, self.dip):
            return None

        n = math.cos(self.azimuth*d2r)*math.cos(self.dip*d2r)
        e = math.sin(self.azimuth*d2r)*math.cos(self.dip*d2r)
        d = math.sin(self.dip*d2r)
        return mkvec(e, n, -d)
        
    def __str__(self):
        return '%s %f %f %g' % (self.name, self.azimuth, self.dip, self.gain)


class Station(Object):
    network = String.T()
    station = String.T()
    location = String.T()
    lat = Float.T(default=0.0)
    lon = Float.T(default=0.0)
    elevation = Float.T(default=0.0)
    depth = Float.T(default=0.0)
    name = String.T(default='')
    channels = List.T(Channel.T())

    def __init__(self, network='', station='', location='', lat=0.0, lon=0.0,
                 elevation=0.0, depth=0.0, name='', channels=None):

        Object.__init__(self,
            network=network, station=station, location=location,
            lat=float(lat), lon=float(lon),
            elevation=elevation and float(elevation) or 0.0,
            depth=depth and float(depth) or 0.0,
            name=name or '',
            channels=channels or [])

        self.dist_deg = None
        self.dist_m = None
        self.azimuth = None
        self.backazimuth = None

    def copy(self):
        return copy.deepcopy(self)

    def set_event_relative_data( self, event ):
        self.dist_m = orthodrome.distance_accurate50m( event, self )
        self.dist_deg = self.dist_m / orthodrome.earthradius_equator *orthodrome.r2d
        self.azimuth = orthodrome.azimuth(event, self)
        self.backazimuth = orthodrome.azimuth(self, event)
       
    def set_channels_by_name(self, *args):
        self.set_channels([])
        for name in args:
            self.add_channel(Channel(name))

    def set_channels(self, channels):
        self.channels = []
        for ch in channels:
            self.add_channel(ch)
        
    def get_channels(self):
        return list(self.channels)

    def get_channel_names(self):
        return set(ch.name for ch in self.channels)

    def remove_channel_by_name(self, name):
        todel = [ch for ch in self.channels if ch.name == name]
        for ch in todel:
            self.channels.remove(ch)
        
    def add_channel(self, channel):
        self.remove_channel_by_name(channel.name)
        self.channels.append(channel)
        self.channels.sort(key=lambda ch: ch.name)
            
    def get_channel(self, name):
        for ch in self.channels:
            if ch.name == name:
                return ch
        
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
        if not are_orthogonal(vecs):
            raise ChannelsNotOrthogonal(
                'components are not orthogonal: station %s.%s.%s, channels %s, %s, %s'
                % (self.nsl() + tuple(in_channel_names)))
        
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

    def guess_channel_groups(self):
        cg = {}
        for channel in self.get_channels():
            if len(channel.name) >= 1:
                kind = channel.name[:-1]
                if kind not in cg:
                    cg[kind] = []
                cg[kind].append(channel.name[-1])

        def allin(a,b):
            return all( x in b for x in a )

        out_groups = []
        for kind, components in cg.iteritems():
            for sys in ('ENZ', '12Z'):
                if allin(sys, components):
                    out_groups.append( tuple([ kind+c for c in sys ]) )

        return out_groups

    def guess_projections_to_enu(self, out_channels=('E', 'N', 'U'), **kwargs):
        proj = []
        for cg in self.guess_channel_groups():
            try:
                proj.append(self.projection_to_enu(cg, out_channels=out_channels, **kwargs))
            except ChannelsNotOrthogonal, e:
                logger.warn(str(e))

        return proj

    def guess_projections_to_rtu(self, out_channels=('R', 'T', 'U'), backazimuth=None, **kwargs):
        if backazimuth is None:
            backazimuth = self.backazimuth
        out_channels_ = [
            Channel(out_channels[0], wrap(backazimuth+180., -180.,180.),  0., 1.),
            Channel(out_channels[1], wrap(backazimuth+270., -180.,180.),  0., 1.),
            Channel(out_channels[2], 0.,  -90., 1.) ]
        
        proj = []
        for (m, in_channels, _) in self.guess_projections_to_enu( **kwargs ):
            phi = (backazimuth + 180.)*d2r
            r = num.array([[math.sin(phi),  math.cos(phi), 0.0],
                           [math.cos(phi), -math.sin(phi), 0.0],
                           [          0.0,            0.0, 1.0]])
            proj.append((num.dot(r,m), in_channels, out_channels_))

        return proj

    def projection_to_enu(self, in_channels, out_channels=('E', 'N', 'U'), **kwargs):
        return self._projection_to('enu', in_channels, out_channels, **kwargs)

    def projection_to_ned(self, in_channels, out_channels=('N', 'E', 'D'), **kwargs):
        return self._projection_to('ned', in_channels, out_channels, **kwargs)
        
    def projection_from_enu(self, in_channels=('E','N','U'), out_channels=('X','Y','Z'), **kwargs):
        m, out_channels, in_channels = self._projection_to('enu', out_channels,in_channels, **kwargs)
        return num.linalg.inv(m), in_channels, out_channels
    
    def projection_from_ned(self, in_channels=('N','E','D'), out_channels=('X','Y','Z'), **kwargs):
        m, out_channels, in_channels = self._projection_to('ned', out_channels,in_channels, **kwargs)
        return num.linalg.inv(m), in_channels, out_channels
        
    def nsl_string(self):
        return '.'.join((self.network, self.station, self.location))
    
    def nsl(self):
        return self.network, self.station, self.location

    def oldstr(self):
        nsl = '%s.%s.%s' % (self.network, self.station, self.location)
        s = '%-15s  %14.5f %14.5f %14.1f %14.1f %s' % (
            nsl, self.lat, self.lon, self.elevation, self.depth, self.name)
        return s

def dump_stations(stations, filename):
    '''Write stations file.

    :param stations: list of :py:class:`Station` objects
    :param filename: filename as str 
    '''
    f = open(filename, 'w')
    for sta in stations:
        f.write(sta.oldstr()+'\n')
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
    if s is None:
        return None
    elif isinstance(s, basestring) and s.lower() == 'nan':
        return None
    else:
        return float(s)
    
def load_stations(filename):
    '''Read stations file.

    :param filename: filename
    :returns: list of :py:class:`Station` objects
    '''
    stations = []
    f = open(filename, 'r')
    station = None
    for (iline, line) in enumerate(f):
        toks = line.split(None, 5)
        if len(toks) == 5 or len(toks) == 6:
            net, sta, loc = toks[0].split('.')
            lat, lon, elevation, depth = [ float(x) for x in toks[1:5] ]
            if len(toks) == 5:
                name = ''
            else:
                name =toks[5].rstrip()

            station = Station(net, sta, loc, lat, lon, elevation=elevation, depth=depth, name=name)
            stations.append(station)
        elif len(toks) == 4 and station is not None:
            name, azi, dip, gain = toks[0], float_or_none(toks[1]), float_or_none(toks[2]), float(toks[3])
            channel = Channel(name, azimuth=azi, dip=dip, gain=gain)
            station.add_channel(channel)

        else:
            logger.warn('skipping invalid station/channel definition '
                        '(line: %i, file: %s' % (iline + 1, filename))

    f.close()
    return stations

def load_kps_event_list(filename):
    elist =[]
    f = open(filename, 'r')
    for line in f:
        toks = line.split()
        if len(toks) < 7: continue
        
        tim = util.ctimegm(toks[0]+' '+toks[1])
        lat, lon, depth, magnitude = [ float(x) for x in toks[2:6] ]
        duration = float(toks[10])
        region = toks[-1]
        name = util.gmctime_fn(tim)
        e = Event(lat, lon, tim, name=name, depth=depth, magnitude=magnitude, duration=duration, region=region)
        
        elist.append(e)
        
    f.close()
    return elist
        
def load_gfz_event_list(filename):
    from pyrocko import catalog
    cat = catalog.Geofon()
    
    elist =[]
    f = open(filename, 'r')
    for line in f:
        e = cat.get_event(line.strip())
        elist.append(e)
        
    f.close()
    return elist

def dump_kml(objects, filename):
    station_template = '''
  <Placemark>
    <name>%(network)s.%(station)s.%(location)s</name>
    <description></description>
    <styleUrl>#msn_S</styleUrl>
    <Point>
      <coordinates>%(lon)f,%(lat)f,%(elevation)f</coordinates>
    </Point>
  </Placemark>
'''

    f = open(filename, 'w')
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
    f.write('<Document>\n')
    f.write(''' <Style id="sh_S">
                <IconStyle>
                        <scale>1.3</scale>
                        <Icon>
                                <href>http://maps.google.com/mapfiles/kml/paddle/S.png</href>
                        </Icon>
                        <hotSpot x="32" y="1" xunits="pixels" yunits="pixels"/>
                </IconStyle>
                <ListStyle>
                        <ItemIcon>
                                <href>http://maps.google.com/mapfiles/kml/paddle/S-lv.png</href>
                        </ItemIcon>
                </ListStyle>
        </Style>
        <Style id="sn_S">
                <IconStyle>
                        <scale>1.1</scale>
                        <Icon>
                                <href>http://maps.google.com/mapfiles/kml/paddle/S.png</href>
                        </Icon>
                        <hotSpot x="32" y="1" xunits="pixels" yunits="pixels"/>
                </IconStyle>
                <ListStyle>
                        <ItemIcon>
                                <href>http://maps.google.com/mapfiles/kml/paddle/S-lv.png</href>
                        </ItemIcon>
                </ListStyle>
        </Style>
        <StyleMap id="msn_S">
                <Pair>
                        <key>normal</key>
                        <styleUrl>#sn_S</styleUrl>
                </Pair>
                <Pair>
                        <key>highlight</key>
                        <styleUrl>#sh_S</styleUrl>
                </Pair>
        </StyleMap>
''')
    for obj in objects:

        if isinstance(obj, Station):
             f.write(station_template % obj.__dict__) 
    f.write('</Document>')
    f.write('</kml>\n')
    f.close()



            



