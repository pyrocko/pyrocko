import urllib, urllib2, logging, re
from xml.dom import minidom
import util, model, pz

logger = logging.getLogger('pyrocko.iris_ws')

base_url = 'http://www.iris.edu/ws'

class NotTextNode(Exception):
    pass

def getText(node):
    l = []
    for child in node.childNodes:
        if child.nodeType == node.TEXT_NODE:
            l.append(child.data)
        else:
            raise NotTextNode

    return ''.join(l)

def get_dict(node):
    d = {}
    dupl = []
    for cn in node.childNodes:
        try:
            k = str(cn.nodeName.lower())
            v = getText(cn)
            if k in d:
                dupl.append(k) 
            else:
                d[k] = v 

        except NotTextNode:
            pass

    for k in dupl:
        if k in d:
            del d[k]
    
    attr = node.attributes
    for k in attr.keys():
        d[str(k)] = getText(attr[k])

    return d

def cycle(node, name):
    for child in node.getElementsByTagName(name):
        child.D = util.Anon(**get_dict(child))
        yield child

def tear(node, path, _stack=()):
    if len(path) == 0:
        yield _stack
    else:
        for element in cycle(node, path[0]):
            for x in tear(element, path[1:], _stack + ( element, )):
                yield x

def tdate(s):
    return util.str_to_time(s, '%Y-%m-%d')

def sdate(t):
    return util.time_to_str(t, '%Y-%m-%d')

def tdatetime(s):
    return util.str_to_time(s, format='%Y-%m-%dT%H:%M:%S')

def sdatetime(t):
    return util.time_to_str(t, format='%Y-%m-%dT%H:%M:%S')

class NotFound(Exception):
    def __init__(self, url):
        Exception.__init__(self)
        self._url = url
    
    def __str__(self):
        return 'No results for request %s' % self._url

def ws_request(url, post=False, **kwargs):
    url_values = urllib.urlencode(kwargs)
    url = url + '?' + url_values
    logger.debug('Accessing URL %s' % url)

    req = urllib2.Request(url)
    if post:
        req.add_data(post)

    req.add_header('Accept', '*/*')

    try:
        return urllib2.urlopen(req).read()

    except urllib2.HTTPError, e:
        if e.code == 404:
            raise NotFound(url)
        else:
            raise e 

def ws_station( **kwargs ):
    
    for k in 'startbefore', 'startafter', 'endbefore', 'endafter':
        if k in kwargs:
            kwargs[k] = sdate(kwargs[k])

    if 'timewindow' in kwargs:
        tmin, tmax = kwargs.pop('timewindow')
        kwargs['startbefore'] = sdate(tmin)
        kwargs['endafter'] = sdate(tmax)

    return ws_request(base_url + '/station/query', **kwargs)

def ws_virtualnetwork( **kwargs ):
    
    for k in 'starttime', 'endtime':
        if k in kwargs:
            kwargs[k] = sdate(kwargs[k])

    if 'timewindow' in kwargs:
        tmin, tmax = kwargs.pop('timewindow')
        kwargs['starttime'] = sdate(tmin)
        kwargs['endtime'] = sdate(tmax)

    return ws_request(base_url + '/virtualnetwork/query', **kwargs)

def ws_bulkdataselect( selection, quality=None, minimumlength=None, longestonly=False ):

    l = []
    if quality is not None:
        l.append('quality %s' % quality)
    if minimumlength is not None:
        l.append('minimumlength %s' % minimumlength)
    if longestonly:
        l.append('longestonly')

    for (network, station, location, channel, tmin, tmax) in selection:
        if location == '':
            location = '--'
        
        l.append(' '.join((network, station, location, channel, sdatetime(tmin), sdatetime(tmax))))
    
    return ws_request(base_url + '/bulkdataselect/query', post='\n'.join(l))

def ws_sacpz(network=None, station=None, location=None, channel=None, tmin=None, tmax=None):
    d = {}
    if network:
        d['network'] = network
    if station:
        d['station'] = station
    if location:
        d['location'] = location
    else:
        d['location'] = '--'
    if channel:
        d['channel'] = channel
    
    times = (tmin, tmax)
    if len(times) == 2:
        d['starttime'] = sdatetime(min(times))
        d['endtime'] = sdatetime(max(times))
    elif len(times) == 1:
        d['time'] = sdatetime(times[0])
    
    return ws_request(base_url + '/sacpz/query', **d)

class ChannelInfo:
    def __init__(self, network, station, location, channel, start, end, azimuth, dip, elevation, depth, latitude, longitude, sample, input, output, zpk):
        self.network = network
        self.station = station
        self.location = location
        self.channel = channel
        self.start = start
        self.end = end
        self.azimuth = azimuth
        self.dip = dip
        self.elevation = elevation
        self.depth = depth
        self.latitude = latitude
        self.longitude = longitude
        self.sample = sample
        self.input = input
        self.output = output
        self.zpk = zpk

    def __str__(self):
        return '%s.%s.%s.%s' % (self.network, self.station, self.location, self.channel)

def nslc(x):
    return x.network, x.station, x.location, x.channel

def grok_sacpz(data):
    pzlines = []
    d = {}
    responses = []
    float_keys = ('latitude', 'longitude', 'elevation', 'depth', 'dip', 'azimuth', 'sample')
    string_keys = ('input', 'output', 'network', 'station', 'location', 'channel')
    time_keys = ('start', 'end')
    for line in data.splitlines():
        line = line.strip()
        if line.startswith('*'):
            if pzlines:
                if any(pzlines):
                    d['zpk'] = pz.read_sac_zpk(string='\n'.join(pzlines))
                    responses.append(d)
                d = {}
                pzlines = []

            m = re.match(r'^\* ([A-Z]+)[^:]*:(.*)$', line)
            if m:
                k,v = m.group(1).lower(), m.group(2).strip()
                if k in d:
                    assert False, 'duplicate entry? %s' % k

                if k in float_keys:
                    d[k] = float(v)
                elif k in string_keys:
                    d[k] = v
                elif k in time_keys:
                    d[k] = tdatetime(v)

        else:
            pzlines.append(line)

    if pzlines and any(pzlines):
        d['zpk'] = pz.read_sac_zpk(string='\n'.join(pzlines))
        responses.append(d)

    cis = {}
    for kwargs in responses:
        try:
            for k in float_keys + string_keys + time_keys:
                if k not in kwargs:
                    logger.error('Missing entry: %s' % k)
                    raise Exception()

            ci = ChannelInfo(**kwargs)

            cis[nslc(ci)] = ci
            
        except:
            logger.error('Error while parsing SACPZ data')

    return cis

def grok_station_xml( data, tmin, tmax ):
    dom = minidom.parseString(data)
    
    stations = {}
    station_channels = {}
        
    for (sta, sta_epo, cha, cha_epo) in tear(dom, ('Station', 'StationEpoch', 'Channel', 'Epoch') ):
        sta_beg, sta_end, cha_beg, cha_end = [ tdatetime(x) for x in 
                (sta_epo.D.startdate, sta_epo.D.enddate, cha_epo.D.startdate, cha_epo.D.enddate) ]

        if not (sta_beg <= tmin and tmax <= sta_end and cha_beg <= tmin and tmax <= cha_end):
            continue

        nslc = tuple([ str(x.strip()) for x in 
                (sta.D.net_code, sta.D.sta_code, cha.D.loc_code, cha.D.chan_code) ])
        lat, lon, ele, dep, azi, dip = [ float(x) for x in 
                (cha_epo.D.lat, cha_epo.D.lon, cha_epo.D.elevation, cha_epo.D.depth,
                 cha_epo.D.azimuth, cha_epo.D.dip) ]

        nsl = nslc[:3]
        if nsl not in stations:
            stations[nsl] = model.Station(nsl[0], nsl[1], nsl[2], lat, lon, ele, dep)
        
        stations[nsl].add_channel(model.Channel(nslc[-1], azi, dip))

    return stations.values()

def grok_virtualnet_xml(data):
    net_sta = set() 
    dom = minidom.parseString(data)
    for net, sta in tear(dom, ('network', 'station')):
        net_sta.add(( net.D.code, sta.D.code ))

    return net_sta

def data_selection(stations, tmin, tmax, channel_prio=[[ 'BHZ', 'HHZ' ],
            [ 'BH1', 'BHN', 'HH1', 'HHN' ], ['BH2', 'BHE', 'HH2', 'HHE']]):

    selection = []
    for station in stations:
        wanted = []
        for group in channel_prio:
            gchannels = []
            for channel in station.get_channels():
                if channel.name in group:
                    gchannels.append(channel)
            if gchannels:
                gchannels.sort(lambda a,b: cmp(group.index(a.name), group.index(b.name)) )
                wanted.append(gchannels[0])

        if wanted:
            for channel in wanted:
                selection.append((station.network, station.station, station.location, channel.name, tmin, tmax))

    return selection

