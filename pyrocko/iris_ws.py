import urllib, urllib2
from xml.dom import minidom
import util, model

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

def ws_request(url, post=False, **kwargs):
    url_values = urllib.urlencode(kwargs)
    if post:
        return urllib2.urlopen(url,data=post).read()
    else:
        return urllib2.urlopen(url+'?'+url_values).read()

def ws_station( **kwargs ):
    
    for k in 'startbefore', 'startafter', 'endbefore', 'endafter':
        if k in kwargs:
            kwargs[k] = sdate(kwargs[k])

    if 'timewindow' in kwargs:
        tmin, tmax = kwargs['timewindow']
        kwargs['timewindow'] = '%s,%s' % (sdate(tmin), sdate(tmax))

    return ws_request(base_url + '/station/query', **kwargs)

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

def grok_station_xml( data, tmin, tmax ):
    dom = minidom.parseString(data)
    
    channel_prio = [ [ 'BHZ', 'HHZ' ], [ 'BH1', 'BHN', 'HH1', 'HHN' ], ['BH2', 'BHE', 'HH2', 'HHE'] ]

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

