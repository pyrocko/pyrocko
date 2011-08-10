import urllib, urllib2
from xml.dom import minidom
import util

base_url = 'http://www.iris.edu/ws'

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

def station( **kwargs ):
    return minidom.parseString(ws_request(base_url + '/station/query', **kwargs))

def bulkdataselect( selection, quality=None, minimumlength=None, longestonly=False ):

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

