import urllib, urllib2
from xml.dom import minidom

base_url = 'http://www.iris.edu/ws'

def ws_get(url, **kwargs):
    url_values = urllib.urlencode(kwargs)
    fullurl = url + '?' + url_values
    print fullurl
    return urllib2.urlopen(fullurl).read()

def station( **kwargs ):
    return minidom.parseString(ws_get(base_url + '/station/query', **kwargs))





