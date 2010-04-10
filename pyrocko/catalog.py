import model
import urllib2
import time
import calendar
import re
from xml.dom import minidom

def getText(nodelist):
    rc = ""
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc = rc + node.data
    return rc

def getTextR(node):
    rc = ''
    for child in node.childNodes:
        if child.nodeType == node.TEXT_NODE:
            rc = rc + child.data
        else:
            rc = rc + getTextR(child)
    return rc
        
class EartquakeCatalog:
    
    def get_event_ids(self, time_range = None):
        raise Exception('This method should be implemented in derived class.')
    
    
    def get_event(self, name):
        raise Exception('This method should be implemented in derived class.')
        
    
    def event_filter(self, event):
        return True
    
    def station_filter(self, station):
        return True


def parse_location(s):
    direction = {'N':1,'S':-1,'E':1,'W':-1}
    m = re.match(r'([0-9.]+)\s*([NSEW])\s+([0-9.]+)\s*([NSEW])', s)
    if m:
        a = float(m.group(1)) * direction[m.group(2)]
        b = float(m.group(3)) * direction[m.group(4)]
        if m.group(2) in 'NS' and m.group(4) in 'EW':
            return a,b
        if m.group(2) in 'EW' and m.group(4) in 'NS':
            return b,a
    
    
def parse_km(s):
    m = re.match(r'([0-9.]+)\s*(km)', s)
    if m:
        a = float(m.group(1))*1000.
        return a
        
    
class Geofon:
    
    def get_event(self, name):
        url = 'http://geofon.gfz-potsdam.de/db/eqpage.php?id=%s' % name
        page = urllib2.urlopen(url).read()
        d = self._parse_event_page(page)
        ev = model.Event(
              lat=d['epicenter'][0],
              lon=d['epicenter'][1], 
              time=d['time'],
              name=name,
              depth=d['depth'],
              magnitude=d['magnitude'],
              region=d['region'])
              
        return ev
        
    def _parse_event_page(self, page):
        
        wanted_map = { 
            'region': lambda v: v,
            'time': lambda v: calendar.timegm(time.strptime(v[:19], '%Y-%m-%d %H:%M:%S')),
            'magnitude': lambda v: float(v),
            'epicenter': parse_location,
            'depth': parse_km,
        }
        
        doc = minidom.parseString(page)
        d = {}
        for tr in doc.getElementsByTagName("tr"):
            tds = tr.getElementsByTagName("td")
            if len(tds) == 2:
                k = getTextR(tds[0]).strip().rstrip(':').lower()
                v = getTextR(tds[1]).encode('ascii')
                if k in wanted_map.keys():
                   d[k] = wanted_map[k](v)
                    
        
        return d
        