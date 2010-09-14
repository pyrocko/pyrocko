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
        
class EarthquakeCatalog:
    
    def get_event_names(self, time_range = None):
        raise Exception('This method should be implemented in derived class.')
    
    def get_event(self, name):
        raise Exception('This method should be implemented in derived class.')
    
    def get_events(self, name):
        raise Exception('This method should be implemented in derived class.')

def parse_id_from_link(s):
    pass

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
        
    
class Geofon(EarthquakeCatalog):
    
    def __init__(self):
        self.events = {}
    
    def get_event_names(self, time_range=None, nmax=10000):
        dmin = time.strftime('%Y-%m-%d', time.gmtime(time_range[0]))
        dmax = time.strftime('%Y-%m-%d', time.gmtime(time_range[1]+24*60*60))
        
        url = ('http://geofon.gfz-potsdam.de/db/eqinfo.php?' + '&'.join([
            'datemin=%s' % dmin,
            'datemax=%s' % dmax,
            'latmin=-90',
            'latmax=%2B90',
            'lonmin=-180',
            'lonmax=%2B180',
            'magmin=',
            'fmt=html',
            'nmax=%i' % nmax]))
            
        page = urllib2.urlopen(url).read()
        events = self._parse_events_page(page)
        for ev in events:
            if time_range[0] <= ev.time and ev.time <= time_range[1]:
                self.events[ev.name] = ev
            
        return self.events.keys()
    
    def get_event(self, name):
        if name in self.events:
            return self.events[name]
        
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
        
    def get_events(self, time_range, nmax=10000):
        names = self.get_event_names(time_range, nmax=nmax)
        events = []
        for name in names:
            events.append(self.events[name])
        
        return events
        
    def _parse_events_page(self, page):
        doc = minidom.parseString(page)
        events = []
        for tr in doc.getElementsByTagName("tr"):
            tds = tr.getElementsByTagName("td")
            if len(tds) == 7:
                elinks = tds[0].getElementsByTagName("a")
                if len(elinks) != 1: continue
                if not 'href' in elinks[0].attributes.keys(): continue
                link = elinks[0].attributes['href'].value.encode('ascii')
                m = re.search(r'\?id=(gfz[0-9]+[a-z]+)$', link)
                if not m: continue
                eid = m.group(1)
                vals = [ getTextR(td).encode('ascii') for td in tds ]
                tevent = calendar.timegm(time.strptime(vals[0][:19], '%Y-%m-%d %H:%M:%S'))
                mag = float(vals[1])
                epicenter = parse_location( vals[2]+' '+vals[3] )
                depth = float(vals[4])*1000.
                region = vals[6]
                ev = model.Event(
                    lat=epicenter[0],
                    lon=epicenter[1], 
                    time=tevent,
                    name=eid,
                    depth=depth,
                    magnitude=mag,
                    region=region)
                events.append(ev)
        return events
        
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
        
        
class GlobalCMT(EarthquakeCatalog):
    
    def get_event_names(time_range=None):
        
        ttbeg = time.gmtime(time_range[0])
        ttend = time.gmtime(time_range[1])
        
        
        