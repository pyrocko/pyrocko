import model
from moment_tensor import MomentTensor
import urllib2
import time
import calendar
import re
import logging

import numpy as num

from xml.dom import minidom
from xml.parsers.expat import ExpatError 

logger = logging.getLogger('pyrocko.catalog')

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
    def get_event(self, name):
        raise Exception('This method should be implemented in derived class.')
    
    def iter_event_names(self, time_range, **kwargs):
        raise Exception('This method should be implemented in derived class.')

    def get_event_names(self, time_range, **kwargs):
        return list(self.iter_event_names(time_range, **kwargs))
    
    def get_events(self, time_range, **kwargs):
        return list(self.iter_events(time_range, **kwargs))

    def iter_events(self, time_range, **kwargs):
        events = []
        for name in self.iter_event_names(time_range, **kwargs):
            yield self.get_event(name)

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
    
    def iter_event_names(self, time_range=None, nmax=10000, magmin=None, latmin=-90., latmax=90., lonmin=-180., lonmax=180.):
        dmin = time.strftime('%Y-%m-%d', time.gmtime(time_range[0]))
        dmax = time.strftime('%Y-%m-%d', time.gmtime(time_range[1]+24*60*60))
        
        if magmin is None:
            magmin = ''
        else:
            magmin = '%g' % magmin
       
        url = ('http://geofon.gfz-potsdam.de/db/eqinfo.php?' + '&'.join([
            'datemin=%s' % dmin,
            'datemax=%s' % dmax,
            'latmin=%g' % latmin,
            'latmax=%g' % latmax,
            'lonmin=%g' % lonmin,
            'lonmax=%g' % lonmax,
            'magmin=%s' % magmin,
            'fmt=html',
            'nmax=%i' % nmax]))
            
        page = urllib2.urlopen(url).read()
        events = self._parse_events_page(page)
        for ev in events:
            if time_range[0] <= ev.time and ev.time <= time_range[1]:
                self.events[ev.name] = ev
                yield ev.name

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
              region=d['region'],
              catalog='GEOFON')
              
        return ev

        
    def _parse_events_page(self, page):
        page = re.sub('&nbsp([^;])', '&nbsp;\\1', page)  # fix broken &nbsp; tags
        page = re.sub('border=0', 'border="0"', page)
        try:
            doc = minidom.parseString(page)
        except ExpatError, e:
            lines = page.splitlines()
            r = max(e.lineno - 1 - 2,0), min(e.lineno - 1 +3, len(lines))
            ilineline = zip( range(r[0]+1,r[1]+1), lines[r[0]:r[1]] )
            
            logger.error('A problem occured while parsing HTML from GEOFON page (line=%i, col=%i):\n\n' % (e.lineno, e.offset) +
            '\n'.join( ['  line %i: %s' % (iline, line[:e.offset] + '### HERE ###' + line[e.offset:]) for (iline, line) in ilineline ] ))
            logger.error('... maybe the format of the GEOFON web catalog has changed.')
            raise
        
        events = []
        for tr in doc.getElementsByTagName("tr"):
            tds = tr.getElementsByTagName("td")
            if len(tds) == 9:
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
                region = vals[7]
                ev = model.Event(
                    lat=epicenter[0],
                    lon=epicenter[1], 
                    time=tevent,
                    name=eid,
                    depth=depth,
                    magnitude=mag,
                    region=region,
                    catalog='GEOFON')
                
                logger.debug('Adding event from GEOFON catalog: %s' % ev)
                
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

class Anon:
    pass

class GlobalCMT(EarthquakeCatalog):
    
    def __init__(self):
        self.events = {}
    
    def iter_event_names(self, time_range=None, magmin=0., magmax=10., latmin=-90., latmax=90., lonmin=-180., lonmax=180.):
         
        yearbeg, monbeg, daybeg = time.gmtime(time_range[0])[:3]
        yearend, monend, dayend = time.gmtime(time_range[1])[:3]
        
        url = 'http://www.globalcmt.org/cgi-bin/globalcmt-cgi-bin/CMT4/form?' + '&'.join( [
            'itype=ymd', 'yr=%i' % yearbeg, 'mo=%i' % monbeg, 'day=%i' % daybeg,
            'otype=ymd', 'oyr=%i' % yearend, 'omo=%i' % monend, 'oday=%i' % dayend,
            'jyr=1976', 'jday=1', 'ojyr=1976', 'ojday=1', 'nday=1',
            'lmw=%g' % magmin, 'umw=%g' % magmax,
            'lms=0', 'ums=10',
            'lmb=0', 'umb=10',
            'llat=%g' % latmin, 'ulat=%g' % latmax,
            'llon=%g' % lonmin, 'ulon=%g' % lonmax,
            'lhd=0', 'uhd=1000',
            'lts=-9999', 'uts=9999',
            'lpe1=0', 'upe1=90',
            'lpe2=0', 'upe2=90',
            'list=5' ])

        while True:
            page = urllib2.urlopen(url).read()
            events, more = self._parse_events_page(page)

            for ev in events:
                if time_range[0] <= ev.time and ev.time <= time_range[1]:
                    self.events[ev.name] = ev
                    yield ev.name
            if more:
                url = more
            else:
                break

    def get_event(self, name):
        return self.events[name]

    def _parse_events_page(self, page):

        lines = page.splitlines()
        state = 0

        events = []

        def complete(data):
            try:
                t = calendar.timegm((data.year, data.month, data.day, data.hour, data.minute, data.seconds))
                m = num.array([data.mrr, data.mrt, data.mrp, 
                               data.mrt, data.mtt, data.mtp,
                               data.mrp, data.mtp, data.mpp],
                        dtype=num.float).reshape(3,3)

                m *= 10**(data.exponent-7)
                mt = MomentTensor(m_up_south_east=m)
                ev = model.Event(
                    lat=data.lat,
                    lon=data.lon, 
                    time=t,
                    name=data.eventname,
                    depth=data.depth_km*1000.,
                    magnitude=mt.moment_magnitude(),
                    duration=data.half_duration * 2.,
                    region=data.region,
                    catalog=data.catalog)

                ev.moment_tensor = mt
                events.append(ev)

            except AttributeError:
                pass

        catalog = 'Global-CMT'

        data = None
        more = None
        for line in lines:
            if state == 0:

                m = re.search(r'<a href="([^"]+)">More solutions', line) 
                if m:
                    more = m.group(1)
                
                m = re.search(r'From Quick CMT catalog', line)
                if m:
                    catalog = 'Global-CMT-Quick'

                m = re.search(r'Event name:\s+(\S+)', line)
                if m:
                    if data:
                        complete(data)

                    data = Anon()
                    data.eventname = m.group(1)
                    data.catalog = catalog

                if data:
                    m = re.search(r'Region name:\s+([^<]+)', line)
                    if m:
                        data.region = m.group(1)

                    m = re.search(r'Date \(y/m/d\): (\d\d\d\d)/(\d+)/(\d+)', line)
                    if m:
                        data.year, data.month, data.day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            
                    m = re.search(r'Timing and location information', line)
                    if m:
                        state = 1
            
            if state == 1:
                toks = line.split()
                if toks and toks[0] == 'CMT':
                    data.hour, data.minute = [ int(x) for x in toks[1:3] ]
                    data.seconds, data.lat, data.lon, data.depth_km = [ float(x) for x in toks[3:] ]
               
                m = re.search(r'Assumed half duration:\s+(\S+)', line)
                if m:
                    data.half_duration = float(m.group(1))

                m = re.search(r'Mechanism information', line)
                if m:
                    state = 2

            if state == 2:
                m = re.search(r'Exponent for moment tensor:\s+(\d+)', line)
                if m:
                    data.exponent = int(m.group(1))

                toks = line.split()
                if toks and toks[0] == 'CMT':
                    data.mrr, data.mtt, data.mpp, data.mrt, data.mrp, data.mtp = [ float(x) for x in toks[1:] ]
            
                m = re.search(r'^Eigenvector:', line)
                if m:
                    state = 0

        if data is not None:
            complete(data)

        return events, more
        
