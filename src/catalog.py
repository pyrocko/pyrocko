from pyrocko import model, util
from pyrocko.moment_tensor import MomentTensor, symmat6
import urllib2
import urllib
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
        for name in self.iter_event_names(time_range, **kwargs):
            yield self.get_event(name)


def parse_location(s):
    direction = {'N': 1, 'S': -1, 'E': 1, 'W': -1}
    m = re.match(r'([0-9.]+)\s*([NSEW])\s+([0-9.]+)\s*([NSEW])', s)
    if m:
        a = float(m.group(1)) * direction[m.group(2)]
        b = float(m.group(3)) * direction[m.group(4)]
        if m.group(2) in 'NS' and m.group(4) in 'EW':
            return a, b
        if m.group(2) in 'EW' and m.group(4) in 'NS':
            return b, a


def parse_km(s):
    m = re.match(r'([0-9.]+)\s*(km)', s)
    if m:
        a = float(m.group(1))*1000.
        return a


class Geofon(EarthquakeCatalog):

    def __init__(self):
        self.events = {}

    def flush(self):
        self.events = {}

    def iter_event_names(
            self,
            time_range=None,
            nmax=1000,
            magmin=None,
            latmin=-90.,
            latmax=90.,
            lonmin=-180.,
            lonmax=180.):

        logger.debug('In Geofon.iter_event_names(...)')

        dmin = time.strftime('%Y-%m-%d', time.gmtime(time_range[0]))
        dmax = time.strftime('%Y-%m-%d', time.gmtime(time_range[1]+24*60*60))

        if magmin is None:
            magmin = ''
        else:
            magmin = '%g' % magmin

        ipage = 1
        while True:
            url = ('http://geofon.gfz-potsdam.de/eqinfo/list.php?' + '&'.join([
                'page=%i' % ipage,
                'datemin=%s' % dmin,
                'datemax=%s' % dmax,
                'latmin=%g' % latmin,
                'latmax=%g' % latmax,
                'lonmin=%g' % lonmin,
                'lonmax=%g' % lonmax,
                'magmin=%s' % magmin,
                'fmt=html',
                'nmax=%i' % nmax]))

            logger.debug('Opening URL: %s' % url)
            page = urllib2.urlopen(url).read()
            logger.debug('Received page (%i bytes)' % len(page))
            events = self._parse_events_page(page)

            if not events:
                break

            for ev in events:
                if time_range[0] <= ev.time and ev.time <= time_range[1]:
                    self.events[ev.name] = ev
                    yield ev.name

            ipage += 1

    def get_event(self, name):
        logger.debug('In Geofon.get_event("%s")' % name)

        if name not in self.events:
            url = 'http://geofon.gfz-potsdam.de/eqinfo/event.php?id=%s' % name
            logger.debug('Opening URL: %s' % url)
            page = urllib2.urlopen(url).read()
            logger.debug('Received page (%i bytes)' % len(page))

            try:
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

                if d['have_moment_tensor']:
                    ev.moment_tensor = True

                self.events[name] = ev

            except NotFound:
                raise NotFound(url)  # reraise with url

        ev = self.events[name]

        if ev.moment_tensor is True:
            ev.moment_tensor = self.get_mt(ev)

        return ev

    def parse_xml(self, page):
        try:
            return minidom.parseString(page)
        except ExpatError, e:
            lines = page.splitlines()
            r = max(e.lineno - 1 - 2, 0), min(e.lineno - 1 + 3, len(lines))
            ilineline = zip(range(r[0]+1, r[1]+1), lines[r[0]:r[1]])

            logger.error(
                'A problem occured while parsing HTML from GEOFON page '
                '(line=%i, col=%i):\n\n' % (e.lineno, e.offset) +
                '\n'.join([
                    '  line %i: %s' % (iline, line)
                    for (iline, line) in ilineline]))
            logger.error(
                '... maybe the format of the GEOFON web catalog has changed.')
            raise

    def _parse_events_page(self, page):
        logger.debug('In Geofon._parse_events_page(...)')
        # fix broken &nbsp; tags
        page = re.sub('&nbsp([^;])', '&nbsp;\\1', page)
        page = re.sub('border=0', 'border="0"', page)
        page = re.sub(r'</html>.*', '</html>', page, flags=re.DOTALL)

        doc = self.parse_xml(page)

        events = []
        for tr in doc.getElementsByTagName("tr"):
            logger.debug('Found <tr> tag')
            tds = tr.getElementsByTagName("td")
            if len(tds) != 8:
                logger.debug('Does not contain 8 <td> tags.')
                continue

            elinks = tds[0].getElementsByTagName("a")
            if len(elinks) != 1 or ('href' not in elinks[0].attributes.keys()):
                logger.debug('Could not find link to event details page.')
                continue

            link = elinks[0].attributes['href'].value.encode('ascii')
            m = re.search(r'\?id=(gfz[0-9]+[a-z]+)$', link)
            if not m:
                logger.debug('Could not find event id.')
                continue

            eid = m.group(1)
            vals = [getTextR(td).encode('ascii') for td in tds]
            tevent = calendar.timegm(
                time.strptime(vals[0][:19], '%Y-%m-%d %H:%M:%S'))
            mag = float(vals[1])
            epicenter = parse_location(vals[2]+' '+vals[3])
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

            if vals[6] == 'MT':
                ev.moment_tensor = True

            logger.debug('Adding event from GEOFON catalog: %s' % ev)

            events.append(ev)

        return events

    def get_mt(self, ev):
        syear = time.strftime('%Y', time.gmtime(ev.time))
        url = 'http://geofon.gfz-potsdam.de/data/alerts/%s/%s/mt.txt' % (
            syear, ev.name)
        logger.debug('Opening URL: %s' % url)
        page = urllib2.urlopen(url).read()
        logger.debug('Received page (%i bytes)' % len(page))

        return self._parse_mt_page(page)

    def _parse_mt_page(self, page):
        d = {}
        for k in 'Scale', 'Mrr', 'Mtt', 'Mpp', 'Mrt', 'Mrp', 'Mtp':
            r = k+r'\s*=?\s*(\S+)'
            m = re.search(r, page)
            if m:
                s = m.group(1).replace('10**', '1e')
                d[k.lower()] = float(s)

        m = symmat6(*(d[x] for x in 'mrr mtt mpp mrt mrp mtp'.split()))
        m *= d['scale']
        mt = MomentTensor(m_up_south_east=m)

        return mt

    def _parse_event_page(self, page):
        logger.debug('In Geofon._parse_event_page(...)')

        wanted_map = {
            'region': lambda v: v,
            'time': lambda v: calendar.timegm(
                time.strptime(v[:19], '%Y-%m-%d %H:%M:%S')),
            'magnitude': lambda v: float(v.split()[0]),
            'epicenter': parse_location,
            'depth': parse_km,
        }

        # fix broken tag
        page = re.sub('align=center', 'align="center"', page)
        page = re.sub(r'</html>.*', '</html>', page, flags=re.DOTALL)

        doc = self.parse_xml(page)

        d = {}
        for tr in doc.getElementsByTagName("tr"):
            tds = tr.getElementsByTagName("td")
            if len(tds) >= 2:
                s = getTextR(tds[0]).strip()
                t = s.split()
                if t:
                    k = t[-1].rstrip(':').lower()
                    v = getTextR(tds[1]).encode('ascii')
                    logger.debug('%s => %s' % (k, v))
                    if k in wanted_map.keys():
                        d[k] = wanted_map[k](v)

        d['have_moment_tensor'] = page.find('Moment tensor solution') != -1

        for k in wanted_map:
            if k not in d:
                raise NotFound()

        return d


class Anon:
    pass


class GlobalCMT(EarthquakeCatalog):

    def __init__(self):
        self.events = {}

    def flush(self):
        self.events = {}

    def iter_event_names(
            self,
            time_range=None,
            magmin=0.,
            magmax=10.,
            latmin=-90.,
            latmax=90.,
            lonmin=-180.,
            lonmax=180.):

        yearbeg, monbeg, daybeg = time.gmtime(time_range[0])[:3]
        yearend, monend, dayend = time.gmtime(time_range[1])[:3]

        url = 'http://www.globalcmt.org/cgi-bin/globalcmt-cgi-bin/CMT4/form?' \
            + '&'.join([
                'itype=ymd',
                'yr=%i' % yearbeg, 'mo=%i' % monbeg, 'day=%i' % daybeg,
                'otype=ymd',
                'oyr=%i' % yearend, 'omo=%i' % monend, 'oday=%i' % dayend,
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
                'list=5'])

        while True:
            logger.debug('Opening URL: %s' % url)
            page = urllib2.urlopen(url).read()
            logger.debug('Received page (%i bytes)' % len(page))

            events, more = self._parse_events_page(page)

            for ev in events:
                self.events[ev.name] = ev

            for ev in events:
                if time_range[0] <= ev.time and ev.time <= time_range[1]:
                    yield ev.name

            if more:
                url = more
            else:
                break

    def get_event(self, name):
        if name not in self.events:
            t = self._name_to_date(name)
            for name2 in self.iter_event_names(
                    time_range=(t-24*60*60, t+2*24*60*60)):

                if name2 == name:
                    break

        return self.events[name]

    def _parse_events_page(self, page):

        lines = page.splitlines()
        state = 0

        events = []

        def complete(data):
            try:
                t = calendar.timegm((
                    data.year, data.month, data.day,
                    data.hour, data.minute, data.seconds))

                m = num.array(
                    [data.mrr, data.mrt, data.mrp,
                     data.mrt, data.mtt, data.mtp,
                     data.mrp, data.mtp, data.mpp],
                    dtype=num.float).reshape(3, 3)

                m *= 10.0**(data.exponent-7)
                mt = MomentTensor(m_up_south_east=m)
                ev = model.Event(
                    lat=data.lat,
                    lon=data.lon,
                    time=t,
                    name=data.eventname,
                    depth=data.depth_km*1000.,
                    magnitude=float(mt.moment_magnitude()),
                    duration=data.half_duration * 2.,
                    region=data.region.rstrip(),
                    catalog=data.catalog)

                ev.moment_tensor = mt
                events.append(ev)

            except AttributeError:
                pass

        catalog = 'gCMT'

        data = None
        more = None
        for line in lines:
            if state == 0:

                m = re.search(r'<a href="([^"]+)">More solutions', line)
                if m:
                    more = m.group(1)

                m = re.search(r'From Quick CMT catalog', line)
                if m:
                    catalog = 'gCMT-Q'

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

                    m = re.search(
                        r'Date \(y/m/d\): (\d\d\d\d)/(\d+)/(\d+)', line)

                    if m:
                        data.year, data.month, data.day = (
                            int(m.group(1)), int(m.group(2)), int(m.group(3)))

                    m = re.search(r'Timing and location information', line)
                    if m:
                        state = 1

            if state == 1:
                toks = line.split()
                if toks and toks[0] == 'CMT':
                    data.hour, data.minute = [int(x) for x in toks[1:3]]
                    data.seconds, data.lat, data.lon, data.depth_km = [
                        float(x) for x in toks[3:]]

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
                    data.mrr, data.mtt, data.mpp, \
                        data.mrt, data.mrp, data.mtp = [
                            float(x) for x in toks[1:]]

                m = re.search(r'^Eigenvector:', line)
                if m:
                    state = 0

        if data is not None:
            complete(data)

        return events, more

    def _name_to_date(self, name):

        if len(name) == 7:
            y, m, d = time.strptime(name[:6], '%m%d%y')[:3]
            if y > 2005:
                y -= 100

        else:
            y, m, d = time.strptime(name[:8], '%Y%m%d')[:3]

        t = calendar.timegm((y, m, d, 0, 0, 0))
        return t


class USGS(EarthquakeCatalog):

    def __init__(self, catalog=None):
        self.catalog = catalog
        self.events = {}

    def flush(self):
        self.events = {}

    def iter_event_names(
            self,
            time_range=None,
            magmin=0.,
            magmax=10.,
            latmin=-90.,
            latmax=90.,
            lonmin=-180.,
            lonmax=180.):

        yearbeg, monbeg, daybeg = time.gmtime(time_range[0])[:3]
        yearend, monend, dayend = time.gmtime(time_range[1])[:3]

        p = []
        a = p.append
        a('format=geojson')
        if self.catalog is not None:
            a('catalog=%s' % self.catalog.lower())

        a('starttime=%s' % util.time_to_str(
            time_range[0], format='%Y-%m-%dT%H:%M:%S'))

        a('endtime=%s' % util.time_to_str(
            time_range[1], format='%Y-%m-%dT%H:%M:%S'))

        if latmin != -90.:
            a('minlatitude=%g' % latmin)
        if latmax != 90.:
            a('maxlatitude=%g' % latmax)
        if lonmin != -180.:
            a('minlongitude=%g' % lonmin)
        if lonmax != 180.:
            a('maxlongitude=%g' % lonmax)
        if magmin != 0.:
            a('minmagnitude=%g' % magmin)
        if magmax != 10.:
            a('maxmagnitude=%g' % magmax)

        url = 'http://earthquake.usgs.gov/fdsnws/event/1/query?' + '&'.join(p)

        logger.debug('Opening URL: %s' % url)
        page = urllib2.urlopen(url).read()
        logger.debug('Received page (%i bytes)' % len(page))

        events = self._parse_events_page(page)

        for ev in events:
            self.events[ev.name] = ev

        for ev in events:
            if time_range[0] <= ev.time and ev.time <= time_range[1]:
                yield ev.name

    def _parse_events_page(self, page):

        import json
        doc = json.loads(page)

        events = []
        for feat in doc['features']:
            props = feat['properties']
            geo = feat['geometry']
            lon, lat, depth = [float(x) for x in geo['coordinates']]
            t = util.str_to_time('1970-01-01 00:00:00') + \
                props['time'] * 0.001

            if props['mag'] is not None:
                mag = float(props['mag'])
            else:
                mag = None

            if props['place'] is not None:
                region = props['place'].encode('ascii', 'replace')
            else:
                region = None

            catalog = str(props['net'].upper())
            name = 'USGS-%s-' % catalog + util.time_to_str(
                t, format='%Y-%m-%d_%H-%M-%S.3FRAC')

            ev = model.Event(
                lat=lat,
                lon=lon,
                time=t,
                name=name,
                depth=depth*1000.,
                magnitude=mag,
                region=region,
                catalog=catalog)

            events.append(ev)

        return events

    def get_event(self, name):
        if name not in self.events:
            t = self._name_to_date(name)
            for name2 in self.iter_event_names(
                    time_range=(t-24*60*60, t+24*60*60)):

                if name2 == name:
                    break

        return self.events[name]

    def _name_to_date(self, name):
        ds = name[-23:]
        t = util.str_to_time(ds, format='%Y-%m-%d_%H-%M-%S.3FRAC')
        return t


class NotFound(Exception):
    def __init__(self, url=None):
        Exception.__init__(self)
        self._url = url

    def __str__(self):
        if self._url:
            return 'No results for request %s' % self._url
        else:
            return 'No results for request'


def ws_request(url, post=False, **kwargs):
    url_values = urllib.urlencode(kwargs)
    url = url + '?' + url_values
    logger.debug('Accessing URL %s' % url)

    req = urllib2.Request(url)
    if post:
        req.add_data(post)

    req.add_header('Accept', '*/*')

    try:
        return urllib2.urlopen(req)

    except urllib2.HTTPError, e:
        if e.code == 404:
            raise NotFound(url)
        else:
            raise e


class Kinherd(EarthquakeCatalog):

    def __init__(self, catalog='KPS'):
        self.catalog = catalog
        self.events = {}

    def retrieve(self, **kwargs):
        import yaml

        kwargs['format'] = 'yaml'

        url = 'http://kinherd.org/quakes/%s' % self.catalog

        f = ws_request(url, **kwargs)

        names = []
        for eq in yaml.safe_load_all(f):
            pset = eq['parametersets'][0]
            tref = calendar.timegm(pset['reference_time'].timetuple())
            tpost = calendar.timegm(pset['posted_time'].timetuple())
            params = pset['parameters']

            mt = MomentTensor(
                strike=params['strike'],
                dip=params['dip'],
                rake=params['slip_rake'],
                scalar_moment=params['moment'])

            event = model.Event(
                time=tref + params['time'],
                lat=params['latitude'],
                lon=params['longitude'],
                depth=params['depth'],
                magnitude=params['magnitude'],
                duration=params['rise_time'],
                name=eq['name'],
                catalog=self.catalog,
                moment_tensor=mt)

            event.ext_confidence_intervals = {}
            trans = {'latitude': 'lat', 'longitude': 'lon'}
            for par in 'latitude longitude depth magnitude'.split():
                event.ext_confidence_intervals[trans.get(par, par)] = \
                    (params[par+'_ci_low'], params[par+'_ci_high'])

            event.ext_posted_time = tpost

            name = eq['name']
            self.events[name] = event
            names.append(name)

        return names

    def iter_event_names(self, time_range=None, **kwargs):

        qkwargs = {}
        for k in 'magmin magmax latmin latmax lonmin lonmax'.split():
            if k in kwargs and kwargs[k] is not None:
                qkwargs[k] = '%f' % kwargs[k]

        if time_range is not None:
            form = '%Y-%m-%d_%H-%M-%S'
            if time_range[0] is not None:
                qkwargs['tmin'] = util.time_to_str(time_range[0], form)
            if time_range[1] is not None:
                qkwargs['tmax'] = util.time_to_str(time_range[1], form)

        for name in self.retrieve(**qkwargs):
            yield name

    def get_event(self, name):
        if name not in self.events:
            self.retrieve(name=name)

        return self.events[name]


class Sachsen(EarthquakeCatalog):

    def __init__(self):
        self._events = None

    def retrieve(self):
        url = 'http://home.uni-leipzig.de/collm/auswertung_temp.html'

        f = ws_request(url)
        text = f.read()
        sec = 0
        events = {}
        for line in text.splitlines():
            line = line.strip()
            if line == '<PRE>':
                sec += 1
                continue

            if sec == 1 and not line:
                sec += 1
                continue

            if sec == 1:
                t = line.split(' ', 1)
                name = t[0]
                sdate = t[1][0:11]
                stime = t[1][12:22]
                sloc = t[1][23:36]
                sdepth = t[1][37:42]
                smag = t[1][51:55]
                region = t[1][60:]

                sday, smon, syear = sdate.split('-')
                smon = {
                    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                    'Mai': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                    'Sep': '09', 'Okt': '10', 'Nov': '11', 'Dez': '12'}[smon]

                time = util.str_to_time(
                    '%s-%s-%s %s' % (syear, smon, sday, stime))

                slat, slon = sloc.split(';')

                ev = model.Event(
                    time=time,
                    lat=float(slat),
                    lon=float(slon),
                    depth=float(sdepth) * 1000.,
                    magnitude=float(smag),
                    magnitude_type='Ml',
                    name=name,
                    region=region,
                    catalog='Sachsen')

                events[name] = ev

        self._events = events

    def iter_event_names(self, time_range=None, **kwargs):
        if self._events is None:
            self.retrieve()

        for name in sorted(self._events.keys()):
            if time_range is not None:
                ev = self._events[name]
                if time_range[0] <= ev.time and ev.time <= time_range[1]:
                    yield name
            else:
                yield name

    def get_event(self, name):
        if self._events is None:
            self.retrieve()

        return self._events[name]
