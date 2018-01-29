# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
from __future__ import absolute_import, division

try:
    from future.moves.urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen
from builtins import zip
from builtins import range

import time
import calendar
import re
import logging

from pyrocko import model
from pyrocko.moment_tensor import MomentTensor, symmat6
from .base_catalog import EarthquakeCatalog, NotFound

from xml.dom import minidom
from xml.parsers.expat import ExpatError

logger = logging.getLogger('pyrocko.client.geofon')

km = 1000.


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
    '''Access the Geofon earthquake catalog '''
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
            page = urlopen(url).read()
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
            page = urlopen(url).read()
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
        except ExpatError as e:
            lines = page.splitlines()
            r = max(e.lineno - 1 - 2, 0), min(e.lineno - 1 + 3, len(lines))
            ilineline = list(zip(range(r[0]+1, r[1]+1),
                                 lines[r[0]:r[1]]))

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
        page = re.sub(br'&nbsp([^;])', b'&nbsp;\\1', page)
        page = re.sub(br'border=0', b'border="0"', page)
        page = re.sub(br'<(link|meta).*?>', b'', page, flags=re.DOTALL)
        page = re.sub(br'</html>.*', b'</html>', page, flags=re.DOTALL)

        doc = self.parse_xml(page)

        events = []
        for tr in doc.getElementsByTagName("tr"):
            logger.debug('Found <tr> tag')
            tds = tr.getElementsByTagName("td")
            if len(tds) != 8:
                logger.debug('Does not contain 8 <td> tags.')
                continue

            elinks = tds[0].getElementsByTagName("a")
            if len(elinks) != 1 or not elinks[0].getAttribute('href'):
                logger.debug('Could not find link to event details page.')
                continue

            link = elinks[0].getAttribute('href').encode('ascii')
            m = re.search(br'\?id=(gfz[0-9]+[a-z]+)$', link)
            if not m:
                logger.debug('Could not find event id.')
                continue

            eid = m.group(1)
            vals = [getTextR(td) for td in tds]
            tevent = calendar.timegm(
                time.strptime(vals[0][:19], '%Y-%m-%d %H:%M:%S'))
            mag = float(vals[1])
            epicenter = parse_location((vals[2]+' '+vals[3]))
            depth = float(vals[4])*1000.
            region = vals[7]
            ev = model.Event(
                lat=epicenter[0],
                lon=epicenter[1],
                time=tevent,
                name=str(eid.decode('ascii')),
                depth=depth,
                magnitude=mag,
                region=str(region),
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
        page = urlopen(url).read()
        logger.debug('Received page (%i bytes)' % len(page))

        return self._parse_mt_page(page)

    def _parse_mt_page(self, page):
        d = {}
        for k in 'Scale', 'Mrr', 'Mtt', 'Mpp', 'Mrt', 'Mrp', 'Mtp':
            r = k.encode('ascii')+br'\s*=?\s*(\S+)'
            m = re.search(r, page)
            if m:
                s = m.group(1).replace(b'10**', b'1e')
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
        page = re.sub(br'align=center', b'align="center"', page)
        page = re.sub(br'<(link|meta).*?>', b'', page, flags=re.DOTALL)
        page = re.sub(br'</html>.*', b'</html>', page, flags=re.DOTALL)
        page = re.sub(br'"[^"]+geohack[^"]+"', b'""', page)

        doc = self.parse_xml(page)

        d = {}
        for tr in doc.getElementsByTagName("tr"):
            tds = tr.getElementsByTagName("td")
            if len(tds) >= 2:
                s = getTextR(tds[0]).strip()
                t = s.split()
                if t:
                    k = t[-1].rstrip(':').lower()
                    v = getTextR(tds[1])
                    logger.debug('%s => %s' % (k, v))
                    if k in wanted_map:
                        d[k] = wanted_map[k](v)

        d['have_moment_tensor'] = page.find(b'Moment tensor solution') != -1

        for k in wanted_map:
            if k not in d:
                raise NotFound()

        return d
