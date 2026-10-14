# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Client to get earthquake catalog information from
`USGS <https://earthquake.usgs.gov/>`_.
'''

import time
import json
import logging
from urllib.error import HTTPError

from pyrocko import util, model
from pyrocko.util import urlopen_ssl_check
from .base_catalog import EarthquakeCatalog, NotFound

logger = logging.getLogger('pyrocko.client.usgs')

km = 1000.


class USGS(EarthquakeCatalog):
    '''
    Client to get earthquake catalog information from
    `USGS <https://earthquake.usgs.gov/>`_.
    '''

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

        events = self._parse_events_page(self._query(p))

        for ev in events:
            self.events[ev.name] = ev

        for ev in events:
            if time_range[0] <= ev.time and ev.time <= time_range[1]:
                yield ev.name

    def _query(self, p):
        url = 'https://earthquake.usgs.gov/fdsnws/event/1/query?' + '&'.join(p)

        logger.debug('Opening URL: %s' % url)
        try:
            page = urlopen_ssl_check(url).read()
        except HTTPError as e:
            if e.code == 404:
                raise NotFound(url)

            raise

        logger.debug('Received page (%i bytes)' % len(page))
        return json.loads(page.decode('utf-8'))

    def _parse_events_page(self, doc):
        return [self._feature_to_event(feat) for feat in doc['features']]

    def _feature_to_event(self, feat):
        props = feat['properties']
        geo = feat['geometry']
        lon, lat, depth = [float(x) for x in geo['coordinates']]
        t = util.str_to_time('1970-01-01 00:00:00') + props['time'] * 0.001

        if props['mag'] is not None:
            mag = float(props['mag'])
        else:
            mag = None

        if props['place'] is not None:
            region = props['place'].encode('ascii', 'replace').decode()
        else:
            region = None

        extras = {}
        if props.get('updated') is not None:
            extras['catalog_time_modified'] = props['updated'] * 0.001

        return model.Event(
            lat=lat,
            lon=lon,
            time=t,
            name=str(feat['id']),
            depth=depth*1000.,
            magnitude=mag,
            magnitude_type=props.get('magType') or None,
            region=region,
            catalog=str(props['net'].upper()),
            extras=extras)

    def get_event(self, name):
        '''
        Get event by USGS event ID.

        Any of the IDs associated with an event can be used. The returned
        event has the preferred ID as its name.
        '''

        if name not in self.events:
            ev = self._feature_to_event(
                self._query(['format=geojson', 'eventid=%s' % name]))

            self.events[ev.name] = ev
            self.events[name] = ev

        return self.events[name]
