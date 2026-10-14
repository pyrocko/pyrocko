# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Client to get earthquake catalog information from
`GEOFON <https://geofon.gfz.de/>`_.
'''

import time
import re
import math
import logging

import requests

from pyrocko.io import quakeml
from pyrocko.client import fdsn
from pyrocko.moment_tensor import MomentTensor, symmat6
from .base_catalog import EarthquakeCatalog, NotFound

logger = logging.getLogger('pyrocko.client.geofon')

km = 1000.


class Geofon(EarthquakeCatalog):
    '''
    Access the GEOFON earthquake catalog.

    Events are obtained from the GEOFON FDSN event web service. They are
    named by their GEOFON event ID (e.g. ``'gfz2010avtm'``). The time of the
    latest revision is available in ``extras['catalog_time_modified']``,
    evaluation mode and status of the preferred origin are given as tags
    ``evaluation_mode:...`` and ``evaluation_status:...``.

    To be consistent with GEOFON's own event listings, the moment magnitude
    of the moment tensor solution is used as event magnitude, if available,
    and the GEOFON event status is given as tag ``geofon_status:X``, where
    ``X`` is one of ``A`` (automatic), ``C`` (confirmed) or ``M`` (manually
    revised).
    '''

    def __init__(self, get_moment_tensors=True):
        self.events = {}
        self._get_moment_tensors = get_moment_tensors
        self._session = requests.Session()

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

        # The service takes whole seconds. Exact time range is applied below.
        params = dict(
            starttime=float(math.floor(time_range[0])),
            endtime=float(math.ceil(time_range[1])),
            minlatitude=latmin,
            maxlatitude=latmax,
            minlongitude=lonmin,
            maxlongitude=lonmax,
            orderby='time-asc',
            limit=nmax)

        if magmin is None:
            queries = [params]
        else:
            # The service filters by preferred magnitude, but we use the
            # moment magnitude, if available. A second query gets the events
            # with large enough moment magnitude but smaller preferred
            # magnitude.
            params['minmagnitude'] = magmin
            queries = [params, dict(params, magnitudetype='Mw')]

        events = {}
        for query_params in queries:
            for ev in self._query_paged(query_params, nmax):
                events[ev.name] = ev

        for ev in sorted(events.values(), key=lambda ev: ev.time):
            if magmin is not None and (
                    ev.magnitude is None or ev.magnitude < magmin):
                continue

            if time_range[0] <= ev.time and ev.time <= time_range[1]:
                self.events[ev.name] = ev
                yield ev.name

    def _query_paged(self, params, nmax):
        ipage = 0
        while True:
            events = self._query(dict(params, offset=1 + ipage * nmax))

            yield from events

            if len(events) < nmax:
                break

            ipage += 1

    def get_event(self, name):
        logger.debug('In Geofon.get_event("%s")' % name)

        if name not in self.events:
            events = self._query(dict(eventid=name))
            if not events:
                raise NotFound('GEOFON event "%s"' % name)

            self.events[name] = events[0]

        return self.events[name]

    def _query(self, params):
        if self._get_moment_tensors:
            params = dict(params, includefocalmechanism=True)

        try:
            qml = fdsn.event(site='geofon', parsed=True, **params)
        except fdsn.EmptyResult:
            return []

        events = []
        for qev in qml.event_parameters.event_list:
            try:
                ev = qev.get_pyrocko_event()
            except quakeml.QuakeMLError as e:
                logger.warning('Skipping event: %s' % e)
                continue

            # 'smi:org.gfz-potsdam.de/geofon/gfz2010avtm' -> 'gfz2010avtm'
            ev.name = qev.public_id.split('/')[-1]
            ev.catalog = 'GEOFON'

            focal_mechanism = qev.preferred_focal_mechanism
            if focal_mechanism is None and qev.focal_mechanism_list:
                focal_mechanism = qev.focal_mechanism_list[0]

            magnitude = self._get_moment_magnitude(qev, focal_mechanism)
            if magnitude is not None:
                ev.magnitude = magnitude.mag.value
                ev.magnitude_type = magnitude.type

            origin = qev.preferred_origin
            if origin is not None:
                tags = []
                if origin.evaluation_mode:
                    tags.append('evaluation_mode:%s' % origin.evaluation_mode)

                if origin.evaluation_status:
                    tags.append(
                        'evaluation_status:%s' % origin.evaluation_status)

                if origin.evaluation_mode == 'manual':
                    status = 'M'
                elif origin.evaluation_status == 'confirmed' \
                        or focal_mechanism is not None:
                    status = 'C'
                else:
                    status = 'A'

                tags.append('geofon_status:%s' % status)

                ev.tags = tags

            events.append(ev)

        return events

    def _get_moment_magnitude(self, qev, focal_mechanism):
        if focal_mechanism is None or not focal_mechanism.moment_tensor_list:
            return None

        mt = focal_mechanism.moment_tensor_list[0]
        for magnitude in qev.magnitude_list:
            if magnitude.public_id == mt.moment_magnitude_id \
                    and magnitude.mag is not None:

                return magnitude

        return None

    def get_mt(self, ev):
        '''
        Get moment tensor from GEOFON's ``mt.txt`` alert file.

        Not needed for events obtained with this client, which include the
        moment tensor if available.
        '''

        syear = time.strftime('%Y', time.gmtime(ev.time))
        url = 'https://geofon.gfz.de/data/alerts/%s/%s/mt.txt' % (
            syear, ev.name)

        logger.debug('Opening URL: %s' % url)
        r = self._session.get(url)
        logger.debug('Received page (%i bytes)' % len(r.content))

        if r.status_code == 404:
            logger.warning('No MT found for event "%s".' % ev.name)
            return None

        return self._parse_mt_page(r.content)

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
