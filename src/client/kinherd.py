# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function, division

try:
    from urllib.request import Request
    from urllib.error import HTTPError
    from urllib.parse import urlencode
    from future.moves.urllib.request import urlopen
except ImportError:
    from urllib2 import Request, urlopen
    from urllib2 import HTTPError
    from urllib import urlencode
import calendar
import logging

from pyrocko import util, model
from pyrocko.moment_tensor import MomentTensor
from .base_catalog import EarthquakeCatalog, NotFound

logger = logging.getLogger('pyrocko.client.kinherd')

km = 1000.


def ws_request(url, post=False, **kwargs):
    url_values = urlencode(kwargs)
    url = url + '?' + url_values
    logger.debug('Accessing URL %s' % url)

    req = Request(url)
    if post:
        req.add_data(post)

    req.add_header('Accept', '*/*')

    try:
        return urlopen(req)

    except HTTPError as e:
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
