import urllib
import urllib2
import logging

from pyrocko import util

logger = logging.getLogger('pyrocko.fdsn.ws')

g_url = '%(site)s/fdsnws/%(service)s/%(majorversion)i/%(method)s'

g_site_abbr = {
    'geofon': 'http://geofon-open2.gfz-potsdam.de',
    'iris': 'http://service.iris.edu'}


def sdatetime(t):
    return util.time_to_str(t, format='%Y-%m-%dT%H:%M:%S')


class EmptyResult(Exception):
    def __init__(self, url):
        Exception.__init__(self)
        self._url = url

    def __str__(self):
        return 'No results for request %s' % self._url


class RequestEntityTooLarge(Exception):
    def __init__(self, url):
        Exception.__init__(self)
        self._url = url

    def __str__(self):
        return 'Request entity too large: %s' % self._url


def _request(url, post=False, **kwargs):
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
        if e.code == 204:
            raise EmptyResult(url)
        elif e.code == 413:
            raise RequestEntityTooLarge(url)
        else:
            raise e


def fillurl(url, site, service, majorversion, method='query'):
    return url % dict(
        site=g_site_abbr.get(site, site),
        service=service,
        majorversion=majorversion,
        method=method)


def station(url=g_url, site='geofon', majorversion=1, parsed=True, **kwargs):

    url = fillurl(url, site, 'station', majorversion)

    params = dict(kwargs)

    for k in '''
            starttime endtime startbefore startafter endbefore endafter
            '''.split():
        if k in params:
            params[k] = sdatetime(params[k])

    if params.get('location', None) == '':
        params['location'] = '--'

    if parsed:
        from pyrocko.fdsn import station
        return station.load_xml(stream=_request(url, **params))
    else:
        return _request(url, **params)
