import urllib
import urllib2
import logging

from pyrocko import util

logger = logging.getLogger('pyrocko.fdsn.ws')

g_url = '%(site)s/fdsnws/%(service)s/%(majorversion)i/%(method)s'

g_site_abbr = {
    'geofon': 'http://geofon-open1.gfz-potsdam.de',
    'iris': 'http://service.iris.edu',
    'orfeus': 'http://www.orfeus-eu.org',
}


g_default_site = 'geofon'

g_timeout = 20.


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


class InvalidRequest(Exception):
    pass


def _request(url, post=False, **kwargs):
    url_values = urllib.urlencode(kwargs)
    if url_values:
        url += '?' + url_values
    logger.debug('Accessing URL %s' % url)

    req = urllib2.Request(url)
    if post:
        logger.debug('POST data: \n%s' % post)
        req.add_data(post)

    req.add_header('Accept', '*/*')

    try:
        resp = urllib2.urlopen(req, timeout=g_timeout)
        if resp.getcode() == 204:
            raise EmptyResult(url)
        return resp

    except urllib2.HTTPError, e:
        if e.code == 413:
            raise RequestEntityTooLarge(url)
        else:
            logger.error('error content returned by server:\n%s' % e.read())
            raise e


def fillurl(url, site, service, majorversion, method='query'):
    return url % dict(
        site=g_site_abbr.get(site, site),
        service=service,
        majorversion=majorversion,
        method=method)


def fix_params(d):

    params = dict(d)
    for k in '''start end starttime endtime startbefore startafter endbefore
            endafter'''.split():

        if k in params:
            params[k] = sdatetime(params[k])

    if params.get('location', None) == '':
        params['location'] = '--'

    for k in params:
        if isinstance(params[k], bool):
            params[k] = ['FALSE', 'TRUE'][bool(params[k])]

    return params


def make_data_selection(stations, tmin, tmax,
                        channel_prio=[['BHZ', 'HHZ'],
                                      ['BH1', 'BHN', 'HH1', 'HHN'],
                                      ['BH2', 'BHE', 'HH2', 'HHE']]):

    selection = []
    for station in stations:
        wanted = []
        for group in channel_prio:
            gchannels = []
            for channel in station.get_channels():
                if channel.name in group:
                    gchannels.append(channel)
            if gchannels:
                gchannels.sort(lambda a, b: cmp(group.index(a.name),
                                                group.index(b.name)))
                wanted.append(gchannels[0])

        if wanted:
            for channel in wanted:
                selection.append((station.network, station.station,
                                  station.location, channel.name, tmin, tmax))

    return selection


def station(url=g_url, site=g_default_site, majorversion=1, parsed=True,
            selection=None, **kwargs):

    url = fillurl(url, site, 'station', majorversion)

    params = fix_params(kwargs)

    if selection:
        l = []
        for k, v in params.iteritems():
            l.append('%s=%s' % (k, v))

        for (network, station, location, channel, tmin, tmax) in selection:
            if location == '':
                location = '--'

            l.append(' '.join((network, station, location, channel,
                               sdatetime(tmin), sdatetime(tmax))))

        params = dict(post='\n'.join(l))

    if parsed:
        from pyrocko.fdsn import station
        format = params.get('format', 'xml')
        if format == 'text':
            if params.get('level', 'station') == 'channel':
                return station.load_channel_table(
                    stream=_request(url, **params))
            else:
                raise InvalidRequest('if format="text" shall be parsed, '
                                     'level="channel" is required')

        elif format == 'xml':
            assert params.get('format', 'xml') == 'xml'
            return station.load_xml(stream=_request(url, **params))
        else:
            raise InvalidRequest('format must be "xml" or "text"')
    else:
        return _request(url, **params)


def dataselect(url=g_url, site=g_default_site, majorversion=1, selection=None,
               **kwargs):

    url = fillurl(url, site, 'dataselect', majorversion)

    params = fix_params(kwargs)

    if selection:
        l = []

        if 'minimumlength' not in params:
            params['minimumlength'] = 0.0

        if 'longestonly' not in params:
            params['longestonly'] = 'FALSE'

        for k, v in params.iteritems():
            l.append('%s=%s' % (k, v))

        for (network, station, location, channel, tmin, tmax) in selection:
            if location == '':
                location = '--'

            l.append(' '.join((network, station, location, channel,
                               sdatetime(tmin), sdatetime(tmax))))

        return _request(url, post='\n'.join(l))
    else:
        return _request(url, **params)
