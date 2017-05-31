import re
import urllib
import urllib2
import logging

from pyrocko import util

logger = logging.getLogger('pyrocko.fdsn.ws')

g_url = '%(site)s/fdsnws/%(service)s/%(majorversion)i/%(method)s'

g_site_abbr = {
    'geofon': 'https://geofon.gfz-potsdam.de',
    'iris': 'http://service.iris.edu',
    'orfeus': 'http://www.orfeus-eu.org',
    'bgr': 'http://eida.bgr.de',
    'geonet': 'http://service.geonet.org.nz',
    'knmi': 'http://rdsa.knmi.nl',
}

g_default_site = 'geofon'

g_timeout = 20.

re_realm_from_auth_header = re.compile(r'(realm)\s*[:=]\s*"([^"]*)"?')


class CannotGetRealmFromAuthHeader(Exception):
    pass


class CannotGetCredentialsFromAuthRequest(Exception):
    pass


def get_realm_from_auth_header(headers):
    realm = dict(re_realm_from_auth_header.findall(
        headers['WWW-Authenticate'])).get('realm', None)

    if realm is None:
        raise CannotGetRealmFromAuthHeader('headers=%s' % str(headers))

    return realm


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


def _request(url, post=False, user=None, passwd=None, **kwargs):
    url_values = urllib.urlencode(kwargs)
    if url_values:
        url += '?' + url_values
    logger.debug('Accessing URL %s' % url)

    opener = None

    req = urllib2.Request(url)
    if post:
        logger.debug('POST data: \n%s' % post)
        req.add_data(post)

    req.add_header('Accept', '*/*')

    itry = 0
    while True:
        itry += 1
        try:

            if opener:
                resp = opener.open(req, timeout=g_timeout)
            else:
                resp = urllib2.urlopen(req, timeout=g_timeout)

            if resp.getcode() == 204:
                raise EmptyResult(url)
            return resp

        except urllib2.HTTPError, e:
            if e.code == 413:
                raise RequestEntityTooLarge(url)

            elif e.code == 401:
                headers = getattr(e, 'headers', e.hdrs)

                realm = get_realm_from_auth_header(headers)

                if itry == 1 and user is not None:
                    auth_handler = urllib2.HTTPDigestAuthHandler()
                    auth_handler.add_password(
                        realm=realm,
                        uri=url,
                        user=user,
                        passwd=passwd or '')

                    opener = urllib2.build_opener(auth_handler)
                    continue
                else:
                    logger.error(
                        'authentication failed for realm "%s" when '
                        'accessing url "%s"' % (realm, url))
                    raise e

            else:
                logger.error(
                    'error content returned by server:\n%s' % e.read())
                raise e

        break


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


def get_auth_credentials(
        token, url=g_url, site=g_default_site, majorversion=1):

    url = fillurl(url, site, 'dataselect', majorversion, method='auth')

    f = _request(url, post=token)
    s = f.read()
    try:
        user, passwd = s.strip().split(':')
    except ValueError:
        raise CannotGetCredentialsFromAuthRequest('data="%s"' % s)

    return user, passwd


def dataselect(url=g_url, site=g_default_site, majorversion=1, selection=None,
               user=None, passwd=None, token=None,
               **kwargs):

    if user is not None:
        method = 'queryauth'
    else:
        method = 'query'

    if token is not None:
        user, passwd = get_auth_credentials(
            token, url=url, site=site, majorversion=majorversion)

    url = fillurl(url, site, 'dataselect', majorversion, method=method)

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

        return _request(url, user=user, passwd=passwd, post='\n'.join(l))
    else:
        return _request(url, user=user, passwd=passwd, **params)
