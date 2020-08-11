# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
from __future__ import absolute_import

import re
import logging
import ssl
import socket


from pyrocko import util
from pyrocko.util import DownloadError
from pyrocko import config

from pyrocko.util import \
    urlencode, Request, build_opener, HTTPDigestAuthHandler, urlopen, HTTPError

try:
    newstr = unicode
except NameError:
    newstr = str

logger = logging.getLogger('pyrocko.client.fdsn')

g_url = '%(site)s/fdsnws/%(service)s/%(majorversion)i/%(method)s'
g_url_wadl = '%(site)s/fdsnws/%(service)s/%(majorversion)i/application.wadl'

g_site_abbr = {
    'geofon': 'https://geofon.gfz-potsdam.de',
    'iris': 'http://service.iris.edu',
    'orfeus': 'http://www.orfeus-eu.org',
    'bgr': 'http://eida.bgr.de',
    'geonet': 'http://service.geonet.org.nz',
    'knmi': 'http://rdsa.knmi.nl',
    'ncedc': 'http://service.ncedc.org',
    'scedc': 'http://service.scedc.caltech.edu',
    'usgs': 'http://earthquake.usgs.gov',
    'koeri': 'http://eida-service.koeri.boun.edu.tr',
    'ethz': 'http://eida.ethz.ch',
    'icgc': 'http://ws.icgc.cat',
    'ipgp': 'http://eida.ipgp.fr',
    'ingv': 'http://webservices.ingv.it',
    'isc': 'http://www.isc.ac.uk',
    'lmu': 'http://erde.geophysik.uni-muenchen.de',
    'noa': 'http://eida.gein.noa.gr',
    'resif': 'http://ws.resif.fr',
    'usp': 'http://seisrequest.iag.usp.br',
    'niep': 'http://eida-sc3.infp.ro'
}

g_default_site = 'geofon'


def strip_html(s):
    s = s.decode('utf-8')
    s = re.sub(r'<[^>]+>', '', s)
    s = re.sub(r'\r', '', s)
    s = re.sub(r'\s*\n', '\n', s)
    return s


def indent(s, ind='  '):
    return '\n'.join(ind + line for line in s.splitlines())


def get_sites():
    '''
    Get sorted list of registered site names.
    '''
    return sorted(g_site_abbr.keys())


if config.config().fdsn_timeout is None:
    g_timeout = 20.
else:
    g_timeout = config.config().fdsn_timeout

re_realm_from_auth_header = re.compile(r'(realm)\s*[:=]\s*"([^"]*)"?')


class CannotGetRealmFromAuthHeader(DownloadError):
    pass


class CannotGetCredentialsFromAuthRequest(DownloadError):
    pass


def get_realm_from_auth_header(headers):
    realm = dict(re_realm_from_auth_header.findall(
        headers['WWW-Authenticate'])).get('realm', None)

    if realm is None:
        raise CannotGetRealmFromAuthHeader('headers=%s' % str(headers))

    return realm


def sdatetime(t):
    return util.time_to_str(t, format='%Y-%m-%dT%H:%M:%S')


class EmptyResult(DownloadError):
    def __init__(self, url):
        DownloadError.__init__(self)
        self._url = url

    def __str__(self):
        return 'No results for request %s' % self._url


class RequestEntityTooLarge(DownloadError):
    def __init__(self, url):
        DownloadError.__init__(self)
        self._url = url

    def __str__(self):
        return 'Request entity too large: %s' % self._url


class InvalidRequest(DownloadError):
    pass


class Timeout(DownloadError):
    pass


def _request(url, post=False, user=None, passwd=None,
             allow_TLSv1=False, **kwargs):
    timeout = float(kwargs.pop('timeout', g_timeout))
    url_values = urlencode(kwargs)
    if url_values:
        url += '?' + url_values

    logger.debug('Accessing URL %s' % url)
    url_args = {
        'timeout': timeout
    }

    if allow_TLSv1:
        url_args['context'] = ssl.SSLContext(ssl.PROTOCOL_TLSv1)

    opener = None

    req = Request(url)
    if post:
        if isinstance(post, newstr):
            post = post.encode('utf8')
        logger.debug('POST data: \n%s' % post.decode('utf8'))
        req.data = post

    req.add_header('Accept', '*/*')

    itry = 0
    while True:
        itry += 1
        try:
            urlopen_ = opener.open if opener else urlopen
            while True:
                try:
                    resp = urlopen_(req, **url_args)
                    break
                except TypeError:
                    del url_args['context']  # context not avail before 3.4.3

            logger.debug('Response: %s' % resp.getcode())
            if resp.getcode() == 204:
                raise EmptyResult(url)
            return resp

        except HTTPError as e:
            if e.code == 413:
                raise RequestEntityTooLarge(url)

            elif e.code == 401:
                headers = getattr(e, 'headers', e.hdrs)

                realm = get_realm_from_auth_header(headers)

                if itry == 1 and user is not None:
                    auth_handler = HTTPDigestAuthHandler()
                    auth_handler.add_password(
                        realm=realm,
                        uri=url,
                        user=user,
                        passwd=passwd or '')

                    opener = build_opener(auth_handler)
                    continue
                else:
                    raise DownloadError(
                        'Authentication failed for realm "%s" when accessing '
                        'url "%s". Original error was: %s' % (
                            realm, url, str(e)))

            else:
                raise DownloadError(
                    'Error content returned by server (HTML stripped):\n%s\n'
                    '  Original error was: %s' % (
                        indent(
                            strip_html(e.read()),
                            '  !  '),
                        str(e)))

        except socket.timeout:
            raise Timeout(
                'Timeout error. No response received within %i s. You '
                'may want to retry with a longer timeout setting.' % timeout)

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
            params[k] = ['false', 'true'][bool(params[k])]

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
                gchannels.sort(key=lambda a: group.index(a.name))
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
        lst = []
        for k, v in params.items():
            lst.append('%s=%s' % (k, v))

        for (network, station, location, channel, tmin, tmax) in selection:
            if location == '':
                location = '--'

            lst.append(' '.join((network, station, location, channel,
                                 sdatetime(tmin), sdatetime(tmax))))

        post = '\n'.join(lst)
        params = dict(post=post.encode())

    if parsed:
        from pyrocko.io import stationxml
        format = kwargs.get('format', 'xml')
        if format == 'text':
            if kwargs.get('level', 'station') == 'channel':
                return stationxml.load_channel_table(
                    stream=_request(url, **params))
            else:
                raise InvalidRequest('if format="text" shall be parsed, '
                                     'level="channel" is required')

        elif format == 'xml':
            assert kwargs.get('format', 'xml') == 'xml'
            return stationxml.load_xml(stream=_request(url, **params))
        else:
            raise InvalidRequest('format must be "xml" or "text"')
    else:
        return _request(url, **params)


def get_auth_credentials(
        token, url=g_url, site=g_default_site, majorversion=1):

    url = fillurl(url, site, 'dataselect', majorversion, method='auth')

    f = _request(url, post=token)
    s = f.read().decode()
    try:
        user, passwd = s.strip().split(':')
    except ValueError:
        raise CannotGetCredentialsFromAuthRequest('data="%s"' % s)

    return user, passwd


def dataselect(url=g_url, site=g_default_site, majorversion=1, selection=None,
               user=None, passwd=None, token=None,
               **kwargs):

    if user or token:
        method = 'queryauth'
    else:
        method = 'query'

    if token is not None:
        user, passwd = get_auth_credentials(
            token, url=url, site=site, majorversion=majorversion)

    url = fillurl(url, site, 'dataselect', majorversion, method=method)

    params = fix_params(kwargs)

    if selection:
        lst = []

        for k, v in params.items():
            lst.append('%s=%s' % (k, v))

        for (network, station, location, channel, tmin, tmax) in selection:
            if location == '':
                location = '--'

            lst.append(' '.join((network, station, location, channel,
                                 sdatetime(tmin), sdatetime(tmax))))

        post = '\n'.join(lst)
        return _request(url, user=user, passwd=passwd, post=post.encode(),
                        timeout=params.get('timeout', g_timeout))
    else:
        return _request(url, user=user, passwd=passwd, **params)


def event(url=g_url, site=g_default_site, majorversion=1,
          user=None, passwd=None, token=None, **kwargs):

    '''Query FDSN web service for events

    On success, will return a list of events in QuakeML format.

    Check the documentation of FDSN for allowed arguments:
    https://www.fdsn.org/webservices
    '''

    allowed_kwargs = {
        'starttime', 'endtime', 'minlatitude', 'maxlatitude',
        'minlongitude', 'maxlongitude', 'latitude', 'longitude',
        'minradius', 'maxradius', 'mindepth', 'maxdepth', 'minmagnitude',
        'maxmagnitude', 'magnitudetype', 'eventtype', 'includeallorigins',
        'includeallmagnitudes', 'includearrivals', 'eventid'}

    for k in kwargs.keys():
        if k not in allowed_kwargs:
            raise ValueError('invalid argument: %s' % k)

    if user or token:
        method = 'queryauth'
    else:
        method = 'query'

    if token is not None:
        user, passwd = get_auth_credentials(
            token, url=url, site=site, majorversion=majorversion)

    url = fillurl(url, site, 'event', majorversion, method=method)

    params = fix_params(kwargs)

    return _request(url, user=user, passwd=passwd, **params)


def wadl(service, url=g_url_wadl, site=g_default_site, majorversion=1,
         timeout=g_timeout):

    from pyrocko.client.wadl import load_xml

    url = fillurl(url, site, service, majorversion)

    return load_xml(stream=_request(url, timeout=timeout))


if __name__ == '__main__':
    import sys

    util.setup_logging('pyrocko.client.fdsn', 'info')

    if len(sys.argv) == 1:
        sites = get_sites()
    else:
        sites = sys.argv[1:]

    for site in sites:
        print('=== %s (%s) ===' % (site, g_site_abbr[site]))

        for service in ['station', 'dataselect', 'event']:
            try:
                app = wadl(service, site=site, timeout=2.0)
                print(indent(str(app)))

            except Timeout as e:
                logger.error(str(e))
                print('%s: timeout' % (service,))

            except util.DownloadError as e:
                logger.error(str(e))
                print('%s: no wadl' % (service,))
