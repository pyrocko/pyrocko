# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
from __future__ import absolute_import
from builtins import str

import re
import logging
import ssl
try:
    from urllib.parse import urlencode
except ImportError:
    from urllib import urlencode

try:
    from urllib2 import (Request, build_opener, HTTPDigestAuthHandler,
                         HTTPError, urlopen)
except ImportError:
    from urllib.request import (Request, build_opener, HTTPDigestAuthHandler,
                                urlopen)
    from urllib.error import HTTPError

from pyrocko import util


logger = logging.getLogger('pyrocko.client.fdsn')

g_url = '%(site)s/fdsnws/%(service)s/%(majorversion)i/%(method)s'

g_site_abbr = {
    'geofon': 'https://geofon.gfz-potsdam.de',
    'iris': 'http://service.iris.edu',
    'orfeus': 'http://www.orfeus-eu.org',
    'bgr': 'http://eida.bgr.de',
    'geonet': 'http://service.geonet.org.nz',
    'knmi': 'http://rdsa.knmi.nl',
    'ncedc': 'http://service.ncedc.org',
    'scedc': 'http://scedc.caltech.edu',
    'usgs': 'http://earthquake.usgs.gov',
    'bgr': 'http://eida.bgr.de',
    'koeri': 'http://www.koeri.boun.edu.tr/2/tr',
    'ethz': 'http://eida.ethz.ch/fdsnws',
    'icgc': 'http://www.icgc.cat/en/xarxasismica',
    'ipgp': 'http://centrededonnees.ipgp.fr',
    'ingv': 'http://webservices.rm.ingv.it',
    'isc': 'http://www.isc.ac.uk',
    'lmu': 'http://erde.geophysik.uni-muenchen.de',
    'noa': 'http://bbnet.gein.noa.gr',
    'resif': 'http://portal.resif.fr',
    'usp': 'http://www.moho.iag.usp.br'
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


def _request(url, post=False, user=None, passwd=None,
             allow_TLSv1=True, **kwargs):
    url_values = urlencode(kwargs)
    if url_values:
        url += '?' + url_values
    logger.debug('Accessing URL %s' % url)

    url_args = {
        'timeout': g_timeout
    }

    try:
        url_args['context'] = ssl.SSLContext(
            ssl.PROTOCOL_TLSv1 if allow_TLSv1 else ssl.PROTOCOL_SSLv2)
    except AttributeError:
        try:
            url_args['context'] = ssl.create_default_context()
        except AttributeError:
            pass

    opener = None

    req = Request(url)
    if post:
        logger.debug('POST data: \n%s' % post)
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
        format = params.get('format', 'xml')
        if format == 'text':
            if params.get('level', 'station') == 'channel':
                return stationxml.load_channel_table(
                    stream=_request(url, **params))
            else:
                raise InvalidRequest('if format="text" shall be parsed, '
                                     'level="channel" is required')

        elif format == 'xml':
            assert params.get('format', 'xml') == 'xml'
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

        if 'minimumlength' not in params:
            params['minimumlength'] = 0.0

        if 'longestonly' not in params:
            params['longestonly'] = 'FALSE'

        for k, v in params.items():
            lst.append('%s=%s' % (k, v))

        for (network, station, location, channel, tmin, tmax) in selection:
            if location == '':
                location = '--'

            lst.append(' '.join((network, station, location, channel,
                                 sdatetime(tmin), sdatetime(tmax))))

        post = '\n'.join(lst)
        return _request(url, user=user, passwd=passwd, post=post.encode())
    else:
        return _request(url, user=user, passwd=passwd, **params)
