
import os
import time
import shutil
import urllib
import urllib2
import httplib
import logging
import re

from pyrocko import util

logger = logging.getLogger('pyrocko.gf.ws')

g_url = '%(site)s/gfws/%(service)s/%(majorversion)i/%(method)s'
g_url_static = '%(site)s/gfws/%(service)s'

g_site_abbr = {
    'localhost': 'http://localhost:8080'}

g_default_site = 'localhost'


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
        resp = urllib2.urlopen(req)
        if resp.getcode() == 204:
            raise EmptyResult(url)
        return resp

    except urllib2.HTTPError, e:
        if e.code == 413:
            raise RequestEntityTooLarge(url)
        else:
            raise


def fillurl(url, site, service, majorversion, method='query'):
    return url % dict(
        site=g_site_abbr.get(site, site),
        service=service,
        majorversion=majorversion,
        method=method)


def static(url=g_url_static, site=g_default_site, majorversion=1, **kwargs):

    url = fillurl(url, site, 'static', majorversion)
    return _request(url, **kwargs)


def ujoin(*args):
    return '/'.join(args)


class DownloadError(Exception):
    pass


class PathExists(DownloadError):
    pass


class Incomplete(DownloadError):
    pass


def rget(url, path, force=False, method='download', stats=None,
         status_callback=None, entries_wanted=None):

    if stats is None:
        stats = [0, None] # bytes received, bytes expected
    
    if not url.endswith('/'):
        url = url + '/'

    resp = urllib2.urlopen(url)
    data = resp.read()
    resp.close()

    l = re.findall(r'href="([a-zA-Z0-9_.-]+/?)"', data)
    l = sorted(set(x for x in l if x.rstrip('/') not in ('.', '..')))
    if entries_wanted is not None:
        l = [ x for x in l if x.rstrip('/') in entries_wanted ]

    if method == 'download':
        if os.path.exists(path):
            if not force:
                raise PathExists('path "%s" already exists' % path)
            else:
                shutil.rmtree(path)

        os.mkdir(path)

    for x in l:
        if x.endswith('/'):
            rget(
                url + x,
                os.path.join(path, x), 
                force=force,
                method=method,
                stats=stats,
                status_callback=status_callback)

        else:
            req = urllib2.Request(url + x)
            if method == 'calcsize':
                req.get_method = lambda: 'HEAD'

            resp = urllib2.urlopen(req)
            sexpected = int(resp.headers['content-length'])

            if method == 'download':
                out = open(os.path.join(path, x), 'w')
                sreceived = 0
                while True:
                    data = resp.read(1024*4)
                    if not data:
                        break

                    sreceived += len(data)
                    stats[0] += len(data)
                    if status_callback and stats[1] is not None:
                        status_callback(stats[0], stats[1])

                    out.write(data)

                if sreceived != sexpected:
                    raise Incomplete(
                        'unexpected end of file while downloading %s' % (
                            url + x))

                out.close()

            else:
                stats[0] += sexpected

            resp.close()

    if status_callback and stats[0] == stats[1]:
        status_callback(stats[0], stats[1])

    return stats[0]

def download_gf_store(url=g_url_static, site=g_default_site, majorversion=1,
                      store_id=None, force=False):

    url = fillurl(url, site, 'static', majorversion)

    stores_url = ujoin(url, 'stores')

    tlast = [ time.time() ]
    def status_callback(i,n):
        tnow = time.time()
        if (tnow - tlast[0]) > 5 or i == n:
            print '%s / %s [%.1f%%]' % (util.human_bytesize(i), util.human_bytesize(n), i*100.0/n)
            tlast[0] = tnow

    wanted = [ 'config', 'extra', 'index', 'phases', 'traces' ]

    try:
        if store_id is None:
            print static(url=stores_url+'/', format='text').read()

        else:
            store_url = ujoin(stores_url, store_id)
            stotal = rget(
                store_url, store_id, force=force, method='calcsize',
                entries_wanted=wanted)
                          
            rget(
                store_url, store_id, force=force, stats=[0, stotal],
                status_callback=status_callback, entries_wanted=wanted)

    except (urllib2.URLError, urllib2.HTTPError, httplib.HTTPException), e:
        raise DownloadError('download failed. Original error was: %s, %s' % (
            type(e).__name__, e))

