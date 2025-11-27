# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Simple Green's function store web service.
'''

import datetime
import hashlib
import io
import json
import os
import re
import email

from tornado import iostream
from tornado import web
from tornado import httputil
from tornado.escape import utf8

import matplotlib.pyplot as plt  # noqa

from pyrocko import server, gf, plot
from pyrocko.plot import cake_plot


add_cli_options = server.add_cli_options
stop = server.stop


class FomostoRequestHandler(server.RequestHandler):
    pass


g_templates = None
g_engine = None


def get_templates():
    from jinja2 import Template

    global g_templates

    if g_templates is None:
        g_templates = {
            'list-html': Template('''
<!DOCTYPE html>
<html lang="en">
<head>
<title>{{ title }}</title>
</head>
<body>
<h2>{{ title }}</h2>
<hr>
<table>
<tr>
    <th style="text-align:left">Store ID</th>
    <th style="text-align:center">Type</th>
    <th style="text-align:center">Extent</th>
    <th style="text-align:center">Sample-rate</th>
    <th style="text-align:center">Size (index + traces)</th>
</tr>
{% for store in stores %}
<tr>
    <td><a href="{{ store.config.id }}/">{{ store.config.id|e }}/</a></td>
    <td style="text-align:center">{{ store.config.short_type }}</td>
    <td style="text-align:right">{{ store.config.short_extent }} km</td>
    <td style="text-align:right">{{ store.config.sample_rate }} Hz</td>
    <td style="text-align:right">{{ store.size_index_and_data_human }}</td>
</tr>
{% endfor %}
</table>
</hr>
</body>
</html>'''.lstrip()),

            'list-text': Template('''
{% for store in stores %}{#
#}{{ store.config.id.ljust(25) }} {#
#}{{ store.config.short_type.center(5) }} {#
#}{{ store.config.short_extent.rjust(30) }} km {#
#}{{ "%10.2g"|format(store.config.sample_rate) }} Hz {#
#}{{ store.size_index_and_data_human.rjust(8) }}
{% endfor %}'''.lstrip()),

            'dirlist-html': Template('''
<!DOCTYPE html>
<html lang="en">
<head>
<title>Directory listing for {{ path }}</title>
</head>
<body>
<h2>Directory listing for {{ path }}</h2>
<ul>
{% for item in items %}
<li><a href="{{ item.url }}">{{ item.name_display }}</a>
{% endfor %}
</ul>'''.lstrip())}

    return g_templates


class ListStoresHandler(FomostoRequestHandler):

    def initialize(self, default_format='html'):
        self._default_format = default_format

    def get(self):
        return self.list_stores()

    def list_stores(self):
        '''Create listing of stores.'''

        if not self.request.path.endswith('/'):
            self.redirect(self.request.path + '/', permanent=True)

        store_ids = list(g_engine.get_store_ids())
        store_ids.sort(key=lambda x: x.lower())

        stores = [g_engine.get_store(store_id) for store_id in store_ids]

        format = self.get_query_argument('format', self._default_format)

        title = "Green's function stores listing"

        mime_types = {
            'html': 'text/html; charset=utf-8',
            'text': 'text/plain; charset=utf-8',
            'json': 'application/json',
        }

        if format in ('html', 'text'):
            s = get_templates()['list-' + format].render(
                stores=stores, title=title).encode('utf8')

        elif format == 'json':

            def get_store_dict(store):
                store.ensure_reference()

                return {
                    'id': store.config.id,
                    'short_type': store.config.short_type,
                    'modelling_code_id': store.config.modelling_code_id,
                    'source_depth_min': store.config.source_depth_min,
                    'source_depth_max': store.config.source_depth_max,
                    'source_depth_delta': store.config.source_depth_delta,
                    'distance_min': store.config.distance_min,
                    'distance_max': store.config.distance_max,
                    'distance_delta': store.config.distance_delta,
                    'sample_rate': store.config.sample_rate,
                    'size': store.size_index_and_data,
                    'uuid': store.config.uuid,
                    'reference': store.config.reference
                }

            stores = {
                'stores': [get_store_dict(g_engine.get_store(store_id))
                           for store_id in store_ids]
            }

            s = json.dumps(stores)
        else:
            raise web.HTTPError(
                400, reason='format not available')

        self.set_header('Content-Type', mime_types[format])
        self.set_header('Content-Length', str(len(s)))
        self.set_header('Access-Control-Allow-Origin', '*')
        self.write(s)


def cleanpath(path):
    return os.path.normpath(os.path.abspath(path)).rstrip('/')


class StaticStoresHandler(server.RequestHandler):

    def head(self, store_id, path):
        return self.get(store_id, path, include_body=False)

    async def get(self, store_id, path, include_body=True):
        path = path.strip('/')
        try:
            store_dir = cleanpath(g_engine.get_store_dir(store_id))
        except gf.NoSuchStore:
            raise web.HTTPError(404)

        abspath = cleanpath(os.path.join(store_dir, path))

        if not abspath.startswith(store_dir):
            raise web.HTTPError(404)

        if os.path.isdir(abspath):
            if not self.request.path.endswith('/'):
                self.redirect(self.request.path + '/', permanent=True)

            self.set_header('Content-Type', 'text/html; charset=utf-8')
            bytes = self.get_directory_listing(abspath)
            self.set_header('Content-Length', len(bytes))
            self.set_header('Etag', self.get_content_etag(abspath))
            self.write(bytes)
            return

        self.set_header('Etag', self.get_content_etag(abspath))
        if self.match_header_etags():
            self.set_status(304)
            return

        self.set_header('Last-Modified', self.get_content_modified(abspath))
        if self.match_header_if_modified_since():
            self.set_status(304)

        self.set_header('Content-Type', self.get_content_type(abspath))

        size = self.get_content_size(abspath)

        start, end, content_length = self.get_request_range(size)

        self.set_header('Content-Length', content_length)

        if include_body:
            for bytes in self.get_content(abspath, start, end):
                try:
                    self.write(bytes)
                    await self.flush()
                except iostream.StreamClosedError:
                    return

    def get_content_modified(self, path):
        modified = datetime.datetime.fromtimestamp(
            int(os.stat(path).st_mtime), datetime.timezone.utc
        )
        return modified

    def get_content_type(self, path):
        if os.path.basename(path) == 'config':
            return 'text/plain'
        if os.path.basename(os.path.dirname(path)) == 'extra':
            return 'text/plain'
        else:
            return 'application/octet-stream'

    def get_content_size(self, path):
        return os.stat(path).st_size

    def get_content_etag(self, path):
        hasher = hashlib.sha1()
        hasher.update(('%i' % os.stat(path).st_mtime_ns).encode('utf8'))
        hasher.update(path.encode('utf8'))
        return '"%s"' % hasher.hexdigest()

    def get_request_range(self, size):
        request_range = None
        range_header = self.request.headers.get('Range')
        if range_header:
            # As per RFC 2616 14.16, if an invalid Range header is specified,
            # the request will be treated as if the header didn't exist.
            request_range = httputil._parse_request_range(range_header)

        if request_range:
            start, end = request_range
            if start is not None and start < 0:
                start += size
                if start < 0:
                    start = 0
            if (
                start is not None
                and (start >= size or (end is not None and start >= end))
            ) or end == 0:
                # As per RFC 2616 14.35.1, a range is not satisfiable only: if
                # the first requested byte is equal to or greater than the
                # content, or when a suffix with length 0 is specified.
                # https://tools.ietf.org/html/rfc7233#section-2.1
                # A byte-range-spec is invalid if the last-byte-pos value is
                # present and less than the first-byte-pos.
                self.set_status(416)  # Range Not Satisfiable
                self.set_header('Content-Type', 'text/plain')
                self.set_header('Content-Range', 'bytes */%s' % (size,))
                return
            if end is not None and end > size:
                # Clients sometimes blindly use a large range to limit their
                # download size; cap the endpoint at the actual file size.
                end = size
            # Note: only return HTTP 206 if less than the entire range has been
            # requested. Not only is this semantically correct, but Chrome
            # refuses to play audio if it gets an HTTP 206 in response to
            # ``Range: bytes=0-``.
            if size != (end or size) - (start or 0):
                self.set_status(206)  # Partial Content
                self.set_header(
                    'Content-Range',
                    httputil._get_content_range(start, end, size)
                )
        else:
            start = end = None

        if start is not None and end is not None:
            content_length = end - start
        elif end is not None:
            content_length = end
        elif start is not None:
            content_length = size - start
        else:
            content_length = size

        return start, end, content_length
        # --- end range handling (from StaticFileHandler)

    def match_header_etags(self):

        # --- start based on StaticFileHandler
        computed_etag = utf8(self._headers.get("Etag", ""))
        # Find all weak and strong etag values from If-None-Match header
        # because RFC 7232 allows multiple etag values in a single header.
        etags = re.findall(
            rb'\*|(?:W/)?"[^"]*"',
            utf8(self.request.headers.get("If-None-Match", "")))

        if not computed_etag or not etags:
            return False

        if etags[0] == b"*":
            return True
        else:
            # Use a weak comparison when comparing entity-tags.
            def val(x: bytes) -> bytes:
                return x[2:] if x.startswith(b"W/") else x

            return any(val(etag) == val(computed_etag) for etag in etags)
        # --- end based on StaticFileHandler

    def match_header_if_modified_since(self):
        modified_value = utf8(self._headers.get("Last-Modified", ""))
        if not modified_value:
            return False

        ims_value = self.request.headers.get("If-Modified-Since")
        if ims_value is None:
            return False

        if_since = email.utils.parsedate_to_datetime(ims_value)
        if if_since.tzinfo is None:
            if_since = if_since.replace(tzinfo=datetime.timezone.utc)

        modified = email.utils.parsedate_to_datetime(modified_value)
        return if_since >= modified

    def get_content(self, path, start=None, end=None):
        with open(path, 'rb') as file:
            if start is not None:
                file.seek(start)
            if end is not None:
                remaining = end - (start or 0)
            else:
                remaining = None
            while True:
                chunk_size = 1024 * 1024
                if remaining is not None and remaining < chunk_size:
                    chunk_size = remaining
                chunk = file.read(chunk_size)
                if chunk:
                    if remaining is not None:
                        remaining -= len(chunk)
                    yield chunk
                else:
                    if remaining is not None:
                        assert remaining == 0
                    return

    def get_directory_listing(self, dir_path):
        items = os.listdir(dir_path)

        data = []
        for item in items:
            item_path = os.path.join(dir_path, item)
            suffix = ('/' if os.path.isdir(item_path) else '')
            data.append({
                'name_display': item + suffix,
                'url': item + suffix,
            })

        template = get_templates()['dirlist-html']
        return template.render(
            items=data, path=self.request.path).encode('utf8')


class StoreConfigHandler(server.RequestHandler):

    # needed for Green's Mill js web interface

    def get(self, store_id):
        try:
            store = g_engine.get_store(store_id)
        except Exception:
            raise web.HTTPError(404)

        data = {}
        data['id'] = store_id
        data['config'] = str(store.config)

        bytes = json.dumps(data).encode('utf8')

        self.set_header('Content-Type', 'application/json')
        self.set_header('Content-Length', len(bytes))
        self.set_header('Access-Control-Allow-Origin', '*')
        self.write(bytes)


class StoreProfileHandler(server.RequestHandler):

    # needed for Green's Mill js web interface

    def get(self, store_id):
        fontsize = 9.0
        plot.mpl_init(fontsize=fontsize)
        fig = plt.figure(figsize=plot.mpl_papersize('a5', 'landscape'))

        try:
            store = g_engine.get_store(store_id)
        except Exception:
            raise web.HTTPError(404)

        if store.config.earthmodel_1d is None:
            raise web.HTTPError(404)

        cake_plot.my_model_plot(store.config.earthmodel_1d, fig=fig)

        f = io.BytesIO()
        fig.savefig(f, format='png')
        bytes = f.getvalue()

        self.set_header('Content-Type', 'image/png;')
        self.set_header('Content-Length', str(len(bytes)))
        self.set_header('Access-Control-Allow-Origin', '*')

        self.write(bytes)


def run(engine,
        host='localhost',
        port=8085,
        open=False,
        debug=False):

    global g_engine
    g_engine = engine

    store_id_pattern = gf.StringID.pattern[1:-1]

    handlers = [
        (
            r'/(?:gfws(?:/(?:static(?:/(?:stores)?)?)?)?)?',
            web.RedirectHandler,
            dict(url='/gfws/static/stores/', permanent=False),
        ),
        (
            r'/gfws/static/stores/?',
            ListStoresHandler,
        ),
        (
            r'/gfws/static/stores/(' + store_id_pattern + ')(?:(/.*|))',
            StaticStoresHandler,
        ),
        (
            r'/gfws/api/?',
            ListStoresHandler,
            dict(default_format='json'),
        ),
        (
            r'/gfws/api/(' + store_id_pattern + ')',
            StoreConfigHandler,
        ),
        (
            r'/gfws/api/(' + store_id_pattern + ')/profile',
            StoreProfileHandler,
        ),
    ]

    server.run(
        host=host,
        port=port,
        handlers=handlers,
        debug=debug)
