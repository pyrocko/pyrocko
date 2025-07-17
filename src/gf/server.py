# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
'''
Simple async HTTP server

Based on this recipe:

    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/440665

which is based on this one:

    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/259148
'''

import asynchat
import asyncore
import socket
try:
    from http.server import SimpleHTTPRequestHandler as SHRH
    from html import escape
except ImportError:
    from SimpleHTTPServer import SimpleHTTPRequestHandler as SHRH
    from cgi import escape

import sys

import json
import cgi
from io import BytesIO
import io
import os
import traceback
import posixpath
import re
from collections import deque
import logging

import matplotlib
matplotlib.use('Agg')  # noqa

import matplotlib.pyplot as plt  # noqa
from pyrocko.plot import cake_plot  # noqa
from pyrocko import gf, util  # noqa
from pyrocko.util import quote, unquote  # noqa

logger = logging.getLogger('pyrocko.gf.server')

__version__ = '1.0'

store_id_pattern = gf.StringID.pattern[1:-1]


def enc(s):
    try:
        return s.encode('utf-8')
    except Exception:
        return s


def popall(self):
    # Preallocate the list to save memory resizing.
    r = len(self)*[None]
    for i in range(len(r)):
        r[i] = self.popleft()
    return r


class writewrapper(object):
    def __init__(self, d, blocksize=4096):
        self.blocksize = blocksize
        self.d = d

    def write(self, data):
        if self.blocksize in (None, -1):
            self.d.append(data)
        else:
            BS = self.blocksize
            xtra = 0
            if len(data) % BS:
                xtra = len(data) % BS + BS
            buf = self.d
            for i in range(0, len(data)-xtra, BS):
                buf.append(data[i:i+BS])
            if xtra:
                buf.append(data[-xtra:])


class RequestHandler(asynchat.async_chat, SHRH):

    server_version = 'Seismosizer/'+__version__
    protocol_version = 'HTTP/1.1'
    blocksize = 4096

    # In enabling the use of buffer objects by setting use_buffer to True,
    # any data block sent will remain in memory until it has actually been
    # sent.
    use_buffer = False

    def __init__(self, conn, addr, server):
        asynchat.async_chat.__init__(self, conn)
        self.client_address = addr
        self.connection = conn
        self.server = server
        self.opened = []
        # set the terminator : when it is received, this means that the
        # http request is complete ; control will be passed to
        # self.found_terminator
        self.set_terminator(b'\r\n\r\n')
        self.incoming = deque()
        self.outgoing = deque()
        self.rfile = None
        self.wfile = writewrapper(
            self.outgoing,
            -self.use_buffer or self.blocksize)
        self.found_terminator = self.handle_request_line
        self.request_version = 'HTTP/1.1'
        self.code = None
        # buffer the response and headers to avoid several calls to select()

    def update_b(self, fsize):
        if fsize > 1048576:
            self.use_buffer = True
            self.blocksize = 131072

    def collect_incoming_data(self, data):
        '''Collect the data arriving on the connexion'''
        if not data:
            self.ac_in_buffer = ''
            return
        self.incoming.append(data)

    def prepare_POST(self):
        '''Prepare to read the request body'''
        try:
            bytesToRead = int(self.headers.getheader('Content-length'))
        except AttributeError:
            bytesToRead = int(self.headers['Content-length'])
        # set terminator to length (will read bytesToRead bytes)
        self.set_terminator(bytesToRead)
        self.incoming.clear()
        # control will be passed to a new found_terminator
        self.found_terminator = self.handle_post_data

    def handle_post_data(self):
        '''Called when a POST request body has been read'''
        self.rfile = BytesIO(b''.join(popall(self.incoming)))
        self.rfile.seek(0)
        self.do_POST()

    def parse_request_url(self):
        # Check for query string in URL
        qspos = self.path.find('?')
        if qspos >= 0:
            self.body = cgi.parse_qs(self.path[qspos+1:], keep_blank_values=1)
            self.path = self.path[:qspos]
        else:
            self.body = {}

    def do_HEAD(self):
        '''Begins serving a HEAD request'''
        self.parse_request_url()
        f = self.send_head()
        if f:
            f.close()
        self.log_request(self.code)

    def do_GET(self):
        '''Begins serving a GET request'''
        self.parse_request_url()
        self.handle_data()

    def do_POST(self):
        '''Begins serving a POST request. The request data must be readable
        on a file-like object called self.rfile'''
        try:
            data = cgi.parse_header(self.headers.getheader('content-type'))
            length = int(self.headers.getheader('content-length', 0))
        except AttributeError:
            data = cgi.parse_header(self.headers.get('content-type'))
            length = int(self.headers.get('content-length', 0))

        ctype, pdict = data if data else (None, None)

        if ctype == 'multipart/form-data':
            self.body = cgi.parse_multipart(self.rfile, pdict)
        elif ctype == 'application/x-www-form-urlencoded':
            qs = self.rfile.read(length)
            self.body = cgi.parse_qs(qs, keep_blank_values=1)
        else:
            self.body = {}
        # self.handle_post_body()
        self.handle_data()

    def handle_close(self):
        for f in self.opened:
            if not f.closed:
                f.close()
        asynchat.async_chat.handle_close(self)

    def handle_data(self):
        '''Class to override'''

        f = self.send_head()
        if f:
            # do some special things with file objects so that we don't have
            # to read them all into memory at the same time...may leave a
            # file handle open for longer than is really desired, but it does
            # make it able to handle files of unlimited size.
            try:
                size = sys.getsizeof(f)
            except (AttributeError, io.UnsupportedOperation):
                size = len(f.getvalue())

            self.update_b(size)
            self.log_request(self.code, size)
            self.outgoing.append(f)
        else:
            self.log_request(self.code)
            # signal the end of this request
            self.outgoing.append(None)

    def handle_request_line(self):
        '''Called when the http request line and headers have been received'''
        # prepare attributes needed in parse_request()
        self.rfile = BytesIO(b''.join(popall(self.incoming)))
        self.rfile.seek(0)
        self.raw_requestline = self.rfile.readline()
        self.parse_request()

        if self.command in ['GET', 'HEAD']:
            # if method is GET or HEAD, call do_GET or do_HEAD and finish
            method = 'do_'+self.command
            if hasattr(self, method):
                getattr(self, method)()
        elif self.command == 'POST':
            # if method is POST, call prepare_POST, don't finish yet
            self.prepare_POST()
        else:
            self.send_error(501, 'Unsupported method (%s)' % self.command)

    def handle_error(self):
        try:
            traceback.print_exc(sys.stderr)
        except Exception:
            logger.error(
                'An error occurred and another one while printing the '
                'traceback. Please debug me...')

        self.close()

    def writable(self):
        return len(self.outgoing) and self.connected

    def handle_write(self):
        out = self.outgoing
        while len(out):
            a = out.popleft()

            a = enc(a)
            # handle end of request disconnection
            if a is None:
                # Some clients have issues with keep-alive connections, or
                # perhaps I implemented them wrong.

                # If the user is running a Python version < 2.4.1, there is a
                # bug with SimpleHTTPServer:
                #     http://python.org/sf/1097597
                # So we should be closing anyways, even though the client will
                # claim a partial download, so as to prevent hung-connections.
                # if self.close_connection:
                self.close()
                return

            # handle file objects
            elif hasattr(a, 'read'):
                _a, a = a, a.read(self.blocksize)
                if not len(a):
                    _a.close()
                    del _a
                    continue
                else:
                    out.appendleft(_a)  # noqa
                    break

            # handle string/buffer objects
            elif len(a):
                break
        else:
            # if we get here, the outgoing deque is empty
            return

        # if we get here, 'a' is a string or buffer object of length > 0
        try:
            num_sent = self.send(a)
            if num_sent < len(a):
                if not num_sent:
                    # this is probably overkill, but it can save the
                    # allocations of buffers when they are enabled
                    out.appendleft(a)
                elif self.use_buffer:
                    out.appendleft(buffer(a, num_sent))  # noqa
                else:
                    out.appendleft(a[num_sent:])

        except socket.error as why:
            if isinstance(why, str):
                self.log_error(why)
            elif isinstance(why, tuple) and isinstance(why[-1], str):
                self.log_error(why[-1])
            else:
                self.log_error(str(why))
            self.handle_error()

    def log(self, message):
        self.log_info(message)

    def log_info(self, message, type='info'):
        {
            'debug': logger.debug,
            'info': logger.info,
            'warning': logger.warning,
            'error': logger.error
        }.get(type, logger.info)(str(message))

    def log_message(self, format, *args):
        self.log_info('%s - - [%s] %s \"%s\" \"%s\"\n' % (
            self.address_string(),
            self.log_date_time_string(),
            format % args,
            self.headers.get('referer', ''),
            self.headers.get('user-agent', '')))

    def listdir(self, path):
        return os.listdir(path)

    def list_directory(self, path):
        '''Helper to produce a directory listing (absent index.html).

        Return value is either a file object, or None (indicating an
        error).  In either case, the headers are sent, making the
        interface the same as for send_head().

        '''
        try:
            list = self.listdir(path)
        except os.error:
            self.send_error(404, 'No permission to list directory')
            return None

        list.sort(key=lambda a: a.lower())
        f = BytesIO()
        displaypath = escape(unquote(self.path))
        f.write(enc('<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">'))
        f.write(enc('<html>\n<title>Directory listing for %s</title>\n'
                % displaypath))
        f.write(
            enc('<body>\n<h2>Directory listing for %s</h2>\n' % displaypath))
        f.write(enc('<hr>\n<ul>\n'))
        for name in list:
            fullname = os.path.join(path, name)
            displayname = linkname = name
            # Append / for directories or @ for symbolic links
            if os.path.isdir(fullname):
                displayname = name + '/'
                linkname = name + '/'
            if os.path.islink(fullname):
                displayname = name + '@'
                # Note: a link to a directory displays with @ and links with /
            f.write(enc('<li><a href="%s">%s</a>\n' %
                        (quote(linkname),
                         escape(displayname))))
        f.write(enc('</ul>\n<hr>\n</body>\n</html>\n'))
        length = f.tell()
        f.seek(0)
        encoding = sys.getfilesystemencoding()

        self.send_response(200, 'OK')
        self.send_header('Content-Length', str(length))
        self.send_header('Content-Type', 'text/html; charset=%s' % encoding)
        self.end_headers()

        return f

    def redirect(self, path):
        self.send_response(301)
        self.send_header('Location', path)
        self.end_headers()

    def send_head(self):
        '''Common code for GET and HEAD commands.

        This sends the response code and MIME headers.

        Return value is either a file object (which has to be copied
        to the outputfile by the caller unless the command was HEAD,
        and must be closed by the caller under all circumstances), or
        None, in which case the caller has nothing further to do.

        '''
        path = self.translate_path(self.path)
        if path is None:
            self.send_error(404, 'File not found')
            return None

        f = None
        if os.path.isdir(path):
            if not self.path.endswith('/'):
                # redirect browser - doing basically what apache does
                return self.redirect(self.path + '/')
            else:
                return self.list_directory(path)

        ctype = self.guess_type(path)
        try:
            # Always read in binary mode. Opening files in text mode may cause
            # newline translations, making the actual size of the content
            # transmitted *less* than the content-length!
            f = open(path, 'rb')
            self.opened.append(f)
        except IOError:
            self.send_error(404, 'File not found')
            return None
        fs = os.fstat(f.fileno())
        self.send_response(200, 'OK')
        self.send_header('Last-Modified', self.date_time_string(fs.st_mtime))
        self.send_header('Content-Length', str(fs[6]))
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Disposition', 'attachment')
        self.end_headers()
        return f


class SeismosizerHandler(RequestHandler):

    stores_path = '/gfws/static/stores/'
    api_path = '/gfws/api/'
    process_path = '/gfws/seismosizer/1/query'

    def send_head(self):
        S = self.stores_path
        P = self.process_path
        A = self.api_path
        for x in (S,):
            if re.match(r'^' + x[:-1] + '$', self.path):
                return self.redirect(x)

        if re.match(r'^' + S + store_id_pattern, self.path):
            return RequestHandler.send_head(self)

        elif re.match(r'^' + S + '$', self.path):
            return self.list_stores()

        elif re.match(r'^' + A + '$', self.path):
            return self.list_stores_json()

        elif re.match(r'^' + A + store_id_pattern + '$', self.path):
            return self.get_store_config()

        elif re.match(r'^' + A + store_id_pattern + '/profile$', self.path):
            return self.get_store_velocity_profile()

        elif re.match(r'^' + P + '$', self.path):
            return self.process()

        else:
            self.send_error(404, 'File not found')
            self.end_headers()
            return None

    def translate_path(self, path):
        path = path.split('?', 1)[0]
        path = path.split('#', 1)[0]
        path = posixpath.normpath(unquote(path))
        words = path.split('/')
        words = [_f for _f in words if _f]

        path = '/'
        if words[:3] == self.stores_path.split('/')[1:-1] and len(words) > 3:
            engine = self.server.engine
            if words[3] not in engine.get_store_ids():
                return None
            else:
                path = engine.get_store_dir(words[3])
                words = words[4:]
        else:
            return None

        for word in words:
            drive, word = os.path.splitdrive(word)
            head, word = os.path.split(word)
            if word in (os.curdir, os.pardir):
                continue
            path = os.path.join(path, word)

        return path

    def listdir(self, path):
        if path == self.stores_path:
            return list(self.server.engine.get_store_ids())
        else:
            return RequestHandler.listdir(self, path)

    def list_stores(self):
        '''Create listing of stores.'''
        from jinja2 import Template

        engine = self.server.engine

        store_ids = list(engine.get_store_ids())
        store_ids.sort(key=lambda x: x.lower())

        stores = [engine.get_store(store_id) for store_id in store_ids]

        templates = {
            'html': Template('''
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<html>
<title>{{ title }}</title>
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
</html>
'''.lstrip()),
            'text': Template('''
{% for store in stores %}{#
#}{{ store.config.id.ljust(25) }} {#
#}{{ store.config.short_type.center(5) }} {#
#}{{ store.config.short_extent.rjust(30) }} km {#
#}{{ "%10.2g"|format(store.config.sample_rate) }} Hz {#
#}{{ store.size_index_and_data_human.rjust(8) }}
{% endfor %}'''.lstrip())}

        format = self.body.get('format', ['html'])[0]
        if format not in ('html', 'text'):
            format = 'html'

        title = "Green's function stores listing"
        s = templates[format].render(stores=stores, title=title).encode('utf8')
        length = len(s)
        f = BytesIO(s)
        self.send_response(200, 'OK')
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(length))
        self.end_headers()
        return f

    def list_stores_json(self):
        engine = self.server.engine

        store_ids = list(engine.get_store_ids())
        store_ids.sort(key=lambda x: x.lower())

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
            'stores': [get_store_dict(engine.get_store(store_id))
                       for store_id in store_ids]
        }

        s = json.dumps(stores)
        length = len(s)
        f = BytesIO(s.encode('ascii'))
        self.send_response(200, 'OK')
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(length))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        return f

    def get_store_config(self):
        engine = self.server.engine

        store_ids = list(engine.get_store_ids())
        store_ids.sort(key=lambda x: x.lower())

        for match in re.finditer(r'/gfws/api/(' + store_id_pattern + ')',
                                 self.path):
            store_id = match.groups()[0]

        try:
            store = engine.get_store(store_id)
        except Exception:
            self.send_error(404)
            self.end_headers()
            return

        data = {}
        data['id'] = store_id
        data['config'] = str(store.config)

        s = json.dumps(data)
        length = len(s)
        f = BytesIO(s.encode('ascii'))
        self.send_response(200, 'OK')
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(length))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        return f

    def get_store_velocity_profile(self):
        engine = self.server.engine

        fig = plt.figure()
        axes = fig.gca()

        store_ids = list(engine.get_store_ids())
        store_ids.sort(key=lambda x: x.lower())

        for match in re.finditer(
                r'/gfws/api/(' + store_id_pattern + ')/profile', self.path):
            store_id = match.groups()[0]

        try:
            store = engine.get_store(store_id)
        except Exception:
            self.send_error(404)
            self.end_headers()
            return

        if store.config.earthmodel_1d is None:
            self.send_error(404)
            self.end_headers()
            return

        cake_plot.my_model_plot(store.config.earthmodel_1d, fig=axes.figure)

        f = BytesIO()
        fig.savefig(f, format='png')

        length = f.tell()
        self.send_response(200, 'OK')
        self.send_header('Content-Type', 'image/png;')
        self.send_header('Content-Length', str(length))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        f.seek(0)
        return f.read()

    def process(self):

        request = gf.load(string=self.body['request'][0])
        try:
            resp = self.server.engine.process(request=request)
        except (gf.BadRequest, gf.StoreError) as e:
            self.send_error(400, str(e))
            return

        f = BytesIO()
        resp.dump(stream=f)
        length = f.tell()

        f.seek(0)

        self.send_response(200, 'OK')
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(length))
        self.end_headers()
        return f

    def guess_type(self, path):
        bn = os.path.basename
        dn = os.path.dirname
        if bn(path) == 'config':
            return 'text/plain'

        if bn(dn(path)) == 'extra':
            return 'text/plain'

        else:
            return RequestHandler.guess_type(self, path) \
                or 'application/x-octet'


class Server(asyncore.dispatcher):
    def __init__(self, ip, port, handler, engine):
        self.ensure_uuids(engine)
        logger.info('starting Server at http://%s:%d', ip, port)

        asyncore.dispatcher.__init__(self)
        self.ip = ip
        self.port = port
        self.handler = handler
        asyncore.dispatcher.__init__(self)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)

        self.set_reuse_addr()
        self.bind((ip, port))
        self.engine = engine

        # Quoting the socket module documentation...
        # listen(backlog)
        #     Listen for connections made to the socket. The backlog argument
        #     specifies the maximum number of queued connections and should
        #     be at least 1; the maximum value is system-dependent (usually
        #     5).
        self.listen(5)

    @staticmethod
    def ensure_uuids(engine):
        logger.info('ensuring UUIDs of available stores')
        store_ids = list(engine.get_store_ids())
        for store_id in store_ids:
            store = engine.get_store(store_id)
            store.ensure_reference()

    def handle_accept(self):
        try:
            conn, addr = self.accept()
        except socket.error:
            self.log_info('warning: server accept() threw an exception',
                          'warning')
            return
        except TypeError:
            self.log_info('warning: server accept() threw EWOULDBLOCK',
                          'warning')
            return

        self.handler(conn, addr, self)

    def log(self, message):
        self.log_info(message)

    def handle_close(self):
        self.close()

    def log_info(self, message, type='info'):
        {
            'debug': logger.debug,
            'info': logger.info,
            'warning': logger.warning,
            'error': logger.error
        }.get(type, 'info')(str(message))


def run(ip, port, engine):
    s = Server(ip, port, SeismosizerHandler, engine)
    asyncore.loop()
    del s


if __name__ == '__main__':
    util.setup_logging('pyrocko.gf.server', 'info')
    port = 8085
    engine = gf.LocalEngine(store_superdirs=sys.argv[1:])
    run('127.0.0.1', port, engine)
