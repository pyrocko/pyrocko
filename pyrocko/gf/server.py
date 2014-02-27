
"""
Simple async HTTP server

Based on this recipe:

    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/440665

which is based on this one:

    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/259148
"""

import asynchat
import asyncore
import socket
from SimpleHTTPServer import SimpleHTTPRequestHandler as SHRH
import sys
import cgi
from cStringIO import StringIO
import os
import traceback
import zlib
import posixpath
import urllib
import re
from collections import deque

from pyrocko import gf

__version__ = '1.0'


def popall(self):
    #Preallocate the list to save memory resizing.
    r = len(self)*[None]
    for i in xrange(len(r)):
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
            for i in xrange(0, len(data)-xtra, BS):
                buf.append(data[i:i+BS])
            if xtra:
                buf.append(data[-xtra:])


class ParseHeaders(dict):
    '''Replacement for the deprecated mimetools.Message class
        Works like a dictionary with case-insensitive keys'''

    def __init__(self, infile, *args):
        self._ci_dict = {}
        lines = infile.readlines()
        for line in lines:
            k, v = line.split(":", 1)
            self._ci_dict[k.lower()] = self[k] = v.strip()
        self.headers = self.keys()

    def getheader(self, key, default=""):
        return self._ci_dict.get(key.lower(), default)

    def get(self, key, default=""):
        return self._ci_dict.get(key.lower(), default)


class RequestHandler(asynchat.async_chat, SHRH):

    server_version = 'Seismosizer/'+__version__
    protocol_version = "HTTP/1.1"
    MessageClass = ParseHeaders
    blocksize = 4096

    #In enabling the use of buffer objects by setting use_buffer to True,
    #any data block sent will remain in memory until it has actually been
    #sent.
    use_buffer = False

    def __init__(self, conn, addr, server):
        asynchat.async_chat.__init__(self, conn)
        self.client_address = addr
        self.connection = conn
        self.server = server
        # set the terminator : when it is received, this means that the
        # http request is complete ; control will be passed to
        # self.found_terminator
        self.set_terminator('\r\n\r\n')
        self.incoming = deque()
        self.outgoing = deque()
        self.rfile = None
        self.wfile = writewrapper(
            self.outgoing,
            -self.use_buffer or self.blocksize)
        self.found_terminator = self.handle_request_line
        self.request_version = "HTTP/1.1"
        self.code = None
        # buffer the response and headers to avoid several calls to select()

    def update_b(self, fsize):
        if fsize > 1048576:
            self.use_buffer = True
            self.blocksize = 131072

    def collect_incoming_data(self, data):
        """Collect the data arriving on the connexion"""
        if not data:
            self.ac_in_buffer = ""
            return
        self.incoming.append(data)

    def prepare_POST(self):
        """Prepare to read the request body"""
        bytesToRead = int(self.headers.getheader('content-length'))
        # set terminator to length (will read bytesToRead bytes)
        self.set_terminator(bytesToRead)
        self.incoming.clear()
        # control will be passed to a new found_terminator
        self.found_terminator = self.handle_post_data

    def handle_post_data(self):
        """Called when a POST request body has been read"""
        self.rfile = StringIO(''.join(popall(self.incoming)))
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
        """Begins serving a HEAD request"""
        self.parse_request_url()
        f = self.send_head()
        if f:
            f.close()
        self.log_request(self.code)

    def do_GET(self):
        """Begins serving a GET request"""
        self.parse_request_url()
        self.handle_data()

    def do_POST(self):
        """Begins serving a POST request. The request data must be readable
        on a file-like object called self.rfile"""
        ctype, pdict = cgi.parse_header(self.headers.getheader('content-type'))
        length = int(self.headers.getheader('content-length'))
        if ctype == 'multipart/form-data':
            self.body = cgi.parse_multipart(self.rfile, pdict)
        elif ctype == 'application/x-www-form-urlencoded':
            qs = self.rfile.read(length)
            self.body = cgi.parse_qs(qs, keep_blank_values=1)
        else:
            self.body = {}
        #self.handle_post_body()
        self.handle_data()

    def handle_data(self):
        """Class to override"""

        f = self.send_head()
        if f:
            # do some special things with file objects so that we don't have
            # to read them all into memory at the same time...may leave a
            # file handle open for longer than is really desired, but it does
            # make it able to handle files of unlimited size.
            try:
                size = os.fstat(f.fileno())[6]
            except AttributeError:
                size = len(f.getvalue())
            self.update_b(size)
            self.log_request(self.code, size)
            self.outgoing.append(f)
        else:
            self.log_request(self.code)

        # signal the end of this request
        self.outgoing.append(None)

    def handle_request_line(self):
        """Called when the http request line and headers have been received"""
        # prepare attributes needed in parse_request()
        self.rfile = StringIO(''.join(popall(self.incoming)))
        self.rfile.seek(0)
        self.raw_requestline = self.rfile.readline()
        self.parse_request()

        if self.command in ['GET', 'HEAD']:
            # if method is GET or HEAD, call do_GET or do_HEAD and finish
            method = "do_"+self.command
            if hasattr(self, method):
                getattr(self, method)()
        elif self.command == "POST":
            # if method is POST, call prepare_POST, don't finish yet
            self.prepare_POST()
        else:
            self.send_error(501, "Unsupported method (%s)" % self.command)

    def end_headers(self):
        """Send the blank line ending the MIME headers, send the buffered
        response and headers on the connection"""
        if self.request_version != 'HTTP/0.9':
            self.outgoing.append("\r\n")

    def handle_error(self):
        traceback.print_exc(sys.stderr)
        self.close()

    def writable(self):
        return len(self.outgoing) and self.connected

    def handle_write(self):
        O = self.outgoing
        while len(O):
            a = O.popleft()
            #handle end of request disconnection
            if a is None:
                #Some clients have issues with keep-alive connections, or
                #perhaps I implemented them wrong.

                #If the user is running a Python version < 2.4.1, there is a
                #bug with SimpleHTTPServer:
                #    http://python.org/sf/1097597
                #So we should be closing anyways, even though the client will
                #claim a partial download, so as to prevent hung-connections.
                #if self.close_connection:
                self.close()
                return
            #handle file objects
            elif hasattr(a, 'read'):
                _a, a = a, a.read(self.blocksize)
                if not a:
                    _a.close()
                    del _a
                    continue
                else:
                    O.appendleft(_a)  # noqa
                    break

            #handle string/buffer objects
            elif len(a):
                break
        else:
            #if we get here, the outgoing deque is empty
            return

        #if we get here, 'a' is a string or buffer object of length > 0
        try:
            num_sent = self.send(a)
            if num_sent < len(a):
                if not num_sent:
                    # this is probably overkill, but it can save the
                    # allocations of buffers when they are enabled
                    O.appendleft(a)
                elif self.use_buffer:
                    O.appendleft(buffer(a, num_sent))
                else:
                    O.appendleft(a[num_sent:])

        except socket.error, why:
            if isinstance(why, basestring):
                self.log_error(why)
            elif isinstance(why, tuple) and isinstance(why[-1], basestring):
                self.log_error(why[-1])
            else:
                self.log_error(str(why))
            self.handle_error()

    def send_response(self, code, message=None):
        if self.code:
            return
        self.code = code
        if message is None:
            if code in self.responses:
                message = self.responses[code][0]
            else:
                message = ''
        if self.request_version != 'HTTP/0.9':
            self.wfile.write("%s %d %s\r\n" %
                             (self.protocol_version, code, message))
            # print (self.protocol_version, code, message)
        self.send_header('Server', self.version_string())
        self.send_header('Date', self.date_time_string())

    def log_message(self, format, *args):
        sys.stderr.write("%s - - [%s] %s \"%s\" \"%s\"\n" %
                         (self.address_string(),
                          self.log_date_time_string(),
                          format % args,
                          self.headers.get('referer', ''),
                          self.headers.get('user-agent', '')
                          ))

    def listdir(self, path):
        return os.listdir(path)

    def list_directory(self, path):
        """Helper to produce a directory listing (absent index.html).

        Return value is either a file object, or None (indicating an
        error).  In either case, the headers are sent, making the
        interface the same as for send_head().

        """
        try:
            list = self.listdir(path)
        except os.error:
            self.send_error(404, "No permission to list directory")
            return None

        list.sort(key=lambda a: a.lower())
        f = StringIO()
        displaypath = cgi.escape(urllib.unquote(self.path))
        f.write('<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">')
        f.write("<html>\n<title>Directory listing for %s</title>\n" % displaypath)
        f.write("<body>\n<h2>Directory listing for %s</h2>\n" % displaypath)
        f.write("<hr>\n<ul>\n")
        for name in list:
            fullname = os.path.join(path, name)
            displayname = linkname = name
            # Append / for directories or @ for symbolic links
            if os.path.isdir(fullname):
                displayname = name + "/"
                linkname = name + "/"
            if os.path.islink(fullname):
                displayname = name + "@"
                # Note: a link to a directory displays with @ and links with /
            f.write('<li><a href="%s">%s</a>\n'
                    % (urllib.quote(linkname), cgi.escape(displayname)))
        f.write("</ul>\n<hr>\n</body>\n</html>\n")
        length = f.tell()
        f.seek(0)
        self.send_response(200)
        encoding = sys.getfilesystemencoding()
        self.send_header("Content-type", "text/html; charset=%s" % encoding)
        self.send_header("Content-Length", str(length))
        self.end_headers()
        return f

    def redirect(self, path):
        self.send_response(301)
        self.send_header("Location", path)
        self.end_headers()

    def send_head(self):
        """Common code for GET and HEAD commands.

        This sends the response code and MIME headers.

        Return value is either a file object (which has to be copied
        to the outputfile by the caller unless the command was HEAD,
        and must be closed by the caller under all circumstances), or
        None, in which case the caller has nothing further to do.

        """
        path = self.translate_path(self.path)
        if path is None:
            self.send_error(404, "File not found")
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
        except IOError:
            self.send_error(404, "File not found")
            return None

        self.send_response(200)
        self.send_header("Content-type", ctype)
        fs = os.fstat(f.fileno())
        self.send_header("Content-Length", str(fs[6]))
        self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.end_headers()
        return f

class SeismosizerHandler(RequestHandler):

    stores_path = '/gfws/static/stores/'

    def send_head(self):
        S = self.stores_path
        if re.match(r'^' + S + gf.StringID.pattern[1:-1], self.path):
            return RequestHandler.send_head(self)

        elif re.match(r'^' + S[:-1] +'$', self.path):
            return self.redirect(S)

        elif re.match(r'^' + S + '$', self.path):
            return self.list_stores()

        else:
            self.send_error(404, "File not found")
            return None
    
    def translate_path(self, path):
        path = path.split('?',1)[0]
        path = path.split('#',1)[0]
        path = posixpath.normpath(urllib.unquote(path))
        words = path.split('/')
        words = filter(None, words)
        
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
            if word in (os.curdir, os.pardir): continue
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

        stores = [ engine.get_store(store_id) for store_id in store_ids ]

        templates = {
            'html': Template(
'''<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
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
'''),
            'text': Template(
'''{% for store in stores %}{#
#}{{ store.config.id.ljust(25) }} {#
#}{{ store.config.short_type.center(5) }} {#
#}{{ store.config.short_extent.rjust(30) }} km {#
#}{{ "%10.2g"|format(store.config.sample_rate) }} Hz {#
#}{{ store.size_index_and_data_human.rjust(8) }}
{% endfor %}''')}

        
        format = self.body.get('format', ['html'])[0]
        if format not in ('html', 'text'):
            format = 'html'


        title = "Green's function stores listing"
        s = templates[format].render(stores=stores, title=title).encode('utf8')
        length = len(s)
        f = StringIO(s)
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(length))
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
            RequestHandler.guess_type(self, path)

class Server(asyncore.dispatcher):
    def __init__(self, ip, port, handler, engine):
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


def run(ip, port, engine):
    s = Server(ip, port, SeismosizerHandler, engine)
    asyncore.loop()

if __name__ == '__main__':
    engine = gf.LocalEngine(store_superdirs=sys.argv[1:])
    run_server('', 8080, engine)

