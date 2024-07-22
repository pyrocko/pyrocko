# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Implementation of :app:`squirrel service`.
'''

import asyncio
import json
import logging
import os
import signal
import time
import uuid
from datetime import datetime

from tornado import web, autoreload

from pyrocko import info
from pyrocko import util
from pyrocko import squirrel as squirrel_module
from pyrocko.squirrel import model
from pyrocko.guts import Object
from pyrocko.squirrel.error import ToolError, SquirrelError

logger = logging.getLogger('psq.cli.service')

headline = 'Fire up a simple web UI.'


def str_choice(s, choices):
    s = str(s)
    if s not in choices:
        raise ValueError(
            'Invalid argument: %s. Choices: %s' % (
                s, ', '.join(choices)))

    return s


def optional_time(x):
    return util.str_to_time_fillup(x) if x is not None else None


def make_subparser(subparsers):
    return subparsers.add_parser(
        'service',
        help=headline,
        description=headline + '''


''')


def setup(parser):
    parser.add_squirrel_selection_arguments()

    parser.add_argument(
        '--port',
        dest='port',
        type=int,
        default=2323,
        metavar='INT',
        help='Set the port to bind to.')

    parser.add_argument(
        '--host',
        dest='host',
        default='localhost',
        help='IP address to bind to, or ```public``` to bind to what appears '
             'to be the public IP of the host in the local network, '
             'or ```all``` to bind to all available interfaces, or '
             '```localhost``` (default).')

    parser.add_argument(
        '--open',
        dest='open',
        action='store_true',
        default=False,
        help='Open in web browser.')

    parser.add_argument(
        '--debug',
        dest='debug',
        action='store_true',
        default=False,
        help='Activate debug mode.')


class GutsJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Object):
            d = dict(
                (name, val) for (name, val) in o.T.inamevals_to_save(o))
            d['_T'] = o.T.tagname
            return d

        elif isinstance(o, datetime):
            return o.isoformat() + 'Z'

        else:
            return json.JSONEncoder.default(self, o)


class SquirrelHandler(web.RequestHandler):
    def __init__(self, *args, squirrel=None, server_info=None, **kwargs):
        web.RequestHandler.__init__(self, *args, **kwargs)
        self._squirrel = squirrel
        self._server_info = server_info

    async def get(self, method_name):
        if method_name == 'heartbeat':
            logger.info('Client connected.')
            self.set_header('Content-Type', 'application/octet-stream')
            self.set_header('X-Accel-Buffering', 'no')
            await self.flush()
            try:
                time_start = time.time()
                while True:
                    if g_shutdown:
                        break

                    self.write(
                        json.dumps(dict(
                            time_start=time_start,
                            time_now=time.time())))

                    try:
                        await self.flush()
                        await asyncio.sleep(1)
                    except (asyncio.exceptions.CancelledError, Exception):
                        break

            finally:
                logger.info('Client disconnected.')
                self.finish()
        else:
            raise web.HTTPError(
                400, reason='Invalid method: %s' % method_name)

    def post(self, method_name):
        self._server_info['n_requests'] += 1
        method = getattr(self, 'squirrel_' + method_name, None)
        if method is None:
            raise web.HTTPError(
                400, reason='Invalid method: %s' % method_name)
        else:
            try:
                print(self.request.body)
                if self.request.body:
                    parameters = json.loads(self.request.body)
                    if not isinstance(parameters, dict):
                        raise ValueError('Mapping required.')
                else:
                    parameters = {}

            except Exception as e:
                raise web.HTTPError(
                    400, reason='Bad request: %s' % str(e))

            try:
                raw = method(parameters)
            except SquirrelError as e:
                raise web.HTTPError(
                    400, reason='Squirrel error: %s' + str(e))

            self.set_header('Content-Type', 'application/json')
            self.write(json.dumps(raw, cls=GutsJSONEncoder))

    def get_cleaned(self, names, parameters):
        print(parameters)
        if isinstance(names, str):
            names = names.split()

        clean = {
             'kind': lambda x: str_choice(x, model.g_content_kinds),
             'tmin': lambda x: optional_time(x),
             'tmax': lambda x: optional_time(x)}

        try:
            return [clean[name](parameters.get(name, None)) for name in names]
        except Exception as e:
            raise web.HTTPError(400, 'Bad request: %s' % str(e))

    def squirrel_server_info(self, parameters):
        return self._server_info

    def squirrel_get_events(self, parameters):
        return self._squirrel.get_events()

    def squirrel_get_sensors(self, parameters):
        return self._squirrel.get_sensors()

    def squirrel_get_coverage(self, parameters):
        kind, tmin, tmax = self.get_cleaned('kind tmin tmax', parameters)
        return self._squirrel.get_coverage(kind, tmin=tmin, tmax=tmax)

    def squirrel_get_codes(self, parameters):
        kind, = self.get_cleaned('kind', parameters)
        return [c.safe_str for c in self._squirrel.get_codes(kind=kind)]


def get_ip(host):
    if host == 'localhost':
        return '::1'

    elif host == 'public':
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('4.4.4.4', 80))
            try:
                return s.getsockname()[0]
            finally:
                s.close()

        except Exception:
            ToolError(
                'Could not determine default external IP address to bind to.')

    elif host == 'all':
        return ''
    else:
        return host


def describe_ip(ip):
    ip_describe = {
        '': 'all available interfaces',
        '::': 'all available IPv6 interfaces',
        '0.0.0.0': 'all available IPv4 interfaces'}

    return ip_describe.get(ip, 'ip "%s"' % ip)


def url_ip(ip):
    if '' == ip:
        return '[::1]'
    elif '0.0.0.0' == ip:
        return '127.0.0.1'
    elif ':' in ip:
        return '[%s]' % ip
    else:
        return ip


def log_request(handler):
    if handler.get_status() < 400:
        log_method = logger.debug
    elif handler.get_status() < 500:
        log_method = logger.warning
    else:
        log_method = logger.error
    request_time = 1000.0 * handler.request.request_time()
    log_method(
        "%d %s %.2fms",
        handler.get_status(),
        handler._request_summary(),
        request_time,
    )


async def serve(
        squirrel,
        host='localhost',
        port=8000,
        open=False,
        debug=False,
        static_path=None):

    ip = get_ip(host)

    if static_path is None:
        static_path = os.path.join(
            os.path.split(squirrel_module.__file__)[0],
            'service',
            'page')

    server_info = dict(
        run_id=str(uuid.uuid4()),
        n_requests=0,
        debug=debug,
        pyrocko_version=info.version,
    )

    if debug:
        server_info.update(
            ip=ip,
            port=port,
            static_path=static_path,
            pyrocko_long_version=info.long_version,
            pyrocko_src_path=info.src_path)

    app = web.Application(
        [
            (
                r'/((?:css|js|images)/.*'
                r'|index.html|site.webmanifest|favicon.ico|)',
                web.StaticFileHandler,
                dict(
                    path=static_path,
                    default_filename='index.html'
                )
            ),
            (
                r'/squirrel/([a-z0-9_]+)', SquirrelHandler,
                dict(
                    squirrel=squirrel,
                    server_info=server_info,
                )
            ),
        ],
        log_function=log_request,
        debug=debug,
    )

    logger.info(
        'Binding service to %s, port %i.',
        describe_ip(ip), port)

    http_server = app.listen(port, ip)

    url = 'http://%s:%d' % (url_ip(ip), port)
    if open:
        import webbrowser
        logger.info('Pointing browser to: %s', url)
        webbrowser.open(url)
    else:
        logger.info('Point browser to: %s', url)

    while True:
        try:
            await asyncio.sleep(3)
        finally:
            if g_shutdown:
                logger.info('Shutting down service.')
                http_server.stop()
                await http_server.close_all_connections()
                break


g_shutdown = False


async def shutdown(sig, loop):
    global g_shutdown

    logger.debug(
        'Received exit signal %s.', sig.name)

    g_shutdown = True

    tasks = [
        t for t in asyncio.all_tasks()
        if t is not asyncio.current_task()]

    [task.cancel() for task in tasks]

    logger.debug(
        'Cancelling %s outstanding task%s.',
        len(tasks), '' if len(tasks) == 1 else 's')

    await asyncio.gather(*tasks)

    logger.debug(
        'Done waiting for tasks to finish.')

    loop.stop()


def run(parser, args):
    static_path = None
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('Debug mode activated.')
        autoreload.add_reload_hook(lambda: time.sleep(2))
        dev_static_path = os.path.join('src', 'squirrel', 'service', 'page')
        if os.path.exists(dev_static_path):
            static_path = dev_static_path

    squirrel = args.make_squirrel()

    loop = asyncio.get_event_loop()
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(
            s, lambda s=s: asyncio.create_task(shutdown(s, loop)))

    try:
        loop.create_task(
            serve(
                squirrel,
                host=args.host,
                port=args.port,
                open=args.open,
                debug=args.debug,
                static_path=static_path))

        loop.run_forever()

    finally:
        loop.close()
        logger.debug('Shutdown complete.')
