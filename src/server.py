
import asyncio
import logging
import signal
import time
import uuid
import threading

from tornado import web, autoreload

from pyrocko import info

logger = logging.getLogger('pyrocko.server')


g_server_info = None
g_shutdown = False
g_exception = None


class ServerError(Exception):
    pass


class RequestHandler(web.RequestHandler):
    def prepare(self):
        g_server_info['n_requests'] += 1
        self.set_header('Server', 'Pyrocko/1.0')


def get_address(host):
    if host == 'localhost':
        return 'localhost'

    elif host == 'ip4-localhost':
        return '127.0.0.1'

    elif host == 'ip6-localhost':
        return '::1'

    elif host == 'all':
        return ''

    elif host == 'ip4-all':
        return '0.0.0.0'

    elif host == 'ip6-all':
        return '::'

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
            raise ServerError(
                'Could not determine default external IP address to bind to.')

    else:
        return host


def describe_address(address):
    substitutions = {
        '': 'all available interfaces',
        '::': 'all available IPv6 interfaces',
        '0.0.0.0': 'all available IPv4 interfaces'}

    return substitutions.get(address, 'address "%s"' % address)


def url_address(address):
    if '' == address or '::' == address:
        return '[::1]'
    elif '0.0.0.0' == address:
        return '127.0.0.1'
    elif ':' in address:
        return '[%s]' % address
    else:
        return address


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


async def shutdown(sig=None, loop=None):
    global g_shutdown

    if sig is not None:
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

    if loop is None:
        loop = asyncio.get_event_loop()

    loop.stop()


async def fail(e):
    global g_exception
    g_exception = e
    await shutdown()


def stop():
    asyncio.create_task(shutdown())


async def serve(
        host='localhost',
        port=8000,
        handlers=[],
        open=False,
        debug=False):

    address = get_address(host)

    global g_server_info

    g_server_info = dict(
        run_id=str(uuid.uuid4()),
        n_requests=0,
        debug=debug,
        pyrocko_version=info.version,
    )

    if debug:
        g_server_info.update(
            address=address,
            port=port,
            pyrocko_long_version=info.long_version,
            pyrocko_src_path=info.src_path)

    app = web.Application(
        handlers,
        log_function=log_request,
        debug=debug,
    )

    logger.info(
        'Binding service to %s, port %i.',
        describe_address(address), port)

    try:
        http_server = app.listen(port, address)

    except OSError as e:
        if e.errno == 98:
            await fail(ServerError(
                'Address already in use, try specifying a different port '
                'number with --port INT.'))
            return

    url = 'http://%s:%d' % (url_address(address), port)
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


def run(
        host='localhost',
        port=2323,
        handlers=[],
        open=False,
        debug=False,
        page_path=None,
        page_matcher=r'/((?:css|js|images)/.*'
                     r'|index.html|site.webmanifest|favicon.ico|)'):

    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('Debug mode activated.')
        autoreload.add_reload_hook(lambda: time.sleep(2))

    if page_path:
        handlers = [(
            r'/((?:css|js|images)/.*'
            r'|index.html|site.webmanifest|favicon.ico|)',
            web.StaticFileHandler,
            dict(
                path=page_path,
                default_filename='index.html',
            ))] + handlers

        logger.debug('Serving static files from: %s', page_path)

    loop = asyncio.get_event_loop()

    if threading.current_thread() is threading.main_thread():
        # adding signal handlers only works on main thread
        signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
        for s in signals:
            loop.add_signal_handler(
                s, lambda s=s: asyncio.create_task(shutdown(s)))

    try:
        loop.create_task(
            serve(
                host=host,
                port=port,
                handlers=handlers,
                open=open,
                debug=debug))

        loop.run_forever()

    finally:
        loop.close()
        if g_exception is not None:
            raise g_exception

        logger.debug('Shutdown complete.')


def _add_cli_arguments(method, exclude=(), default_port=2323):

    if 'port' not in exclude:
        method(
            '--port',
            dest='port',
            type=int,
            default=default_port,
            metavar='INT',
            help='Set the port to bind to.')

    if 'host' not in exclude:
        method(
            '--host',
            dest='host',
            default='localhost',
            help='IP address to bind to, or ```public``` to bind to what '
                 'appears to be the public IPv4 address of the host in the '
                 'local network, or ```ip4-all```, ```ip6-all```, '
                 '```ip4-all``` to bind to all available interfaces of the '
                 'protocol family, or ```localhost``` (default) '
                 '```ip4-localhost``` or ```ip6-localhost```.')

    if 'open' not in exclude:
        method(
            '--open',
            dest='open',
            action='store_true',
            default=False,
            help='Open in web browser.')

    if 'page' not in exclude:
        method(
            '--page',
            dest='page_path',
            metavar='PATH',
            help='Serve custom pages from PATH.')

    if 'debug' not in exclude:
        method(
            '--debug',
            dest='debug',
            action='store_true',
            default=False,
            help='Activate debug mode. In debug mode, server tracebacks are '
                 'shown in the browser, verbosity level of the server logger '
                 'is increased, auto-restart on module change is activated, '
                 'and static pages are served from the Pyrocko sources '
                 'directory (if available and not overridden with ``--page``) '
                 'rather than from the installed files.')


def add_cli_arguments(parser, exclude=(), default_port=2323):
    _add_cli_arguments(
        parser.add_argument,
        exclude=exclude,
        default_port=default_port)


def add_cli_options(parser, exclude=(), default_port=2323):
    _add_cli_arguments(
        parser.add_option,
        exclude=exclude,
        default_port=default_port)
