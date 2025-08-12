# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Web server component for Squirrel web services.
'''

import concurrent
import re
import asyncio
import json
import logging
import os
import signal
import time
import uuid
import base64
from datetime import datetime
from io import BytesIO

import numpy as num
from matplotlib import pyplot as plt
import matplotlib as mpl
from tornado import web, autoreload

from pyrocko import info
from pyrocko import util
from pyrocko import squirrel as squirrel_module
from pyrocko.squirrel import model
from pyrocko import guts
from pyrocko.squirrel.error import ToolError, SquirrelError
from pyrocko.squirrel import operators as ops
from pyrocko import moment_tensor as pmt
from pyrocko.plot import beachball
from pyrocko.color import Color, g_pyrocko_color_cycle_base

logger = logging.getLogger('psq.service.server')


def str_choice(s, choices):
    s = str(s)
    if s not in choices:
        raise ValueError(
            'Invalid argument: %s. Choices: %s' % (
                s, ', '.join(choices)))

    return s


def to_codes_list(xs):
    if not isinstance(xs, list) or not all(isinstance(x, str) for x in xs):
        raise ValueError('List of strings required.')

    return [model.to_codes_guess(x.strip()) for x in xs]


def get_parameters_dict(body):
    if not body:
        return {}

    parameters = json.loads(body)
    if not isinstance(parameters, dict):
        raise ValueError('Mapping required.')

    return parameters


class GutsJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, guts.Object):
            d = dict(
                (name, val) for (name, val) in o.T.inamevals_to_save(o))
            d['_T'] = o.T.tagname
            return d

        elif isinstance(o, datetime):
            return o.isoformat() + 'Z'

        elif isinstance(o, num.int64):
            return int(o)

        else:
            return json.JSONEncoder.default(self, o)


class SquirrelRequestHandler(web.RequestHandler):

    def initialize(self, squirrel=None, server_info=None):
        self._squirrel = squirrel
        self._server_info = server_info

    def prepare(self):
        self._server_info['n_requests'] += 1

    def get_cleaned(self, names, parameters):
        if isinstance(names, str):
            names = names.split()

        clean = {
            'kind': lambda x: str_choice(x, model.g_content_kinds),
            'tmin': util.str_to_time_fillup,
            'tmax': util.str_to_time_fillup,
            'codes': to_codes_list,
            'ymin': float,
            'ymax': float,
            'nx': int,
            'ny': int,
            'overview_method': lambda x: str_choice(x, ['mean', 'min', 'max']),
        }

        def clean_or_none(f, x):
            return f(x) if x is not None else None

        try:
            return [
                clean_or_none(clean[name], parameters.get(name, None))
                for name in names]

        except Exception as e:
            raise web.HTTPError(400, 'Bad request: %s' % str(e))

    def post(self, method_name, extra=()):
        method = getattr(self, 'p_' + method_name, None)
        if method is None:
            raise web.HTTPError(
                400, reason='Invalid method: %s' % method_name)
        else:
            try:
                parameters = get_parameters_dict(self.request.body)

            except Exception as e:
                raise web.HTTPError(
                    400, reason='Bad request: %s' % str(e))

            try:
                raw = method(parameters, *extra)
            except SquirrelError as e:
                raise web.HTTPError(
                    400, reason='Squirrel error: %s' + str(e))

            self.set_header('Content-Type', 'application/json')
            self.write(json.dumps(raw, cls=GutsJSONEncoder))


class SquirrelHeartbeatHandler(SquirrelRequestHandler):

    async def get(self):
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
                        time_now=time.time(),
                        server_info=self._server_info)))

                try:
                    await self.flush()
                    await asyncio.sleep(1)
                except (asyncio.exceptions.CancelledError, Exception):
                    break

        finally:
            logger.info('Client disconnected.')
            self.finish()


class SquirrelInfoHandler(SquirrelRequestHandler):
    def p_server(self, parameters):
        return self._server_info


class SquirrelRawHandler(SquirrelRequestHandler):

    def p_get_codes(self, parameters):
        kind, = self.get_cleaned('kind', parameters)
        return [c.safe_str for c in self._squirrel.get_codes(kind=kind)]

    def p_get_time_span(self, parameters, gate):
        kind, = self.get_cleaned('kind', parameters)
        return TimeSpan(
            *self._squirrel.get_time_span(
                kinds=[kind],
                dummy_limits=False))

    def p_get_events(self, parameters):
        tmin, tmax = self.get_cleaned('tmin tmax', parameters)
        return self._squirrel.get_events()

    def p_get_channels(self, parameters):
        tmin, tmax, codes = self.get_cleaned('tmin tmax codes', parameters)
        return self._squirrel.get_channels(tmin=tmin, tmax=tmax, codes=codes)

    def p_get_sensors(self, parameters):
        tmin, tmax, codes = self.get_cleaned('tmin tmax codes', parameters)
        return self._squirrel.get_sensors(tmin=tmin, tmax=tmax, codes=codes)

    def p_get_responses(self, parameters):
        tmin, tmax, codes = self.get_cleaned('tmin tmax codes', parameters)
        return self._squirrel.get_responses(tmin=tmin, tmax=tmax, codes=codes)

    def p_get_coverage(self, parameters):
        kind, tmin, tmax = self.get_cleaned('kind tmin tmax', parameters)
        return self._squirrel.get_coverage(kind, tmin=tmin, tmax=tmax)


class ScaleChoice(guts.StringChoice):
    choices = ['lin', 'log']


class CarpetImage(guts.Object):
    codes = model.CodesNSLCE.T()
    shape = guts.Tuple.T(2, guts.Int.T())
    tmin = guts.Timestamp.T()
    tmax = guts.Timestamp.T()
    ymin = guts.Float.T()
    ymax = guts.Float.T()
    fscale = ScaleChoice.T()
    overview_method = guts.String.T(optional=True)
    image_data_base64 = guts.String.T()

    @property
    def summary(self):
        return 'CarpetImage, %s, (%i, %i), %s - %s, %s' % (
            str(self.codes),
            self.shape[0],
            self.shape[1],
            util.time_to_str(self.tmin),
            util.time_to_str(self.tmax),
            util.human_bytesize(len(self.image_data_base64)))


class TimeSpan(guts.Object):
    tmin = model.Timestamp.T(optional=True)
    tmax = model.Timestamp.T(optional=True)

    def __init__(self, *args, **kwargs):
        if args:
            tmin, tmax = args
            kwargs['tmin'] = tmin
            kwargs['tmax'] = tmax

        guts.Object.__init__(self, **kwargs)


def drop_resolution(codes):
    return codes.replace(
        extra=re.sub(r'-L(min|max|mean)\d\d$', '', codes.extra))


def drop_resolution_codes(codes_list):
    return [
        codes for codes in codes_list
        if not re.match(r'-L(min|max|mean)\d\d$', codes.extra)]


class Gate(guts.Object):

    tmin = guts.Timestamp.T(optional=True)
    tmax = guts.Timestamp.T(optional=True)
    operators = guts.List.T(ops.Operator.T())

    def post_init(self):
        self._outlet = None

    @classmethod
    def from_query_arguments(cls, codes=None, tmin=None, tmax=None, time=None):
        operators = []

        # operators.append(
        #     ops.MultiSpectrogramOperator(
        #         filtering=ops.CodesPatternFiltering(codes=codes),
        #         windowing=ops.Pow2Windowing(
        #             nblock=2**10,
        #             nlevels=3,
        #             weighting_exponent=4)))

        return cls(
            tmin=tmin,
            tmax=tmax,
            operators=operators)

    def set_squirrel(self, squirrel):
        source = squirrel
        for operator in self.operators:
            operator.set_input(source)
            source = operator

        self._outlet = source

    def get_time_span(self, *args, **kwargs):
        return self._outlet.get_time_span(*args, **kwargs)

    def get_codes(self, *args, **kwargs):
        return drop_resolution_codes(self._outlet.get_codes(*args, **kwargs))

    def get_channels(self, *args, **kwargs):
        return self._outlet.get_channels(*args, **kwargs)

    def get_sensors(self, *args, **kwargs):
        return self._outlet.get_sensors(*args, **kwargs)

    def get_responses(self, *args, **kwargs):
        return self._outlet.get_responses(*args, **kwargs)

    def get_events(self, *args, **kwargs):
        return self._outlet.get_events(*args, **kwargs)

    def get_coverage(self, *args, **kwargs):
        kwargs['codes'] = '*.*.*.*.'
        return self._outlet.get_coverage(*args, **kwargs)

    def get_carpet_images(
            self,
            *args,
            ymin=None,
            ymax=None,
            nx=6000,
            ny=100,
            overview_method='mean',
            format='webp',
            **kwargs):

        def carpet_to_image(carpet):
            times_this = []
            times_this.append(time.time())
            overview_method = carpet.meta['overview_method']
            try:
                carpet = carpet.resample_band(ymin, ymax, ny)
            except ValueError as e:
                logger.error('Cannot create carpet: %s', str(e))
                return None, None

            if carpet.nsamples == 0 or carpet.ncomponents == 0:
                return None, None

            vmin, vmax = carpet.stats.min, carpet.stats.max
            if vmin == vmax:
                vmin -= 0.5
                vmax += 0.5

            rgb = (num.round(mpl.colormaps['inferno'](
                num.linspace(0., 1., 256)) * 255.)).astype(num.uint8)[:, :3]

            image_data = num.zeros(
                carpet.data.shape + (4,), dtype=num.uint8)

            ok = num.isfinite(carpet.data)
            values = num.zeros(carpet.data.shape)
            values[ok] = num.clip(carpet.data[ok], vmin, vmax)
            values[ok] -= vmax
            values[ok] *= 255.0 / (vmin - vmax)

            # image_data[::-1, :, :3] \
            #     = values.astype(num.uint8)[:, :, num.newaxis]
            image_data[::-1, :, :3] \
                = rgb[values.astype(num.uint8)]
            ok_alpha = num.array([[[False, False, False, True]]], dtype=bool)
            ok3 = num.logical_and(ok[:, :, num.newaxis], ok_alpha)
            image_data[ok3[::-1, :, :]] = 255

            times_this.append(time.time())

            from PIL import Image
            im = Image.fromarray(image_data, mode='RGBA')
            buffer = BytesIO()
            im.save(buffer, format=format)

            times_this.append(time.time())

            image = CarpetImage(
                codes=drop_resolution(carpet.codes),
                tmin=carpet.times[0],
                tmax=carpet.times[-1],
                ymin=carpet.component_axes['frequency'][0],
                ymax=carpet.component_axes['frequency'][-1],
                fscale='log',
                shape=carpet.data.shape,
                overview_method=overview_method,
                image_data_base64='data:image/%s;base64,%s' % (
                    format,
                    base64.b64encode(buffer.getvalue()).decode('ascii')))

            times_this.append(time.time())
            return image, times_this

        codes = kwargs.pop('codes', None)
        if codes is not None and not isinstance(codes, list):
            codes = [codes]

        if overview_method is not None:
            if codes is None:
                codes = [
                    model.CodesNSLCE('*.*.*.*.'),
                    model.CodesNSLCE('*.*.*.*.-L%s??' % overview_method)]

            else:
                codes_overview = [
                    v.replace(extra='-L%s??' % overview_method)
                    for v in codes
                    if v is not None]

                codes.extend(codes_overview)

        t0 = time.time()

        carpets = self._outlet.get_carpets(
            *args, **kwargs, codes=codes, nsamples_limit=nx)

        for carpet in carpets:
            carpet.meta['overview_method'] = overview_method

        images = []
        times = []

        t1 = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            for (image, times_this) in executor.map(carpet_to_image, carpets):
                if image is not None:
                    images.append(image)
                    times.append(times_this)

        t2 = time.time()

        if times:
            times = num.array(times)
            dtimes = num.diff(num.sum(times, 0))
            logger.debug('Processing costs: %i %s %i %.1f' % (
                int((t1 - t0)*1000.),
                (dtimes * 1000.).astype(int),
                int((t2 - t1)*1000.),
                num.sum(dtimes) / (t2 - t1)))

        return images

    def advance_accessor(self, accessor_id='default', cache_id=None):
        self._outlet.advance_accessor(accessor_id, cache_id)


g_gates = {}


class SquirrelGatesHandler(SquirrelRequestHandler):
    icurrent = 0

    def next_name(self):
        while True:
            candidate = '%i' % self.icurrent
            if '%i' % self.icurrent not in g_gates:
                return candidate
            self.icurrent += 1

    def get(self, name=None):
        if not name:
            self.set_header('Content-Type', 'application/json')
            self.write(json.dumps(sorted(g_gates.keys())))

    def post(self, name=None):
        if not name:
            name = self.next_name()

        parameters = get_parameters_dict(self.request.body)
        parameters
        g_gates[name] = Gate()

        self.set_header('Content-Type', 'application/json')
        self.write(json.dumps(name))


class SquirrelGateHandler(SquirrelRequestHandler):

    def post(self, gate_name, method_name):
        try:
            gate = g_gates[gate_name]
        except KeyError:
            raise web.HTTPError(
                400, reason='Squirrel error: no such gate: %s' % gate_name)

        SquirrelRequestHandler.post(self, method_name, extra=(gate,))

    def p_get_codes(self, parameters, gate):
        kind, = self.get_cleaned('kind', parameters)
        return [c.safe_str for c in gate.get_codes(kind=kind)]

    def p_get_time_span(self, parameters, gate):
        kind, = self.get_cleaned('kind', parameters)
        return TimeSpan(*gate.get_time_span(kinds=[kind], dummy_limits=False))

    def p_get_events(self, parameters, gate):
        tmin, tmax = self.get_cleaned('tmin tmax', parameters)
        return gate.get_events(tmin=tmin, tmax=tmax)

    def p_get_channels(self, parameters, gate):
        tmin, tmax, codes = self.get_cleaned('tmin tmax codes', parameters)
        return gate.get_channels(tmin=tmin, tmax=tmax, codes=codes)

    def p_get_sensors(self, parameters, gate):
        tmin, tmax, codes = self.get_cleaned('tmin tmax codes', parameters)
        return gate.get_sensors(tmin=tmin, tmax=tmax, codes=codes)

    def p_get_responses(self, parameters, gate):
        tmin, tmax, codes = self.get_cleaned('tmin tmax codes', parameters)
        return gate.get_responses(tmin=tmin, tmax=tmax, codes=codes)

    def p_get_coverage(self, parameters, gate):
        kind, tmin, tmax = self.get_cleaned('kind tmin tmax', parameters)
        return gate.get_coverage(kind, tmin=tmin, tmax=tmax)

    def p_get_spectrograms(self, parameters, gate):
        tmin, tmax, ymin, ymax = self.get_cleaned(
            'tmin tmax ymin ymax',
            parameters)

        return gate.get_spectrogram_images(
            tmin=tmin,
            tmax=tmax,
            ymin=ymin,
            ymax=ymax)

    def p_get_carpets(self, parameters, gate):
        tmin, tmax, ymin, ymax, nx, ny, codes, overview_method \
            = self.get_cleaned(
                'tmin tmax ymin ymax nx ny codes overview_method',
                parameters)

        images = gate.get_carpet_images(
            tmin=tmin,
            tmax=tmax,
            codes=codes,
            ymin=ymin,
            ymax=ymax,
            nx=nx or 6000,
            ny=ny or 400,
            overview_method=overview_method)

        gate.advance_accessor()
        return images


color_themes = {
    'black': dict(
        edgecolor=Color('black').rgba,
        color_t=Color('black').rgba)}

for name in g_pyrocko_color_cycle_base:
    color_themes[name] = dict(
        edgecolor=Color(name+'-dark').rgba,
        color_t=Color(name).rgba)


class BeachballHandler(SquirrelRequestHandler):
    def get(self):
        m6 = [
            float(self.get_query_argument(component))
            for component in 'mnn mee mdd mne mnd med'.split()]

        color_theme_name = self.get_query_argument('theme', 'black')

        mt = pmt.as_mt(m6)

        fig = plt.figure(figsize=(0.5, 0.5))
        axes = fig.add_subplot(1, 1, 1, aspect=1.)
        axes.axison = False
        axes.set_xlim(-0.52, 0.52)
        axes.set_ylim(-0.52, 0.52)

        beachball.plot_beachball_mpl(
            mt, axes,
            position=(0, 0),
            size_units='data',
            **color_themes[color_theme_name],
            linewidth=0.8)

        buffer = BytesIO()
        fig.savefig(buffer, format='svg')

        plt.close(fig)

        self.set_header('Content-Type', 'image/svg+xml')
        self.write(buffer.getvalue())


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
            raise ToolError(
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
        page_path=None):

    ip = get_ip(host)

    if page_path is None:
        page_path = os.path.join(
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
            page_path=page_path,
            pyrocko_long_version=info.long_version,
            pyrocko_src_path=info.src_path)

    squirrel_handler_dict = dict(
        squirrel=squirrel,
        server_info=server_info,
    )

    app = web.Application(
        [
            (
                r'/((?:css|js|images)/.*'
                r'|index.html|site.webmanifest|favicon.ico|)',
                web.StaticFileHandler,
                dict(
                    path=page_path,
                    default_filename='index.html'
                )
            ),
            (
                r'/squirrel/heartbeat',
                SquirrelHeartbeatHandler,
                squirrel_handler_dict,
            ),
            (
                r'/squirrel/info/([a-z0-9_]+)',
                SquirrelInfoHandler,
                squirrel_handler_dict,
            ),
            (
                r'/squirrel/raw/([a-z0-9_]+)',
                SquirrelRawHandler,
                squirrel_handler_dict,
            ),
            (
                r'/squirrel/gate(?:/([a-z0-9_]+|/?))',
                SquirrelGatesHandler,
                squirrel_handler_dict,
            ),
            (
                r'/squirrel/gate/([a-z0-9_]+)/([a-z0-9_]+)',
                SquirrelGateHandler,
                squirrel_handler_dict,
            ),
            (
                r'/beachball',
                BeachballHandler,
                squirrel_handler_dict,
            ),
        ],
        log_function=log_request,
        debug=debug,
    )

    logger.info(
        'Binding service to %s, port %i.',
        describe_ip(ip), port)

    try:
        http_server = app.listen(port, ip)
    except OSError as e:
        if e.errno == 98:
            await fail(ToolError(
                'Address already in use, try specifying a different port '
                'number with --port INT.'))
            return

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
g_exception = None


async def fail(e):
    global g_exception
    g_exception = e
    await shutdown()


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

    loop = asyncio.get_event_loop()
    loop.stop()


def run(
        squirrel,
        gates={},
        host='localhost',
        port=2323,
        open=False,
        debug=False,
        page_path=None):

    for gate in gates.values():
        gate.set_squirrel(squirrel)

    g_gates.update(gates)

    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('Debug mode activated.')
        autoreload.add_reload_hook(lambda: time.sleep(2))
        if page_path is None:
            dev_static_path = os.path.join(
                info.src_path, 'src', 'squirrel', 'service', 'page')
            if os.path.exists(dev_static_path):
                page_path = dev_static_path

        logger.debug('Serving static files from: %s', page_path)

    loop = asyncio.get_event_loop()
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(
            s, lambda s=s: asyncio.create_task(shutdown(s)))

    try:
        loop.create_task(
            serve(
                squirrel,
                host=host,
                port=port,
                open=open,
                debug=debug,
                page_path=page_path))

        loop.run_forever()

    finally:
        loop.close()
        if g_exception is not None:
            raise g_exception

        logger.debug('Shutdown complete.')
