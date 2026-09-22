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
import time
import base64
import uuid
from datetime import datetime
from io import BytesIO

import numpy as num
from matplotlib import pyplot as plt
import matplotlib as mpl
from tornado import web

from pyrocko import util, server, info, trace
from pyrocko.carpet import CarpetResampleError
from pyrocko import squirrel as squirrel_module
from pyrocko.squirrel import model
from pyrocko import guts
from pyrocko.squirrel.error import ToolError, SquirrelError
from pyrocko.squirrel.mantra import Mantra
from pyrocko import moment_tensor as pmt
from pyrocko.plot import beachball
from pyrocko.color import Color, g_pyrocko_color_cycle_base

from . import scouts


add_cli_arguments = server.add_cli_arguments


logger = logging.getLogger('psq.service.server')


class Checkpoints:
    def __init__(self):
        self.checkpoints = []

    def add(self, label):
        self.checkpoints.append((time.time(), label))

    def report(self):
        self.checkpoints.append((time.time(), ''))
        ll = max(len(label) for (_, label) in self.checkpoints)
        return '\n'.join(
            '  %s %10.2f ms' % ((label+':').ljust(ll+1), (t1 - t0)*1000)
            for ((t0, label), (t1, _))
            in zip(self.checkpoints[:-1], self.checkpoints[1:]))


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


g_session_data = {}


def get_session_data(session_id):
    return g_session_data[session_id]


g_accessor_data = {}


def get_accessor_data(accessor_id):
    if accessor_id not in g_accessor_data:
        g_accessor_data[accessor_id] = {}

    return g_accessor_data[accessor_id]


def int_gt_zero(s):
    i = int(s)
    if i <= 0:
        raise ValueError('Value must be greater than zero.')
    return i


class SquirrelRequestHandler(server.RequestHandler):

    def initialize(self, squirrel=None):
        self._squirrel = squirrel

    def check_csrf(self):
        # Cross-site requests which browsers send without a CORS preflight
        # (plain HTML forms, `fetch` with a "simple" content type such as
        # text/plain) cannot carry `Content-Type: application/json`. Requests
        # with that content type from other origins are preflighted, and we
        # do not answer preflights. Requiring it on all state-changing
        # requests therefore rejects cross-site request forgery attempts.

        if self.request.method in ('GET', 'HEAD', 'OPTIONS'):
            return

        content_type = self.request.headers.get('Content-Type', '')
        if content_type.split(';')[0].strip().lower() != 'application/json':
            raise web.HTTPError(
                415,
                reason='Content-Type "application/json" required.')

    def prepare(self):
        server.RequestHandler.prepare(self)
        self.check_csrf()

        session_id = self.get_secure_cookie('session')
        if session_id:
            session_id = session_id.decode('ascii')

        if not session_id:
            session_id = str(uuid.uuid4())
            # expires_days=None sets lifetime of the browser session
            self.set_secure_cookie(
                'session', session_id.encode('ascii'), expires_days=None)

        if session_id not in g_session_data:
            g_session_data[session_id] = {}

        self.session_id = session_id

    def get_cleaned(self, names, parameters):
        if isinstance(names, str):
            names = names.split()

        clean = {
            'kind': lambda x: str_choice(x, model.g_content_kinds),
            'time': util.str_to_time_fillup,
            'tmin': util.str_to_time_fillup,
            'tmax': util.str_to_time_fillup,
            'codes': to_codes_list,
            'codes_visible': to_codes_list,
            'ymin': float,
            'ymax': float,
            'fmin': float,
            'fmax': float,
            'nx': int_gt_zero,
            'ny': int_gt_zero,
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
                    400, reason='Squirrel error: %s' % ' '.join(
                        str(e).split()))

            self.set_header('Content-Type', 'application/json')
            self.write(json.dumps(raw, cls=GutsJSONEncoder))


class SquirrelHeartbeatHandler(SquirrelRequestHandler):

    n_clients_active = 0
    n_clients = 0

    async def get(self):
        SHH = SquirrelHeartbeatHandler
        SHH.n_clients_active += 1
        SHH.n_clients += 1
        logger.info(
            'Client connected.')

        logger.info(
            'Clients: %i active, %i seen' % (
                SHH.n_clients_active,
                SHH.n_clients))

        self.set_header('Content-Type', 'application/octet-stream')
        self.set_header('X-Accel-Buffering', 'no')
        await self.flush()
        try:
            time_start = time.time()
            while True:
                if server.g_shutdown_event.is_set():
                    break

                self.write(
                    json.dumps(dict(
                        time_start=time_start,
                        time_now=time.time(),
                        server_info=server.g_server_info)))

                try:
                    await self.flush()
                    await asyncio.sleep(1)
                except (asyncio.exceptions.CancelledError, Exception):
                    break

        finally:
            logger.info('Client disconnected.')
            SHH.n_clients_active -= 1
            logger.info(
                'Clients: %i active, %i seen' % (
                    SHH.n_clients_active,
                    SHH.n_clients))

            self.finish()


class SquirrelInfoHandler(SquirrelRequestHandler):
    def p_server(self, parameters):
        return server.g_server_info

    def p_gates(self, parameters):
        return describe_gates()


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

    def p_get_rich_coverage(self, parameters):
        tmin, tmax = self.get_cleaned('tmin tmax', parameters)
        return self._squirrel.get_rich_coverage(
            tmin=tmin, tmax=tmax, limit=500)


class ScaleChoice(guts.StringChoice):
    choices = ['lin', 'log']


class CarpetImage(guts.Object):
    codes = model.CodesNSLCE.T()
    shape = guts.Tuple.T(2, guts.Int.T())
    tmin = guts.Timestamp.T()
    tmax = guts.Timestamp.T()
    ymin = guts.Float.T()
    ymax = guts.Float.T()
    yscale = ScaleChoice.T()
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


class Waveview(guts.Object):
    codes = model.CodesNSLCE.T()
    tmin = guts.Timestamp.T()
    tmax = guts.Timestamp.T()
    ymin = guts.Float.T()
    ymax = guts.Float.T()
    fmin = guts.Float.T(optional=True)
    fmax = guts.Float.T(optional=True)
    size = guts.Int.T()
    polygon_data_base64 = guts.String.T()


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
        if not re.search(r'-L(min|max|mean)\d\d$', codes.extra)]


GATE_NAME_PATTERN = r'[a-z0-9_]+'
GATE_NAME_DEFAULT = 'default'


class Gate(guts.Object):
    '''
    A named view on the data, produced by a processing pipeline.

    The gate name is used in URLs, so it is restricted to lowercase letters,
    digits and underscores.
    '''

    name = guts.String.T(default=GATE_NAME_DEFAULT)
    mantra = Mantra.T()

    @classmethod
    def from_mantra(cls, mantra):
        if not re.fullmatch(GATE_NAME_PATTERN, mantra.name):
            raise ToolError(
                'Invalid mantra name "%s": names of mantras used with the '
                'service may only contain lowercase letters, digits and '
                'underscores.' % mantra.name)

        return cls(name=mantra.name, mantra=mantra)

    @classmethod
    def from_query_arguments(cls, codes=None, tmin=None, tmax=None, time=None):
        operators = []

        return cls(
            name=GATE_NAME_DEFAULT,
            mantra=Mantra(name=GATE_NAME_DEFAULT, operators=operators))

    def get_info(self, detailed=False):
        info = dict(
            name=self.name,
            operators=[operator.name for operator in self.mantra.operators])

        if detailed:
            # This can be large (it lists the channel mappings), so it is
            # not included in the summary of all gates.
            info['description'] = self.mantra.describe()

        return info

    def set_squirrel(self, squirrel):
        self.mantra.setup(squirrel)

    def get_time_span(self, *args, **kwargs):
        return self.mantra.outlet.get_time_span(*args, **kwargs)

    def get_codes(self, *args, **kwargs):
        return drop_resolution_codes(
            self.mantra.outlet.get_codes(*args, **kwargs))

    def get_channels(self, *args, **kwargs):
        return self.mantra.outlet.get_channels(*args, **kwargs)

    def get_sensors(self, *args, **kwargs):
        return self.mantra.outlet.get_sensors(*args, **kwargs)

    def get_responses(self, *args, **kwargs):
        return self.mantra.outlet.get_responses(*args, **kwargs)

    def get_events(self, *args, **kwargs):
        return self.mantra.outlet.get_events(*args, **kwargs)

    def get_coverage(self, *args, **kwargs):
        kwargs['codes'] = '*.*.*.*.*'
        return self.mantra.outlet.get_coverage(*args, **kwargs)

    def get_rich_coverage(self, *args, **kwargs):
        kwargs['codes'] = '*.*.*.*.*'
        return self.mantra.outlet.get_rich_coverage(*args, **kwargs)

    def get_carpet_images(
            self,
            *args,
            limits={},
            nx=6000,
            ny=100,
            overview_method='mean',
            format='webp',
            **kwargs):

        def select_axis_and_scale(carpet):
            axes = list(carpet.component_axes.keys())
            axis = None
            if len(axes) == 1:
                axis = axes[0]

            scale = 'lin'
            if axis == 'frequency':
                scale = 'log'

            return axis, scale

        def carpet_to_image(carpet):
            times_this = []
            times_this.append(time.time())
            overview_method = carpet.meta['overview_method']
            try:
                component_axis, scale = select_axis_and_scale(carpet)
                ymin, ymax = limits.get(component_axis, (None, None))
                carpet = carpet.resample_band(
                    ymin, ymax, ny,
                    scale=scale,
                    component_axis=component_axis)

            except CarpetResampleError as e:
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
                ymin=carpet.component_axes[component_axis][0],
                ymax=carpet.component_axes[component_axis][-1],
                yscale=scale,
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
                    model.CodesNSLCE('*.*.*.*.*-L%s??' % overview_method)]

            else:
                codes_overview = [
                    v.replace(extra=v.extra + '-L%s??' % overview_method)
                    for v in codes
                    if v is not None]

                codes.extend(codes_overview)

        t0 = time.time()

        carpets = self.mantra.outlet.get_carpets(
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

    def get_waveviews(
            self,
            tmin,
            tmax,
            codes,
            fmin,
            fmax,
            nx=6000,
            ny=100,
            accessor_id='default'):

        nfold_limit = 1024
        nsamples_limit = nx*nfold_limit

        sample_rate_max = None
        if nsamples_limit is not None:
            if (tmax - tmin) <= 0.0:
                return []

            sample_rate_max = nsamples_limit / (tmax - tmin)

        checkpoints = Checkpoints()

        checkpoints.add('waveform query')

        traces = self.mantra.outlet.get_waveforms(
            tmin=tmin,
            tmax=tmax,
            codes=codes,
            sample_rate_max=sample_rate_max,
            tscale_min=0.01 * (tmax - tmin),
            downloads_enabled=False,
            accessor_id=accessor_id)

        checkpoints.add('filtering')

        for tr in traces:
            if fmin is not None and fmin < 0.5 / tr.deltat:
                tr.highpass(4, fmin)

            if fmax is not None and fmax < 0.5 / tr.deltat:
                tr.lowpass(4, fmax)

        checkpoints.add('packaging')

        waveviews = []
        for tr in traces:
            nfold = min(
                nfold_limit,
                trace.nextpow2(
                    max(1, int(((tmax - tmin) / tr.deltat) / (nx*2)))))

            ncut = (tr.data_len() // nfold) * nfold

            n = ncut // nfold
            if n < 2:
                continue

            y = tr.ydata[:ncut].reshape((n, nfold))
            ymins = num.min(y, axis=1)
            ymaxs = num.max(y, axis=1)

            wmin = tr.tmin + 0.5 * nfold * tr.deltat
            wmax = tr.tmin + (0.5 + (n - 1)) * nfold * tr.deltat

            x = num.arange(n)
            poly = num.empty((2 * n, 2), dtype=num.float32)
            poly[:n, 0] = x
            poly[n:, 0] = x[::-1]
            poly[:n, 1] = ymins
            poly[n:, 1] = ymaxs[::-1]
            data_base64 = base64.b64encode(poly).decode('ascii')

            waveviews.append(
                Waveview(
                    tmin=wmin,
                    tmax=wmax,
                    ymin=float(num.min(ymins)),
                    ymax=float(num.max(ymaxs)),
                    fmin=fmin,
                    fmax=fmax,
                    codes=tr.codes,
                    size=n,
                    polygon_data_base64=data_base64))

        logger.debug('Costs get_waveview:\n%s' % checkpoints.report())

        return waveviews

    def get_inspector_classes(self):
        return {
            'response': scouts.ResponseScout,
        }

    def get_inspector(self, accessor_id, name):
        ad = get_accessor_data(accessor_id)
        adk = ('inspector', name)
        if adk not in ad:
            ad[adk] = self.get_inspector_classes()[name](
                name=name,
                mantra=self.mantra)

        return ad[adk]

    def get_context(
            self,
            name,
            context,
            accessor_id):

        return self.get_inspector(accessor_id, name).update(context)

    def advance_accessor(self, accessor_id='default', cache_id=None):
        self.mantra.outlet.advance_accessor(accessor_id, cache_id)


g_gates = {}


def describe_gates():
    return [gate.get_info() for gate in g_gates.values()]


def gates_from_mantras(mantras, gates):
    '''
    Get gates for the given mantras, in addition to the already existing gates.
    '''

    gates = dict(gates)
    for mantra in mantras:
        gate = Gate.from_mantra(mantra)
        if gate.name in gates:
            raise ToolError(
                'Duplicate gate name "%s". Mantra names must be unique and '
                'must not be "%s".' % (gate.name, GATE_NAME_DEFAULT))

        gates[gate.name] = gate

    return gates


def warn_about_shared_codes(gates):
    '''
    Warn if different gates provide data for the same codes.

    Data from all gates is shown together, so their codes must be distinct.
    Operators can include the mantra name in their output codes for this
    purpose (e.g. by using ``{o.mantra}`` in their codes projection template).
    '''

    for kind in ('waveform', 'carpet'):
        gate_names_by_codes = {}
        for name, gate in gates.items():
            for codes in gate.get_codes(kind=kind):
                gate_names_by_codes.setdefault(codes, []).append(name)

        for codes, gate_names in sorted(gate_names_by_codes.items()):
            if len(gate_names) > 1:
                logger.warning(
                    'Codes "%s" (%s) are provided by multiple gates: %s',
                    codes, kind, ', '.join(gate_names))


class SquirrelGatesHandler(SquirrelRequestHandler):
    SUPPORTED_METHODS = ('GET', 'HEAD')

    def get(self, name=None):
        self.set_header('Content-Type', 'application/json')
        if not name:
            self.write(json.dumps(describe_gates()))
        elif name in g_gates:
            self.write(json.dumps(g_gates[name].get_info(detailed=True)))
        else:
            raise web.HTTPError(404, reason='No such gate: %s' % name)


class SquirrelGateHandler(SquirrelRequestHandler):

    def get_accessor_id(self, gate, suffix=''):
        # Caches must not be shared between gates or between sessions.
        return '%s_%s%s' % (self.session_id, gate.name, suffix)

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

    def p_get_rich_coverage(self, parameters, gate):
        tmin, tmax = self.get_cleaned('tmin tmax', parameters)
        return gate.get_rich_coverage(tmin=tmin, tmax=tmax, limit=500)

    def p_get_carpets(self, parameters, gate):
        tmin, tmax, ymin, ymax, nx, ny, codes, overview_method \
            = self.get_cleaned(
                'tmin tmax ymin ymax nx ny codes overview_method',
                parameters)

        if tmin == tmax:
            tmax += 1.0

        limits = {
            'frequency': (ymin, ymax),
        }

        accessor_id = self.get_accessor_id(gate)

        images = gate.get_carpet_images(
            tmin=tmin,
            tmax=tmax,
            codes=codes,
            limits=limits,
            nx=nx or 6000,
            ny=ny or 400,
            overview_method=overview_method,
            accessor_id=accessor_id)

        gate.advance_accessor(accessor_id=accessor_id, cache_id='carpet')
        return images

    def p_get_waveviews(self, parameters, gate):
        tmin, tmax, fmin, fmax, nx, ny, codes \
            = self.get_cleaned(
                'tmin tmax fmin fmax nx ny codes',
                parameters)

        accessor_id = self.get_accessor_id(gate)

        waveviews = gate.get_waveviews(
            tmin=tmin,
            tmax=tmax,
            codes=codes,
            fmin=fmin,
            fmax=fmax,
            nx=nx or 6000,
            ny=ny or 400,
            accessor_id=accessor_id)

        gate.advance_accessor(accessor_id=accessor_id, cache_id='waveform')
        return waveviews

    def p_get_context(self, parameters, gate):
        time, tmin, tmax, frequency_min, frequency_max, codes, codes_visible \
            = self.get_cleaned(
                'time tmin tmax fmin fmax codes codes_visible',
                parameters)

        names = ['response']

        context = scouts.ScoutContext(
            time=time,
            tmin=tmin,
            tmax=tmax,
            codes=codes,
            codes_visible=codes_visible,
            frequency_min=frequency_min,
            frequency_max=frequency_max)

        accessor_id = self.get_accessor_id(gate, '_context')

        results = []
        for name in names:
            results.extend(gate.get_context(
                name=name,
                context=context,
                accessor_id=accessor_id))

        gate.advance_accessor(accessor_id=accessor_id, cache_id='waveform')
        gate.advance_accessor(accessor_id=accessor_id, cache_id='carpet')

        return results


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


def run(
        squirrel,
        gates={},
        host='localhost',
        port=2323,
        open=False,
        debug=False,
        cookie_secret_path=None,
        page_path=None):

    from pyrocko import server

    if debug:
        logger.setLevel(logging.DEBUG)

    for gate in gates.values():
        gate.set_squirrel(squirrel)

    g_gates.update(gates)
    warn_about_shared_codes(g_gates)

    if page_path is None:
        page_path = os.path.join(
            os.path.split(squirrel_module.__file__)[0],
            'service',
            'page')

        if debug:
            page_path_debug = os.path.join(
                info.src_path, 'src',
                'squirrel',
                'service',
                'page')

            if os.path.exists(page_path_debug):
                page_path = page_path_debug

    handler_data = dict(
        squirrel=squirrel,
    )

    handlers = [
        (
            r'/squirrel/heartbeat',
            SquirrelHeartbeatHandler,
            handler_data,
        ),
        (
            r'/squirrel/info/([a-z0-9_]+)',
            SquirrelInfoHandler,
            handler_data,
        ),
        (
            r'/squirrel/raw/([a-z0-9_]+)',
            SquirrelRawHandler,
            handler_data,
        ),
        (
            r'/squirrel/gate(?:/(%s|/?))' % GATE_NAME_PATTERN,
            SquirrelGatesHandler,
            handler_data,
        ),
        (
            r'/squirrel/gate/(%s)/([a-z0-9_]+)' % GATE_NAME_PATTERN,
            SquirrelGateHandler,
            handler_data,
        ),
        (
            r'/beachball',
            BeachballHandler,
            handler_data,
        ),
    ]

    try:
        server.run(
            host=host,
            port=port,
            handlers=handlers,
            open=open,
            debug=debug,
            cookie_secret_path=cookie_secret_path,
            page_path=page_path)

    except server.ServerError as e:
        raise ToolError(e) from e
