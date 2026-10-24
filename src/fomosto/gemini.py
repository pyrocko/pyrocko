# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Fomosto backend for the GEMINI program system.

GEMINI (Green's function of the Earth by MINor integration) by Jörg Dalkolmo
and Wolfgang Friederich computes complete synthetic seismograms for a
spherically symmetric, transversely isotropic, anelastic earth by numerical
integration of the equations of motion in the frequency domain [Friederich
& Dalkolmo 1995]. It consists of three programs, which this backend expects to
be installed as ``fomosto_gemini_Gemini_2_2``, ``fomosto_gemini_Dispec_2_2``
and ``fomosto_gemini_Totido_2_2``, e.g. with the ``fomosto-gemini`` package:

* ``Gemini`` computes the Green's basis solutions in the frequency-degree
  domain for a given source depth.
* ``Dispec`` combines them with a moment tensor source and sums over the
  spherical harmonics, giving spectra at the receivers.
* ``Totido`` transforms the spectra to the time domain.

The earth model, which may be transversely isotropic, is given in GEMINI's own
polynomial format in the ``earthmodel`` entry of the store's ``extra/gemini``
file. The ``earthmodel_1d`` in the store's ``config`` is an isotropic
approximation of it, used for travel time tables. If the ``earthmodel`` entry
is omitted, it is fitted to ``earthmodel_1d`` when the store is built.

Models can be converted between the formats on the command line::

    python -m pyrocko.fomosto.gemini nd-to-gemini IN OUT
    python -m pyrocko.fomosto.gemini gemini-to-nd IN OUT

Receivers must be at the surface. Components are computed in source-centred
coordinates, with the source at the north pole and the receivers on the 180°
meridian, so that any epicentral distance between 0° and 180° can be used.

.. rubric:: References

Friederich, W. and Dalkolmo, J. (1995): Complete synthetic seismograms for a
spherically symmetric earth by a numerical computation of the Green's function
in the frequency domain. Geophysical Journal International, 122, 537-550.
'''

import asyncio
import contextlib
import hashlib
import heapq
import itertools
import logging
import math
import os
import shutil
import signal
import time
from tempfile import mkdtemp
from os.path import join as pjoin

import numpy as num
from numpy.polynomial import Polynomial

from pyrocko import trace, cake, gf, util
from pyrocko.fomosto import check_variant
from pyrocko.guts import Object, Float, Int, Bool, Tuple, String
from pyrocko.guts import TBase, literal
from pyrocko.moment_tensor import MomentTensor, symmat6

guts_prefix = 'pf'

logger = logging.getLogger('pyrocko.fomosto.gemini')

km = 1000.

# how to call the programs, for each GEMINI version
program_bins = {
    '2.2': {
        'gemini': 'fomosto_gemini_Gemini_2_2',
        'dispec': 'fomosto_gemini_Dispec_2_2',
        'totido': 'fomosto_gemini_Totido_2_2',
    },
}

# variants usable with ``fomosto init`` and the executables they need
variants = dict(
    (version, list(bins.values())) for (version, bins) in program_bins.items())

# Array limits compiled into the Fortran programs.
gemini_nlayer = 20        # Gemini: nlayer, max number of model layers
gemini_ncoefs = 4         # Gemini: max number of polynomial coefficients
gemini_lmx = 10000        # Gemini, Dispec: lmx, max harmonic degree
dispec_nfr = 4095         # Dispec: nfr, max number of frequencies
dispec_nseimx = 149       # Dispec: nseimx - 1, max number of receivers
dispec_ntmx = 200         # Dispec: ntpmx, max length of degree taper
totido_nfmax = 16384      # Totido: nfmax, max number of samples

# Depth ranges [m] in which discontinuities are named when converting GEMINI
# models to cake models.
discontinuity_depth_ranges = {
    'moho': (0.*km, 100.*km),
    'cmb': (2591.*km, 3191.*km),
    'icb': (4850.*km, 5450.*km),
}

# Default bulk attenuation, used where it is undefined or infinite.
qkappa_default = 57823.

# The CMT file holds the moment tensor in units of 10**cmt_exponent dyne cm,
# here 1e15 Nm, keeping the output of Dispec and Totido at moderate values.
cmt_exponent = 22
cmt_moment = 10.0**cmt_exponent * 1e-7  # [Nm]

# Dispec/Totido give nanometers for moments in Nm.
nm = 1e-9


def have_backend():
    '''
    Get the GEMINI versions for which all programs are available.
    '''
    return set(
        version for (version, bins) in program_bins.items()
        if all(shutil.which(exe) for exe in bins.values()))


class GeminiError(gf.store.StoreError):
    pass


class Interrupted(gf.store.StoreError):
    def __str__(self):
        return 'Interrupted.'


def _fortran_floats(tokens):
    return [float(tok.replace('D', 'E').replace('d', 'e')) for tok in tokens]


class GeminiLayer(object):
    '''
    Layer of a GEMINI earth model.

    :param r_bottom: radius of the bottom of the layer [km]
    :param coefs: polynomial coefficients, array of shape ``(ncoefs, 6)``.
        The columns are density [g/cm^3], vertical and horizontal P-wave
        velocity, vertical and horizontal S-wave velocity [km/s], and the
        anisotropy parameter eta. Rows are the coefficients of increasing
        powers of the normalized radius ``x = r / r_surface``.
    :param qmu: shear quality factor, ``<= 0`` for a fluid layer
    :param qkappa: bulk quality factor
    '''

    def __init__(self, r_bottom, coefs, qmu, qkappa):
        self.r_bottom = float(r_bottom)
        self.coefs = num.array(coefs, dtype=float).reshape(-1, 6)
        self.qmu = float(qmu)
        self.qkappa = float(qkappa)

    @property
    def is_fluid(self):
        return self.qmu <= 0.0

    def evaluate(self, x):
        '''
        Evaluate the polynomials at normalized radii ``x``.

        :returns: array of shape ``(6, len(x))`` with the rows density,
            vpv, vph, vsv, vsh, eta
        '''
        x = num.atleast_1d(num.asarray(x, dtype=float))
        result = num.zeros((6, x.size))
        for coefs in self.coefs[::-1]:
            result = result * x[num.newaxis, :] + coefs[:, num.newaxis]

        return result


class GeminiModel(object):
    '''
    Spherically symmetric, possibly transversely isotropic GEMINI earth model.

    Densities are in [g/cm^3], velocities in [km/s] and radii in [km], as in
    the native file format of GEMINI. Layers are ordered from the centre to
    the surface.
    '''

    def __init__(
            self, layers, r_surface,
            reference_frequency=1.0,
            transversely_isotropic=False,
            comments=None):

        self.layers = list(layers)
        self.r_surface = float(r_surface)
        self.reference_frequency = float(reference_frequency)
        self.transversely_isotropic = bool(transversely_isotropic)
        self.comments = list(comments or [])

    @classmethod
    def from_string(cls, s):
        '''
        Parse earth model in GEMINI's native file format.

        This reads the format the same way as GEMINI's ``gtmanis.f`` does.
        '''

        lines = s.splitlines()
        state = dict(iline=0)

        def next_line():
            iline = state['iline']
            if iline >= len(lines):
                raise GeminiError(
                    'Unexpected end of input in GEMINI earth model.')

            state['iline'] += 1
            return lines[iline]

        def read_values(n):
            values = []
            while len(values) < n:
                values.extend(next_line().split())

            try:
                return _fortran_floats(values[:n])
            except ValueError:
                raise GeminiError(
                    'Invalid number in line %i of GEMINI earth model.'
                    % state['iline'])

        comments = []
        while state['iline'] < len(lines) \
                and lines[state['iline']].startswith('#'):
            comment = next_line()[1:].rstrip()
            # marker line, added by to_string()
            if 'Do not insert any comments below' not in comment:
                comments.append(comment)

        try:
            nlayers = int(next_line()[:2])
            line = next_line()
            ncoefs = [int(line[2*i:2*i+2]) for i in range(nlayers)]
        except ValueError:
            raise GeminiError(
                'Cannot read number of layers or number of polynomial '
                'coefficients of GEMINI earth model (fixed format, 2 columns '
                'per value).')

        reference_frequency, = read_values(1)
        ifanis, = read_values(1)
        for _ in range(3):
            next_line()

        layers = []
        for ilayer in range(nlayers):
            r, rho, vpv, vph, vsv, vsh, qmu, qkappa, eta = read_values(9)
            coefs = [[rho, vpv, vph, vsv, vsh, eta]]
            for _ in range(ncoefs[ilayer] - 1):
                coefs.append(read_values(6))

            next_line()
            layers.append(GeminiLayer(r, coefs, qmu, qkappa))

        r_surface, = read_values(1)

        return cls(
            layers, r_surface,
            reference_frequency=reference_frequency,
            transversely_isotropic=int(ifanis) == 1,
            comments=comments)

    def to_string(self):
        '''
        Get earth model in GEMINI's native file format.
        '''

        def fmt(values):
            return ' '.join('%19.13g' % v for v in values)

        nlayers = len(self.layers)
        lines = ['#' + comment for comment in self.comments]
        lines.extend([
            '# +++++++ Do not insert any comments below this line !! '
            '+++++++++++++++',
            '%2i  Number of layers' % nlayers,
            ''.join('%2i' % layer.coefs.shape[0] for layer in self.layers)
            + '  Number of polynomial coefficients in each layer',
            '%g  Reference frequency of quality factors (Hertz)'
            % self.reference_frequency,
            '%i  Transversely isotropic? 1=y, else=n'
            % int(self.transversely_isotropic),
            '-' * 79,
            'Radius Density V-Pv V-Ph V-Sv V-Sh Qmu Qk Eta',
            '-' * 79])

        for layer in self.layers:
            rho, vpv, vph, vsv, vsh, eta = layer.coefs[0]
            lines.append(fmt([
                layer.r_bottom, rho, vpv, vph, vsv, vsh,
                layer.qmu, layer.qkappa, eta]))

            for coefs in layer.coefs[1:]:
                lines.append(' '*20 + fmt(coefs))

            # separator, any content is skipped by gtmanis.f
            lines.append('')

        lines.append(fmt([self.r_surface]))
        lines.append('-' * 79)
        return '\n'.join(lines) + '\n'

    def validate(self, nlayers_max=gemini_nlayer):
        if len(self.layers) == 0:
            raise GeminiError('GEMINI earth model has no layers.')

        if len(self.layers) > nlayers_max:
            raise GeminiError(
                'GEMINI earth model has %i layers, GEMINI supports at most '
                '%i.' % (len(self.layers), nlayers_max))

        if self.layers[0].r_bottom != 0.0:
            raise GeminiError(
                'GEMINI earth model must extend to the centre of the earth.')

        for layer in self.layers:
            if not (1 <= layer.coefs.shape[0] <= gemini_ncoefs):
                raise GeminiError(
                    'GEMINI earth model layers must have 1 to %i polynomial '
                    'coefficients.' % gemini_ncoefs)

        radii = [layer.r_bottom for layer in self.layers] + [self.r_surface]
        if any(r2 <= r1 for (r1, r2) in zip(radii[:-1], radii[1:])):
            raise GeminiError(
                'GEMINI earth model layer radii must be increasing.')

        if self.layers[-1].is_fluid:
            raise GeminiError(
                'GEMINI does not support a fluid top layer (ocean).')

    def layer_bounds(self):
        radii = [layer.r_bottom for layer in self.layers] + [self.r_surface]
        return list(zip(self.layers, radii[:-1], radii[1:]))

    def isotropic_properties(self, layer, r):
        '''
        Get isotropic approximation of the material properties of a layer.

        For a transversely isotropic model the equivalent isotropic Voigt
        average moduli are used, as by GEMINI (after Dziewonski & Anderson
        1980). For an isotropic model only the vertical velocities are used,
        as by GEMINI.

        :param layer: :py:class:`GeminiLayer` of this model
        :param r: radii [km]
        :returns: ``(vp, vs, rho, qp, qs)``, arrays in SI units
        '''

        rho, vpv, vph, vsv, vsh, eta = layer.evaluate(
            num.asarray(r) / self.r_surface)

        if layer.is_fluid:
            vs = num.zeros_like(vpv)
            vp = vpv
        elif self.transversely_isotropic:
            a = rho * vph**2
            c = rho * vpv**2
            n = rho * vsh**2
            l_ = rho * vsv**2
            f = eta * (a - 2.0*l_)
            kappa = (4.0*(a + f - n) + c) / 9.0
            mu = (a + c - 2.0*f + 5.0*n + 6.0*l_) / 15.0
            vp = num.sqrt((kappa + 4.0/3.0 * mu) / rho)
            vs = num.sqrt(mu / rho)
        else:
            vp = vpv
            vs = vsv

        if layer.is_fluid:
            qs = num.zeros_like(vp)
            qp = num.full_like(vp, layer.qkappa)
        else:
            s = (4.0/3.0) * (vs/vp)**2
            qs = num.full_like(vp, layer.qmu)
            qp = 1.0 / (s/layer.qmu + (1.0 - s)/layer.qkappa)

        return vp*km, vs*km, rho*km, qp, qs

    def to_cake(self, rtol=5e-4):
        '''
        Get isotropic approximation as :py:class:`pyrocko.cake.LayeredModel`.

        Each polynomial layer is approximated by gradient layers, with nodes
        inserted until linear interpolation deviates by less than ``rtol``
        (relative) from the polynomials.

        Discontinuities are named, as needed for standard phase definitions,
        if they are roughly at the expected depth (see
        ``discontinuity_depth_ranges``): the top of a fluid layer is
        named ``cmb``, its bottom ``icb``, and the shallowest discontinuity
        with a P-wave velocity increasing across 7.5 km/s is named ``moho``.
        '''

        def props(layer, r):
            return num.array(self.isotropic_properties(layer, r))

        def refine(layer, r_top, r_bottom, p_top, p_bottom, level=0):
            # Nodes from top to bottom.
            rm = 0.5 * (r_top + r_bottom)
            pm = props(layer, [rm])[:, 0]
            pm_lin = 0.5 * (p_top + p_bottom)
            scale = num.maximum(num.abs(pm), num.abs(pm_lin))
            scale[scale == 0.0] = 1.0
            if level < 12 and num.any(num.abs(pm - pm_lin) > rtol * scale):
                return refine(layer, r_top, rm, p_top, pm, level+1)[:-1] \
                    + refine(layer, rm, r_bottom, pm, p_bottom, level+1)

            return [(r_top, p_top), (r_bottom, p_bottom)]

        r_surface = self.r_surface
        lines = []
        moho_found = False
        for layer, r_bottom, r_top in self.layer_bounds()[::-1]:
            p_top, p_bottom = props(layer, [r_top, r_bottom]).T
            nodes = refine(layer, r_top, r_bottom, p_top, p_bottom)

            name = None
            if lines:
                _, m_above, _ = lines[-1]
                vp_below = nodes[0][1][0]
                if m_above.vs != 0.0 and layer.is_fluid:
                    name = 'cmb'
                elif m_above.vs == 0.0 and not layer.is_fluid:
                    name = 'icb'
                elif not moho_found and m_above.vp < 7500. <= vp_below:
                    name = 'moho'
                    moho_found = True

                if name is not None:
                    zmin, zmax = discontinuity_depth_ranges[name]
                    if not (zmin <= (r_surface - r_top)*km <= zmax):
                        name = None

                if name is None and num.allclose(
                        nodes[0][1], m_above.astuple(), rtol=1e-6, atol=0.):
                    nodes = nodes[1:]

            for r, p in nodes:
                lines.append(((r_surface - r)*km, cake.Material(*p), name))
                name = None

        return cake.LayeredModel.from_scanlines(lines)

    @classmethod
    def from_cake(
            cls, mod,
            nlayers_max=gemini_nlayer,
            rtol_velocity=2e-3,
            rtol_q=0.1,
            nsamples_per_layer=20):

        '''
        Approximate :py:class:`pyrocko.cake.LayeredModel` by GEMINI layers.

        The model is split at its discontinuities. The material properties in
        each part are fitted with polynomials of up to third order in the
        normalized radius, the quality factors are averaged. As long as the
        total number of layers stays at most ``nlayers_max``, the worst
        fitting part is split further, at the node minimizing the misfit. The
        misfit is the maximum of the relative misfit of velocities and density
        divided by ``rtol_velocity`` and the relative deviation of the quality
        factors from the layer's constant value divided by ``rtol_q``. A
        warning is issued if it exceeds 1 in the end.
        '''

        sublayers = [
            layer for layer in mod.layers() if layer.zbot > layer.ztop]

        if not sublayers:
            raise GeminiError('Cannot convert empty earth model.')

        r_surface = cake.earthradius / km
        # cake models usually stop 1 m above the centre
        if abs(sublayers[-1].zbot / km - r_surface) > 0.1:
            raise GeminiError(
                'Earth model must extend to the centre of the earth (to a '
                'depth of %g km) to be used with GEMINI.' % r_surface)

        if sublayers[0].ztop != 0.0:
            raise GeminiError('Earth model must start at the surface.')

        if sublayers[0].mtop.vs == 0.0:
            raise GeminiError(
                'GEMINI does not support a fluid top layer (ocean).')

        # Group the cake layers into parts between discontinuities.
        parts = []
        for layer in sublayers:
            if parts and parts[-1][-1].zbot == layer.ztop and \
                    parts[-1][-1].mbot.astuple() == layer.mtop.astuple():
                parts[-1].append(layer)
            else:
                parts.append([layer])

        if len(parts) > nlayers_max:
            raise GeminiError(
                'Earth model has %i discontinuities, more than GEMINI\'s '
                'limit of %i layers.' % (len(parts)-1, nlayers_max))

        fits = [cls._fit_part(
            part, r_surface, rtol_velocity, rtol_q, nsamples_per_layer)
            for part in parts]

        while len(fits) < nlayers_max:
            imax = max(range(len(fits)), key=lambda i: fits[i]['misfit'])
            fit = fits[imax]
            if fit['misfit'] == 0.0 or len(fit['part']) < 2:
                break

            best = None
            for isplit in range(1, len(fit['part'])):
                fits_split = [
                    cls._fit_part(
                        part, r_surface, rtol_velocity, rtol_q,
                        nsamples_per_layer)
                    for part in (fit['part'][:isplit], fit['part'][isplit:])]

                misfit = max(f['misfit'] for f in fits_split)
                if best is None or misfit < best[0]:
                    best = misfit, fits_split

            fits[imax:imax+1] = best[1]

        misfit = max(fit['misfit'] for fit in fits)
        if misfit > 1.0:
            logger.warning(
                'GEMINI earth model approximation exceeds tolerances by a '
                'factor of %g, consider increasing the number of layers or '
                'the tolerances.' % misfit)

        layers = [fit['layer'] for fit in fits[::-1]]
        layers[0].r_bottom = 0.0
        return cls(layers, r_surface, reference_frequency=1.0, comments=[
            ' Isotropic earth model for GEMINI, approximating a layered model',
            ' by polynomials of the normalized radius x = r / r_surface.'])

    @staticmethod
    def _fit_part(part, r_surface, rtol_velocity, rtol_q, nsamples_per_layer):
        z = num.unique(num.concatenate([
            num.linspace(layer.ztop, layer.zbot, nsamples_per_layer+1)
            for layer in part]))

        materials = []
        for zz in z:
            for layer in part:
                if layer.ztop <= zz <= layer.zbot:
                    materials.append(layer.material(zz))
                    break

        vp, vs, rho, qp, qs = num.array(
            [m.astuple() for m in materials]).T

        vp, vs, rho = vp/km, vs/km, rho/km
        x = (r_surface - z/km) / r_surface
        is_fluid = num.all(vs == 0.0)

        nkinks = sum(
            1 for layer in part
            if isinstance(layer, cake.GradientLayer))

        degree = min(gemini_ncoefs - 1, nkinks)

        def fit(values):
            if degree == 0 or num.all(values == values[0]):
                return num.array([values.mean()]), \
                    num.full_like(values, values.mean())

            # Fit in scaled domain for numerical stability, then convert to
            # coefficients of powers of x.
            p = Polynomial.fit(x, values, degree)
            coefs = p.convert().coef
            coefs = num.concatenate(
                [coefs, num.zeros(degree + 1 - coefs.size)])

            return coefs, Polynomial(coefs)(x)

        misfits = []
        columns = []
        for values in (rho, vp, vs):
            coefs, values_fit = fit(values)
            columns.append(coefs)
            scale = max(num.abs(values).max(), 1e-3)
            misfits.append(
                num.abs(values_fit - values).max() / scale / rtol_velocity)

        ncoefs = max(c.size for c in columns)
        coefs = num.zeros((ncoefs, 6))
        crho, cvp, cvs = [
            num.concatenate([c, num.zeros(ncoefs - c.size)]) for c in columns]

        coefs[:, 0] = crho
        coefs[:, 1] = coefs[:, 2] = cvp
        if not is_fluid:
            coefs[:, 3] = coefs[:, 4] = cvs

        coefs[0, 5] = 1.0

        # Constant quality factors, averaging the attenuation 1/Q.
        qkappas = num.array([m.qk() for m in materials])
        qkappas = num.where(
            num.isfinite(qkappas) & (qkappas > 0.0),
            num.minimum(qkappas, qkappa_default),
            qkappa_default)

        qkappa = 1.0 / num.mean(1.0/qkappas)
        if is_fluid:
            qmu = -1.0
        else:
            qmu = 1.0 / num.mean(1.0/qs)
            misfits.append(num.abs(qmu/qs - 1.0).max() / rtol_q)

        layer = GeminiLayer(r_surface - part[-1].zbot/km, coefs, qmu, qkappa)

        return dict(part=part, layer=layer, misfit=max(misfits))

    def max_deviation(self, mod, ddepth=10.*km):
        '''
        Get maximum relative deviation from an isotropic cake model.

        Compares the isotropic approximation of this model to ``mod`` in
        terms of P- and S-wave velocities and density, at regularly sampled
        depths, avoiding the immediate vicinity of discontinuities.

        :returns: maximum relative deviation and depth [m] where it occurs
        '''

        dev_max = 0.0
        z_dev_max = 0.0
        for layer, r_bottom, r_top in self.layer_bounds():
            z_top = (self.r_surface - r_top) * km
            z_bottom = (self.r_surface - r_bottom) * km
            n = max(2, int(math.ceil((z_bottom - z_top) / ddepth)))
            eps = 1e-3 * (z_bottom - z_top)
            z = num.linspace(z_top + eps, z_bottom - eps, n+1)
            props = num.array(self.isotropic_properties(
                layer, self.r_surface - z/km))

            for zz, (vp, vs, rho) in zip(z, props[:3].T):
                m = mod.material(zz)
                for a, b in ((vp, m.vp), (vs, m.vs), (rho, m.rho)):
                    # floor: 1 km/s or 1 g/cm^3 (shear velocity is 0 in
                    # fluids)
                    dev = abs(a - b) / max(abs(b), 1000.)
                    if dev > dev_max:
                        dev_max = dev
                        z_dev_max = zz

        return dev_max, z_dev_max


class GeminiEarthmodel(Object):
    '''
    Earth model in GEMINI's native file format.
    '''

    dummy_for = GeminiModel
    dummy_for_description = 'pyrocko.fomosto.gemini.GeminiModel'

    class __T(TBase):
        def regularize_extra(self, val):
            if isinstance(val, str):
                val = GeminiModel.from_string(val)

            return val

        def to_save(self, val):
            return literal(val.to_string())


class GeminiConfig(Object):
    '''
    Configuration of the GEMINI backend, stored in ``extra/gemini``.
    '''

    gemini_version = String.T(
        optional=True,
        help='Version of the GEMINI programs to use (required, set by '
             '``fomosto init``).')

    earthmodel = GeminiEarthmodel.T(
        optional=True,
        help='Earth model in GEMINI\'s native format (radius in [km], density '
             'in [g/cm^3], velocities in [km/s], coefficients of polynomials '
             'in the normalized radius r/r_surface), possibly transversely '
             'isotropic. If not given, it is fitted to the isotropic '
             '``earthmodel_1d`` of the store\'s config.')

    time_region = Tuple.T(
        2, gf.Timing.T(),
        default=(gf.Timing('begin-50'), gf.Timing('end+100')),
        help='Time window to be covered by the computed seismograms.')

    cut = Tuple.T(
        2, gf.Timing.T(),
        optional=True,
        help='Time window to which the Green\'s functions are cut before '
             'they are stored.')

    time_window_factor = Float.T(
        default=2.0,
        help='Factor to enlarge the length of the FFT time window relative '
             'to ``time_region`` (it is further enlarged to a power of two '
             'samples), to keep energy arriving after the time window from '
             'wrapping around into it.')

    damping_time = Float.T(
        optional=True,
        help='Damping time of the complex frequency [s], to suppress '
             'time-domain aliasing. Default is a fifth of the FFT time window '
             'length.')

    frequency_max = Float.T(
        optional=True,
        help='Maximum frequency [Hz]. Default is just below the Nyquist '
             'frequency of the store.')

    slowness_max = Float.T(
        default=0.6,
        help='Maximum horizontal slowness [s/km], determining the maximum '
             'harmonic degree to compute at each frequency. Truncation of the '
             'harmonic degree range produces an artificial arrival with this '
             'slowness, so it should be above the slowness of the slowest '
             'phases of interest and arrive after the end of the time '
             'window.')

    harmonic_degree_max = Int.T(
        default=gemini_lmx,
        help='Absolute limit for the harmonic degree.')

    degree_taper_length = Int.T(
        default=40,
        help='Length of the cosine taper applied at the upper end of the '
             'harmonic degree range by Dispec. The degree range is '
             'enlarged by this amount.')

    include_physical_dispersion = Bool.T(
        default=True,
        help='Whether to use anelastic, frequency-dependent, complex moduli '
             '(at the reference frequency of the model). If false, GEMINI '
             'computes purely elastic, unattenuated seismograms.')

    accuracy = Float.T(
        default=1e-4,
        help='Accuracy of the Bulirsch-Stoer integration in GEMINI.')

    lowpass_order = Int.T(
        default=0,
        help='Order of Butterworth low-pass filter to apply (0: no '
             'filter).')

    lowpass_corner = Float.T(
        default=1.0,
        help='Corner frequency of Butterworth low-pass filter [Hz].')

    nlayers_max = Int.T(
        default=gemini_nlayer,
        help='Maximum number of earth model layers supported by the GEMINI '
             'executable.')

    keep_basis_solutions = Bool.T(
        default=False,
        help='Whether to keep the files with basis solutions computed by '
             'GEMINI (in the store\'s ``gemini_green`` directory) after the '
             'store has been built.')


def _cmt_lines(m6_use, depth):
    '''
    Source file in Harvard CMT format, as read by Dispec's ``readcmt.f``.

    The moment tensor components are given in units of 1e15 Nm, the source is
    located at the north pole.
    '''

    mrr, mss, mee, mrs, mre, mse = m6_use

    lat, lon = 90.0, 0.0
    if depth > 9999.9:
        raise GeminiError('Source depth too large for CMT file format.')

    for v in m6_use:
        if abs(v) >= 1000.:
            raise GeminiError('Moment tensor too large for CMT file format.')

    return [
        '%-8s %2s/%2s/%2s %2s:%2s:%4.1f%7.2f%8.2f%6.1f%3.1f%3.1f%-24s' % (
            'FOMOSTO', '01', '01', '00', '00', '00', 0.0, lat, lon, depth,
            0.0, 0.0, 'FOMOSTO GEMINI'),
        'PDE BW:%2i%3i%4i MW:%2i%3i%4i DT=%6.1f%4.1f%7.2f%5.2f%8.2f%5.2f'
        '%6.1f%5.1f' % (
            0, 0, 0, 0, 0, 0, 0.0, 0.0, lat, 0.0, lon, 0.0, depth, 0.0),
        ' DUR%4.1f EX %2i' % (0.0, cmt_exponent) + ''.join(
            '%6.2f%5.2f' % (v, 0.0) for v in (mrr, mss, mee, mrs, mre, mse)),
        '%7.2f%3i%4i' % (0.0, 0, 0) * 3 + '%7.2f' % 0.0
        + '%4i%3i%5i' % (0, 0, 0) * 2]


def _station_lines(distances):
    '''
    Station file in IRIS format for receivers on the 180 degree meridian.

    With the source at the north pole, Dispec measures the epicentral
    distance as the receiver's colatitude and the azimuth as 180 degrees minus
    the receiver's longitude, which is then zero.
    '''

    lines = [
        'Station file written by fomosto',
        '',
        '  year station network lat lon elev site']

    for i, distance in enumerate(distances):
        lat = 90.0 - distance * cake.m2d
        lines.append(
            '00 R%04i XX %.6f %.6f 0.0 fomosto' % (i, lat, 180.0))

    return lines


def _window_lines(frequency_max, slowness_max, r_surface, l_add, l_max):
    '''
    Degree limits in frequency-degree domain, as read by ``gtllim.f``.

    The maximum degree is ``2 pi f r_surface slowness_max + l_add``, limited
    to ``l_max``. ``gtllim.f`` interpolates linearly between the given
    frequencies.
    '''

    lines = ['2']
    for f in (0.0, frequency_max):
        lmax = min(
            l_max,
            int(math.ceil(2.0*math.pi * f * r_surface * slowness_max)) + l_add)

        lines.append('%.10g 0 %i' % (f*1000., lmax))

    return lines


def _read_seismogr(fn, nreceivers):
    '''
    Read Totido's ASCII output, as written by ``writeAscii.f``.

    :returns: list of arrays of shape ``(3, nsamples)`` with the columns
        Z, NS, EW (which are r, theta, phi with Dispec's default of
        source-centred coordinates).
    '''

    with open(fn, 'r') as f:
        lines = f.read().splitlines()

    try:
        toks = lines[1].split()
        nsamples = int(toks[1])
        samprat = float(toks[3])
        timeshift = float(toks[5])

        iline = 2
        data = []
        for ireceiver in range(nreceivers):
            assert lines[iline].startswith('RECLat')
            block = lines[iline+2:iline+2+nsamples]
            assert len(block) == nsamples
            data.append(num.array(
                [_fortran_floats(line.split()[1:4]) for line in block],
                dtype=float).T)

            iline += 2 + nsamples

    except (IndexError, ValueError, AssertionError):
        raise GeminiError('Could not read Totido output file: %s' % fn)

    return samprat, timeshift, data


class _Slots(object):
    '''
    Limits the number of jobs running in parallel.

    Waiting jobs get a free slot in order of priority (lowest first), and
    in the order of their requests for equal priority.
    '''

    def __init__(self, nslots):
        self._free = nslots
        self._waiting = []
        self._counter = itertools.count()

    @contextlib.asynccontextmanager
    async def __call__(self, priority=0):
        if self._free > 0 and not self._waiting:
            self._free -= 1
        else:
            future = asyncio.get_running_loop().create_future()
            heapq.heappush(
                self._waiting, (priority, next(self._counter), future))

            try:
                await future
            except asyncio.CancelledError:
                if future.done() and not future.cancelled():
                    # slot was handed over just before the cancellation
                    self._release()

                raise

        try:
            yield
        finally:
            self._release()

    def _release(self):
        while self._waiting:
            _, _, future = heapq.heappop(self._waiting)
            if not future.done():
                future.set_result(None)
                return

        self._free += 1


_program_names = {
    'gemini': 'Gemini',
    'dispec': 'Dispec',
    'totido': 'Totido'}


class _Runner(object):
    '''
    Runs the GEMINI programs as asyncio subprocesses.

    At most ``nworkers`` jobs run in parallel. A job holds one of the slots
    given by :py:meth:`slot` and may run several programs one after the other.
    The programs are started in their own sessions, so that they do not
    receive the terminal's signals, and are terminated by the runner when the
    coroutine running them is cancelled.
    '''

    def __init__(self, nworkers, bins):
        self._bins = bins
        self.slot = _Slots(nworkers)
        self._procs = {}

    def kill_all(self):
        for proc in list(self._procs):
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    async def _terminate(self, proc, timeout=5.0):
        try:
            proc.terminate()
        except ProcessLookupError:
            return

        try:
            await asyncio.wait_for(proc.wait(), timeout)
        except asyncio.TimeoutError:
            logger.warning(
                'Killing "%s" (pid %i), which did not terminate on SIGTERM.'
                % (self._procs.get(proc, 'program'), proc.pid))

            try:
                proc.kill()
            except ProcessLookupError:
                pass

            await proc.wait()

    async def run(
            self, program, input_str, cwd, expected_outputs, on_line=None,
            on_start=None):

        '''
        Run program. The caller must hold a slot.

        ``on_start`` is called once the process has been started,
        ``on_line`` for every line of its standard output.
        '''

        exe = self._bins[program]
        try:
            proc = await asyncio.create_subprocess_exec(
                exe,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                start_new_session=True)

        except OSError as e:
            raise GeminiError('''Could not start %s executable "%s" (%s).
Available fomosto backends and download links to the modelling codes are listed
on

      https://pyrocko.org/docs/current/apps/fomosto/backends.html
''' % (_program_names[program], exe, e))

        self._procs[proc] = exe
        if on_start:
            on_start()

        stderr_task = asyncio.ensure_future(proc.stderr.read())
        try:
            proc.stdin.write(input_str.encode('ascii'))
            try:
                await proc.stdin.drain()
                proc.stdin.close()
            except (BrokenPipeError, ConnectionResetError):
                pass

            output = []
            async for line in proc.stdout:
                line = line.decode('ascii', errors='replace')
                output.append(line)
                if on_line:
                    on_line(line)

            error_str = (await stderr_task).decode(
                'ascii', errors='replace')

            await proc.wait()

        finally:
            if proc.returncode is None:
                await self._terminate(proc)

            if not stderr_task.done():
                stderr_task.cancel()
                await asyncio.wait([stderr_task])

            del self._procs[proc]

        output_str = ''.join(output)
        name = _program_names[program]
        logger.debug(
            '===== begin %s output (%s) =====\n%s===== end %s output ====='
            % (program, cwd, output_str, program))

        errmess = []
        if proc.returncode != 0:
            errmess.append(
                '%s had a non-zero exit state (%i).'
                % (name, proc.returncode))

        if any(line.startswith(('STOP', 'Fortran runtime error'))
               for line in error_str.splitlines()):
            errmess.append('%s stopped with an error.' % name)

        if output_str.find('An error occured') != -1:
            errmess.append(
                'The string "An error occured" appeared in %s output.'
                % name)

        for fn in expected_outputs:
            if not os.path.exists(pjoin(cwd, fn)):
                errmess.append(
                    '%s did not produce output file "%s".' % (name, fn))

        if errmess:
            raise GeminiError('''
===== begin %s input =====
%s===== end %s input =====
===== begin %s output =====
%s===== end %s output =====
===== begin %s error =====
%s===== end %s error =====
%s
%s has been invoked as "%s" in directory "%s".'''.lstrip() % (
                program, input_str, program,
                program, output_str, program,
                program, error_str, program,
                '\n'.join(errmess),
                name, exe, cwd))

        return output_str


async def _gather(aws):
    '''
    Run awaitables concurrently, all or nothing.

    If one of them fails, or if this coroutine is cancelled, the others are
    cancelled, and their cleanup is waited for, before the error is raised.
    '''

    tasks = [asyncio.ensure_future(aw) for aw in aws]
    if not tasks:
        return []

    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    finally:
        for task in tasks:
            task.cancel()

        await asyncio.wait(tasks)

    for task in tasks:
        if not task.cancelled() and task.exception() is not None:
            raise task.exception()

    return [task.result() for task in tasks]


class _Tempdir(object):
    def __init__(self, prefix, dir=None, keep=False):
        self.prefix = prefix
        self.dir = dir
        self.keep = keep

    def __enter__(self):
        self.path = mkdtemp(prefix=self.prefix, dir=self.dir)
        return self.path

    def __exit__(self, *args):
        if self.keep:
            logger.warning(
                'Not removing temporary directory: %s' % self.path)
        else:
            shutil.rmtree(self.path)


class GeminiGFBuilder(object):
    '''
    Builds a GF store with GEMINI.

    Step 1 runs ``Gemini`` for each source depth (one block per depth). Step 2
    runs ``Dispec`` and ``Totido`` for each of the elementary moment tensor
    sources (one block per source depth and range of up to 149 distances) and
    stores the Green's functions.
    '''

    nsteps = 2

    seismogram_types = {
        'displacement': 'd',
        'velocity': 'v',
        'acceleration': 'a'}

    def __init__(self, store_dir, force=False, tmp=None, keep_tmp=False):
        if tmp is None:
            tmp = gf.builder.store_tmp_dir(store_dir)

        self.store_dir = store_dir
        self.force = force
        self.tmp = tmp
        self.keep_tmp = keep_tmp

        self.store = gf.store.Store(store_dir, 'w')
        try:
            self._setup()
        except Exception:
            self.store.close()
            raise

    def close(self):
        self.store.close()

    def _setup(self):
        store_config = self.store.config
        self.conf = conf = self.store.get_extra('gemini')

        if conf.gemini_version is None:
            raise GeminiError(
                '"gemini_version" must be set in "extra/gemini", use one of: '
                '%s' % ', '.join(sorted(variants)))

        if conf.gemini_version not in program_bins:
            raise GeminiError(
                'Unsupported GEMINI version: %s' % conf.gemini_version)

        if not isinstance(store_config, gf.meta.ConfigTypeA):
            raise GeminiError(
                'GEMINI backend only supports stores of type A (receivers '
                'at the surface).')

        if store_config.receiver_depth != 0.0:
            raise GeminiError(
                'GEMINI only computes Green\'s functions for receivers at '
                'the surface, "receiver_depth" must be 0.')

        component_scheme = store_config.component_scheme
        stored_quantity = store_config.effective_stored_quantity

        if component_scheme not in ('elastic10', 'elastic8'):
            raise GeminiError(
                'GEMINI backend does not support component scheme "%s".'
                % component_scheme)

        if stored_quantity not in self.seismogram_types:
            raise GeminiError(
                'GEMINI backend does not support stored quantity "%s".'
                % stored_quantity)

        self.seismogram_type = self.seismogram_types[stored_quantity]

        # Dispec's output components (r, theta, phi) are up, away from the
        # source and anticlockwise around the source. Fomosto's (N, E, D)
        # with the receiver north of the source are away from the source,
        # clockwise around the source and down.
        self.gfmapping = [
            (MomentTensor(m=symmat6(1, 0, 0, 1, 0, 0)),
             {'N': 0, 'E': 3, 'D': 5}),
            (MomentTensor(m=symmat6(0, 0, 0, 0, 1, 1)),
             {'N': 1, 'E': 4, 'D': 6}),
            (MomentTensor(m=symmat6(0, 0, 1, 0, 0, 0)),
             {'N': 2, 'D': 7}),
            (MomentTensor(m=symmat6(0, 1, 0, 0, 0, 0)),
             {'N': 8, 'D': 9}),
        ]

        if component_scheme == 'elastic8':
            self.gfmapping = self.gfmapping[:3]

        if conf.earthmodel is None:
            logger.info(
                'No GEMINI earth model given in "extra/gemini", fitting it to '
                'the store\'s "earthmodel_1d".')

            model = GeminiModel.from_cake(
                store_config.earthmodel_1d, nlayers_max=conf.nlayers_max)
        else:
            model = conf.earthmodel
            dev, z = model.max_deviation(store_config.earthmodel_1d)
            if dev > 0.02:
                logger.warning(
                    'The isotropic approximation of the GEMINI earth model '
                    'in "extra/gemini" deviates from the store\'s '
                    '"earthmodel_1d" by up to %.1f%% (at %g km depth), which '
                    'is used for travel time tables. To fix, use the output '
                    'of\n\n    python -m pyrocko.fomosto.gemini gemini-to-nd '
                    '%s\n\nas "earthmodel_1d" in the store\'s config, and run '
                    '"fomosto ttt" again.'
                    % (dev*100., z/km,
                       pjoin(self.store_dir, 'extra', 'gemini')))

        model.validate(nlayers_max=conf.nlayers_max)
        self.model = model

        if abs(model.r_surface - cake.earthradius/km) > 1e-3:
            logger.warning(
                'Surface radius of GEMINI earth model (%g km) differs from '
                'the radius used to convert surface distances to angles '
                '(%g km).' % (model.r_surface, cake.earthradius/km))

        if conf.degree_taper_length > dispec_ntmx:
            raise GeminiError(
                '"degree_taper_length" must be at most %i.' % dispec_ntmx)

        self._setup_timing()

        self.window_str = '\n'.join(_window_lines(
            self.frequency_max, conf.slowness_max, model.r_surface,
            conf.degree_taper_length + 10,
            min(gemini_lmx, conf.harmonic_degree_max))) + '\n'

        self.model_str = model.to_string()

        self.gf_directory = pjoin(self.store_dir, 'gemini_green')
        util.ensuredir(self.gf_directory)

        ndistances = store_config.ndistances
        nblocks = (ndistances - 1) // dispec_nseimx + 1
        nper = (ndistances - 1) // nblocks + 1
        self.distance_blocks = [
            (i*nper, min((i+1)*nper, ndistances)) for i in range(nblocks)]

    def _setup_timing(self):
        conf = self.conf
        deltat = self.store.config.deltat

        d = self.store.make_timing_params(
            conf.time_region[0], conf.time_region[1], force=self.force)

        tmin = math.floor(d['tmin'] / deltat) * deltat
        tmax = math.ceil(d['tmax'] / deltat) * deltat

        nsamples_out = int(round((tmax - tmin) / deltat)) + 1
        nsamples = trace.nextpow2(
            max(4, int(math.ceil(nsamples_out * conf.time_window_factor))))

        nsamples_out = min(nsamples, nsamples_out)

        tlen = nsamples * deltat

        nend_max = nsamples // 2 - 1
        if conf.frequency_max is None:
            nend = nend_max
        else:
            nend = int(round(conf.frequency_max * tlen))
            if nend > nend_max:
                raise GeminiError(
                    '"frequency_max" (%g Hz) must be below the Nyquist '
                    'frequency (%g Hz) of the store.' % (
                        conf.frequency_max, 0.5 / deltat))

        if nend < 1:
            raise GeminiError('"frequency_max" is too low.')

        # Totido extends the spectrum to 2**n samples and zero-pads by
        # another factor 2**zero_padding, giving 2*nzh samples.
        nzh = trace.nextpow2(nend + 1)
        zero_padding = int(round(math.log2(nsamples // (2*nzh))))
        assert 2 * nzh * 2**zero_padding == nsamples

        if nsamples > totido_nfmax:
            raise GeminiError(
                'Time window needs %i samples, Totido supports at most %i. '
                'Reduce "time_region" or the store\'s sampling rate.' % (
                    nsamples, totido_nfmax))

        if nend > dispec_nfr:
            raise GeminiError(
                'Calculation needs %i frequencies, Dispec supports at most '
                '%i. Reduce "time_region" or "frequency_max".'
                % (nend, dispec_nfr))

        self.tmin = tmin
        self.nsamples_out = nsamples_out
        self.tlen = tlen
        self.nend = nend
        self.frequency_max = nend / tlen
        self.zero_padding = zero_padding
        self.damping_time = conf.damping_time or tlen / 5.0

        logger.info(
            'GEMINI time window length: %g s, %i samples, starting at %g s, '
            '%i frequencies up to %g Hz, damping time: %g s.'
            % (tlen, nsamples, tmin, nend, self.frequency_max,
               self.damping_time))

    def depths(self):
        return self.store.config.coords[0]

    def nblocks(self, step):
        ndepths = len(self.depths())
        if step == 0:
            return ndepths
        else:
            return ndepths * len(self.distance_blocks)

    def step2_block(self, iblock):
        idepth, idistblock = divmod(iblock, len(self.distance_blocks))
        return idepth, idistblock

    def gemini_input(self, depth):
        conf = self.conf
        return '\n'.join(str(x) for x in [
            3,
            0,
            '%.10g' % self.tlen,
            '%.10g' % self.damping_time,
            0,
            '%.10g' % (self.frequency_max * 1000.),
            int(conf.include_physical_dispersion),
            0,
            min(gemini_lmx, conf.harmonic_degree_max),
            1,
            '%.10g' % (depth / km),
            '%g' % conf.accuracy,
            'earthmodel',
            'window',
            'basis',
            1]) + '\n'

    def basis_path(self, depth):
        return pjoin(self.gf_directory, 'basis_%gkm' % (depth / km))

    def basis_signature(self, depth):
        h = hashlib.sha1()
        for s in (self.gemini_input(depth), self.model_str, self.window_str):
            h.update(s.encode('ascii'))

        return h.hexdigest()

    def have_basis(self, depth):
        fn = self.basis_path(depth)
        try:
            with open(fn + '.signature', 'r') as f:
                signature = f.read().strip()
        except OSError:
            return False

        return os.path.exists(fn) and signature == self.basis_signature(depth)

    def remove_basis(self, depth):
        fn = self.basis_path(depth)
        for path in (fn, fn + '.signature'):
            if os.path.exists(path):
                os.unlink(path)

    async def run_gemini(self, runner, depth, label):
        if self.have_basis(depth):
            logger.info(
                '%s: skipping, basis solutions exist already.' % label)
            return

        self.remove_basis(depth)
        tbeg = [None]
        tlast = [None]

        def on_start():
            tbeg[0] = tlast[0] = time.time()
            logger.info('%s: starting Gemini...' % label)

        def on_line(line):
            toks = line.split()
            if len(toks) == 6 and time.time() - tlast[0] > 30.:
                try:
                    nw = int(toks[1])
                except ValueError:
                    return

                tlast[0] = time.time()
                logger.info('%s: Gemini at frequency %i / %i.' % (
                    label, nw, self.nend))

        async with runner.slot(priority=1):
            with _Tempdir('tmp-', dir=self.gf_directory) as rundir:
                for fn, s in [
                        ('earthmodel', self.model_str),
                        ('window', self.window_str)]:

                    with open(pjoin(rundir, fn), 'w') as f:
                        f.write(s)

                await runner.run(
                    'gemini', self.gemini_input(depth), rundir, ['basis'],
                    on_line=on_line, on_start=on_start)

                fn = self.basis_path(depth)
                os.replace(pjoin(rundir, 'basis'), fn)
                with open(fn + '.signature', 'w') as f:
                    f.write(self.basis_signature(depth) + '\n')

        logger.info('%s: done with Gemini, wallclock time: %.0f s.' % (
            label, time.time() - tbeg[0]))

    def dispec_input(self):
        return '\n'.join(str(x) for x in [
            'basis',
            'source',
            'm',
            2,
            'window',
            self.conf.degree_taper_length,
            'stations',
            3,
            'all',
            0]) + '\n'

    def totido_input(self):
        conf = self.conf
        if conf.lowpass_order > 0:
            lowpass = '1 %i %.10g' % (conf.lowpass_order, conf.lowpass_corner)
        else:
            lowpass = '0'

        return '\n'.join(str(x) for x in [
            'spec3k',
            '',
            '%.10g' % self.tmin,
            lowpass,
            '0',
            self.zero_padding,
            self.seismogram_type,
            '%.10g' % ((self.nsamples_out + 0.5) * self.store.config.deltat),
            'a']) + '\n'

    async def run_dispec_totido(
            self, runner, depth, distances, mt, on_start=None):

        # Elementary moment tensor in units of cmt_moment.
        m6 = mt.m6_up_south_east()

        # Dispec and Totido run in the same slot, so that at most one Dispec
        # output file per slot exists. Step 2 jobs have precedence over step
        # 1 jobs, to keep the number of unprocessed basis solutions low.
        async with runner.slot(priority=0):
            util.ensuredir(self.tmp)
            with _Tempdir(
                    'geminirun-', dir=self.tmp, keep=self.keep_tmp) as rundir:

                os.symlink(
                    os.path.abspath(self.basis_path(depth)),
                    pjoin(rundir, 'basis'))

                for fn, lines in [
                        ('window', [self.window_str]),
                        ('source', _cmt_lines(m6, depth/km)),
                        ('stations', _station_lines(distances))]:

                    with open(pjoin(rundir, fn), 'w') as f:
                        f.write('\n'.join(lines).rstrip('\n') + '\n')

                await runner.run(
                    'dispec', self.dispec_input(), rundir, ['spec3k'],
                    on_start=on_start)

                await runner.run(
                    'totido', self.totido_input(), rundir, ['seismogr'])

                samprat, timeshift, data = _read_seismogr(
                    pjoin(rundir, 'seismogr'), len(distances))

        deltat = self.store.config.deltat
        if abs(1.0 / samprat - deltat) > 1e-4 * deltat:
            raise GeminiError(
                'Totido output sampling rate (%g Hz) does not match the '
                'store\'s sampling rate (%g Hz).'
                % (samprat, self.store.config.sample_rate))

        if abs(timeshift - self.tmin) > 1e-3 * deltat + 1e-6 * abs(self.tmin):
            raise GeminiError('Totido output has unexpected time shift.')

        factor = nm / cmt_moment
        traces = []
        for distance, (ur, utheta, uphi) in zip(distances, data):
            for comp, ydata in [
                    ('N', utheta),
                    ('E', -uphi),
                    ('D', -ur)]:

                traces.append(trace.Trace(
                    '', '', '', comp,
                    tmin=self.tmin,
                    deltat=deltat,
                    ydata=ydata * factor,
                    meta=dict(distance=distance)))

        return traces

    async def run_block(self, runner, iblock, label):
        idepth, idistblock = self.step2_block(iblock)
        depth = self.depths()[idepth]
        ia, ib = self.distance_blocks[idistblock]
        distances = self.store.config.coords[1][ia:ib]

        if not self.have_basis(depth):
            raise GeminiError(
                'Basis solutions for source depth %g km are missing, run '
                'step 1 first.' % (depth / km))

        tbeg = []

        def on_start():
            # first of the programs of this block has been started
            if not tbeg:
                tbeg.append(time.time())
                logger.info('%s: starting Dispec/Totido...' % label)

        results = await _gather(
            self.run_dispec_totido(runner, depth, distances, mt, on_start)
            for (mt, _) in self.gfmapping)

        self.put_traces(depth, results)

        logger.info('%s: done with Dispec/Totido, wallclock time: %.0f s.' % (
            label, time.time() - tbeg[0]))

    def put_traces(self, depth, results):
        store = self.store
        conf = self.conf
        skipped = 0
        duplicate_inserts = 0
        store.lock()
        try:
            for (_, gfmap), traces in zip(self.gfmapping, results):
                for tr in traces:
                    if tr.channel not in gfmap:
                        continue

                    args = (depth, tr.meta['distance'], gfmap[tr.channel])

                    if conf.cut:
                        tmin = store.t(conf.cut[0], args[:-1])
                        tmax = store.t(conf.cut[1], args[:-1])
                        if None in (tmin, tmax):
                            skipped += 1
                            continue

                        tr.chop(tmin, tmax)

                    try:
                        store.put(args, gf.store.GFTrace.from_trace(tr))
                    except gf.store.DuplicateInsert:
                        duplicate_inserts += 1

        finally:
            store.unlock()

        if skipped:
            logger.warning(
                'Could not determine cut time window for %i traces, not '
                'stored.' % skipped)

        if duplicate_inserts:
            logger.warning(
                'Skipped %i duplicate insertions.' % duplicate_inserts)

    def block_label(self, step, iblock):
        return 'Step %i / %i, block %i / %i' % (
            step+1, self.nsteps, iblock+1, self.nblocks(step))

    async def run(self, runner, jobs, on_done, remove_basis=False):
        '''
        Run the given jobs, ``(step, iblock)`` pairs.

        Step 2 blocks wait for the step 1 block of their source depth, if it
        is among the jobs. With ``remove_basis``, the basis solutions of a
        source depth are removed when its step 2 jobs are done. This is only
        safe if the jobs include all of its step 2 blocks not done yet.
        '''

        jobs = set(jobs)
        ndistblocks = len(self.distance_blocks)

        async def depth_pipeline(idepth):
            depth = self.depths()[idepth]
            if (0, idepth) in jobs:
                label = self.block_label(0, idepth)
                await self.run_gemini(runner, depth, label)
                on_done(0, idepth)

            iblocks = [
                ib for ib in range(
                    idepth*ndistblocks, (idepth+1)*ndistblocks)
                if (1, ib) in jobs]

            async def block(ib):
                await self.run_block(runner, ib, self.block_label(1, ib))
                on_done(1, ib)

            await _gather(block(ib) for ib in iblocks)

            if remove_basis:
                self.remove_basis(depth)

        idepths = sorted(set(
            iblock if step == 0 else self.step2_block(iblock)[0]
            for (step, iblock) in jobs))

        await _gather(depth_pipeline(idepth) for idepth in idepths)


async def _build_async(builder, jobs, on_done, nworkers, remove_basis):
    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()
    runner = _Runner(nworkers, program_bins[builder.conf.gemini_version])
    received = []

    def handler(signum):
        name = signal.Signals(signum).name
        if not received:
            logger.warning(
                'Received %s, terminating running programs and cleaning up.'
                % name)
            main_task.cancel()
        else:
            logger.warning(
                'Received %s again, killing running programs.' % name)
            runner.kill_all()

        received.append(signum)

    signums = [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]
    for signum in signums:
        loop.add_signal_handler(signum, handler, signum)

    try:
        await builder.run(runner, jobs, on_done, remove_basis)

    except asyncio.CancelledError:
        if received:
            raise Interrupted()

        raise

    finally:
        for signum in signums:
            loop.remove_signal_handler(signum)


def build(
        store_dir,
        force=False,
        nworkers=None,
        continue_=False,
        step=None,
        iblock=None):

    '''
    Build GF store with GEMINI.

    Follows the conventions of the ``build`` method of
    :py:class:`pyrocko.gf.builder.Builder` regarding the ``continue_``,
    ``step`` and ``iblock`` arguments. Step 1 computes the basis solutions
    with ``Gemini`` (one block per source depth), step 2 computes and stores
    the Green's functions with ``Dispec`` and ``Totido``. Without ``step``,
    both steps run as a pipeline.
    '''

    nsteps = GeminiGFBuilder.nsteps
    if step is not None and not (0 <= step < nsteps):
        raise gf.store.StoreError('Invalid step: %i' % (step+1))

    if iblock is not None and step is None:
        raise gf.store.StoreError('Option "--step" must be given.')

    if not continue_ and iblock in (None, -1) and step in (None, 0):
        gf.store.Store.create_dependants(store_dir, force)

    status_fn = pjoin(store_dir, '.status')
    done = set()
    if iblock is None:
        if not continue_:
            with open(status_fn, 'w'):
                pass
        else:
            try:
                with open(status_fn, 'r') as status:
                    for line in status:
                        done.add(tuple(int(x) for x in line.split()))
            except IOError:
                raise gf.store.StoreError('Nothing to continue.')

    builder = GeminiGFBuilder(store_dir, force=force)
    try:
        steps = list(range(nsteps)) if step is None else [step]
        jobs = []
        for istep in steps:
            nblocks = builder.nblocks(istep)
            if iblock in (None, -1):
                jobs.extend(
                    (istep, i) for i in range(nblocks)
                    if (istep, i) not in done)
            else:
                if not (0 <= iblock < nblocks):
                    raise gf.store.StoreError(
                        'Invalid block index: %i' % (iblock+1))

                jobs.append((istep, iblock))

        if iblock == -1:
            for istep, i in jobs:
                c = ['fomosto', 'build']
                if not os.path.samefile(store_dir, '.'):
                    c.append("'%s'" % store_dir)

                c.append('--step=%i' % (istep+1))
                c.append('--block=%i' % (i+1))
                print(' '.join(c))

            return

        def on_done(istep, i):
            if iblock is None:
                with open(status_fn, 'a') as status:
                    status.write('%i %i\n' % (istep, i))

            done.add((istep, i))

        remove_basis = step is None and iblock is None \
            and not builder.conf.keep_basis_solutions

        asyncio.run(_build_async(
            builder, jobs, on_done, nworkers or os.cpu_count() or 1,
            remove_basis))

        if step is None and iblock is None:
            # basis solutions may be left over from interrupted builds
            if not builder.conf.keep_basis_solutions:
                for depth in builder.depths():
                    builder.remove_basis(depth)

                try:
                    os.rmdir(builder.gf_directory)
                except OSError:
                    pass

    finally:
        builder.close()

    if iblock is None:
        os.remove(status_fn)
        gf.builder.remove_store_tmp_dir(store_dir)


def init(store_dir, variant, config_params=None, earthmodel=None):
    '''
    Initialize GF store for GEMINI.

    :param config_params: dict with parameters to override in the store's
        config
    :param earthmodel: :py:class:`GeminiModel`, possibly transversely
        isotropic, to be used. If given, the store's ``earthmodel_1d`` is
        set to its isotropic approximation, unless explicitly given in
        ``config_params``. Otherwise, the GEMINI earth model is fitted to
        ``earthmodel_1d``.
    '''

    check_variant('gemini', variant, variants)

    store_id = os.path.basename(os.path.realpath(store_dir))

    d = dict(
        id=store_id,
        ncomponents=10,
        component_scheme='elastic10',
        stored_quantity=None,
        sample_rate=0.2,
        receiver_depth=0*km,
        source_depth_min=10*km,
        source_depth_max=20*km,
        source_depth_delta=10*km,
        distance_min=100*km,
        distance_max=1000*km,
        distance_delta=10*km,
        earthmodel_1d=cake.load_model(),
        modelling_code_id='gemini.%s' % variant,
        tabulated_phases=[
            gf.meta.TPDef(
                id='begin',
                definition='p,P,p\\,P\\,Pv_(cmb)p'),
            gf.meta.TPDef(
                id='end',
                definition='2.5'),
            gf.meta.TPDef(
                id='P',
                definition='!P'),
            gf.meta.TPDef(
                id='S',
                definition='!S'),
            gf.meta.TPDef(
                id='p',
                definition='!p'),
            gf.meta.TPDef(
                id='s',
                definition='!s')])

    if earthmodel is not None:
        earthmodel.validate()
        d['earthmodel_1d'] = earthmodel.to_cake()

    if config_params is not None:
        d.update(config_params)

    config = gf.meta.ConfigTypeA(**d)
    config.validate()

    if earthmodel is None:
        earthmodel = GeminiModel.from_cake(config.earthmodel_1d)

    gemini = GeminiConfig(gemini_version=variant, earthmodel=earthmodel)

    gemini.cut = gemini.time_region

    return gf.store.Store.create_editables(
        store_dir,
        config=config,
        extra={'gemini': gemini})


def _load_gemini_model(fn):
    with open(fn, 'r') as f:
        s = f.read()

    if s.startswith('--- !pf.GeminiConfig'):
        from pyrocko.guts import load_string
        conf = load_string(s)
        if conf.earthmodel is None:
            raise GeminiError('No earth model given in file: %s' % fn)

        return conf.earthmodel

    return GeminiModel.from_string(s)


def main(args=None):
    '''
    Command line tool to convert earth models between formats.
    '''

    import sys
    from pyrocko.cli import PyrockoArgumentParser

    def add_help(parser):
        parser.add_argument(
            '--help', '-h',
            action='help',
            help='Show this help message and exit.')

    parser = PyrockoArgumentParser(
        add_help=False,
        prog='python -m pyrocko.fomosto.gemini',
        description='Convert earth models between Pyrocko\'s .nd format and '
                    'GEMINI\'s native polynomial format.')

    add_help(parser)

    subparsers = parser.add_subparsers(
        dest='command', required=True, metavar='SUBCOMMAND',
        title='Subcommands')

    p = subparsers.add_parser(
        'nd-to-gemini',
        add_help=False,
        help='Fit GEMINI model to layered earth model.',
        description='Approximate a layered earth model by polynomial layers '
                    'for GEMINI. The result can be used as ``earthmodel`` in '
                    'a store\'s ``extra/gemini`` file.')

    p.add_argument(
        'input', metavar='IN',
        help='Earth model in ``.nd`` format, or name of an earth model '
             'built into Pyrocko, e.g. ```ak135-f-continental.m```.')

    p.add_argument(
        'output', metavar='OUT', nargs='?',
        help='Output file. Default: standard output.')

    p.add_argument(
        '--nlayers-max', type=int, default=gemini_nlayer, metavar='N',
        help='Maximum number of layers. Default: %(default)s.')

    add_help(p)

    p = subparsers.add_parser(
        'gemini-to-nd',
        add_help=False,
        help='Get isotropic approximation of GEMINI model.',
        description='Get the isotropic approximation of a, possibly '
                    'transversely isotropic, GEMINI earth model in ``.nd`` '
                    'format. The result can be used as ``earthmodel_1d`` in a '
                    'store\'s ``config`` file.')

    p.add_argument(
        'input', metavar='IN',
        help='Earth model in GEMINI\'s native format, or a store\'s '
             '``extra/gemini`` file.')

    p.add_argument(
        'output', metavar='OUT', nargs='?',
        help='Output file. Default: standard output.')

    add_help(p)

    args = parser.parse_args(args)

    util.setup_logging('gemini', 'warning')

    try:
        if args.command == 'nd-to-gemini':
            try:
                mod = cake.load_model(args.input)
            except ValueError:
                raise GeminiError(
                    'Cannot read "%s" as earth model in .nd format.'
                    % args.input)

            s = GeminiModel.from_cake(
                mod, nlayers_max=args.nlayers_max).to_string()

        else:
            s = cake.write_nd_model_str(
                _load_gemini_model(args.input).to_cake())

    except (GeminiError, OSError, cake.CakeError) as e:
        parser.exit(1, '%s: error: %s\n' % (parser.prog, e))

    if args.output:
        with open(args.output, 'w') as f:
            f.write(s)
    else:
        sys.stdout.write(s)


if __name__ == '__main__':
    main()
