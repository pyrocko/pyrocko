# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Green's function store builders interfacing with the external modelling codes.

This subpackage contains the GF builders which are used by :py:app:`fomosto` to
interface with the various supported numerical forward modelling tools
(:doc:`/apps/fomosto/backends`). '''

import os
import shutil
import importlib

import numpy as num

op = os.path


modules = [m for m in os.listdir(op.dirname(__file__))
           if m.endswith('.py') and m not in ('__init__.py', 'dummy.py')]
AVAILABLE_BACKENDS = set([op.splitext(m)[0] for m in modules])


def format_variants(name, variants):
    '''
    Get comma-separated list of the modelling code IDs of a backend.
    '''
    return ', '.join(
        name if v is None else '%s.%s' % (name, v)
        for v in sorted(variants, key=lambda v: (v is not None, v)))


def check_variant(name, variant, variants):
    '''
    Check that the variant of a modelling backend is supported.

    To avoid ambiguities, there is no default variant: for backends with
    variants, it must always be given.

    :param name: name of the backend
    :param variant: requested variant, ``None`` if none was given
    :param variants: the backend's ``variants`` dict
    :raises: :py:exc:`~pyrocko.gf.error.StoreError` if the variant is not
        supported
    '''

    if variant in variants:
        return

    from pyrocko.gf.store import StoreError

    if variant is None:
        raise StoreError(
            'The variant of backend "%s" must be given, use one of: %s'
            % (name, format_variants(name, variants)))
    else:
        raise StoreError(
            'Unsupported variant "%s" of backend "%s", use one of: %s'
            % (variant, name, format_variants(name, variants)))


def correct_rectangle_rule(ydata, deltat):
    '''
    Undo the time shift and amplitude error of a rectangle rule integration.

    QSSP and QSEIS integrate their output with a running sum, so that
    ``ydata[k] = deltat * sum(v[:k])``, where ``v`` are exact samples of the
    time derivative. Compared to the exact integral, this delays the result
    by half a sample and amplifies it by ``x / sin(x)``, with ``x = pi * f *
    deltat`` (up to a factor of 1.57 at the Nyquist frequency). This
    function applies the inverse filter ``exp(i x) * sin(x) / x``.

    The trace is padded with its first and last values before filtering, so
    that static offsets are preserved.

    :param ydata: samples of the rectangle rule integration
    :type ydata: :py:class:`numpy.ndarray`
    :param deltat: sampling interval [s]
    :returns: corrected samples, same size as ``ydata``
    '''

    n = ydata.size
    nfft = 1
    while nfft < 3 * n:
        nfft *= 2

    npad_begin = n
    ypad = num.empty(nfft)
    ypad[:npad_begin] = ydata[0]
    ypad[npad_begin:npad_begin+n] = ydata
    ypad[npad_begin+n:] = ydata[-1]

    x = num.pi * num.fft.rfftfreq(nfft, deltat) * deltat
    filt = num.exp(1j * x) * num.sinc(x / num.pi)
    if nfft % 2 == 0:
        # keep Nyquist coefficient real
        filt[-1] = filt[-1].real

    y = num.fft.irfft(num.fft.rfft(ypad) * filt, nfft)
    return y[npad_begin:npad_begin+n]


def get_backend_variants():
    '''
    Get the modelling backend variants which can be used with ``fomosto init``.

    Backends declare their variants in a module-level dict ``variants``,
    mapping the variant name (``None`` for a backend without variants) to the
    list of executables it needs.

    :returns: list of ``(modelling_code_id, missing)`` tuples, where
        ``missing`` is the list of required executables which are not found
        in the ``PATH``, or ``None`` if the backend module cannot be
        imported.
    '''

    result = []
    for name in sorted(AVAILABLE_BACKENDS):
        try:
            module = importlib.import_module('pyrocko.fomosto.' + name)
        except ImportError:
            result.append((name, None))
            continue

        if not hasattr(module, 'init'):
            continue

        variants = getattr(module, 'variants', {None: []})
        for variant in sorted(variants, key=lambda v: (v is not None, v)):
            modelling_code_id = name if variant is None \
                else '%s.%s' % (name, variant)

            missing = [
                exe for exe in variants[variant] if shutil.which(exe) is None]

            result.append((modelling_code_id, missing))

    return result
