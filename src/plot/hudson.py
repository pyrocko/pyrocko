# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
from __future__ import absolute_import
import numpy as num

from pyrocko import moment_tensor as pmt


def project(mt):
    '''
    Calculate Hudson's (u, v) coordinates for a given moment tensor.

    The moment tensor can be given as a
    :py:class:`pyrocko.moment_tensor.MomentTensor` object, or by anything that
    can be converted to a 3x3 NumPy matrix, or as the six independent moment
    tensor entries as ``(mnn, mee, mdd, mne, mnd, med)``.
    '''

    mt = pmt.values_to_matrix(mt)
    eig_m = pmt.eigh_check(mt)[0]
    m3, m2, m1 = eig_m / num.max(num.abs(eig_m))
    u = -2./3. * (m1 + m3 - 2.0*m2)
    v = 1./3. * (m1 + m2 + m3)
    return u, v


def draw_axes(axes, color='black', fontsize=12, linewidth=1.5):

    '''
    Plot axes and annotations of Hudson's MT decomposition diagram.
    '''

    axes.set_axis_off()
    axes.set_aspect(1.0)

    axes.set_xlim(-4./3.-0.1, 4./3.+0.1)
    axes.set_ylim(-1.1, 1.1)

    axes.plot(
        [-4./3., 0., 4./3., 0., -4/3.],
        [-1./3., -1., 1./3., 1., -1./3.],
        zorder=-1,
        linewidth=linewidth,
        color=color)

    axes.plot(
        [-1.0, 1.0],
        [0., 0.],
        zorder=-1,
        linewidth=linewidth,
        color=color)

    axes.plot(
        [0., 0.],
        [-1., 1.],
        zorder=-1,
        linewidth=linewidth,
        color=color)

    d = fontsize/3.
    for txt, pos, off, va, ha in [
            ('+Isotropic', (0., 1.), (-d, d), 'bottom', 'right'),
            ('-Isotropic', (0., -1.), (d, -d), 'top', 'left'),
            ('-CLVD', (+1.0, 0.), (d, -d), 'top', 'left'),
            ('+CLVD', (-1.0, 0.), (-d, d), 'bottom', 'right')]:

        axes.plot(
            pos[0], pos[1], 'o',
            color=color,
            markersize=fontsize/2.)

        axes.annotate(
            txt,
            xy=pos,
            xycoords='data',
            xytext=off,
            textcoords='offset points',
            verticalalignment=va,
            horizontalalignment=ha,
            rotation=0.)

    for txt, pos, off, va, ha in [
            ('-Dipole', (2./3., -1./3.), (d, -d), 'top', 'left'),
            ('+Dipole', (-2./3., 1./3.), (-d, d), 'bottom', 'right'),
            ('-Crack', (4./9., -5./9.), (d, -d), 'top', 'left'),
            ('+Crack', (-4./9., 5./9.), (-d, d), 'bottom', 'right')]:

        axes.plot(
            pos[0], pos[1], 'o',
            color=color,
            markersize=fontsize/2.)

        axes.annotate(
            txt,
            xy=pos,
            xycoords='data',
            xytext=off,
            textcoords='offset points',
            verticalalignment=va,
            horizontalalignment=ha,
            rotation=0.)
