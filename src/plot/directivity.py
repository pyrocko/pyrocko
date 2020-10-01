import logging
import numpy as num
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter

from pyrocko.plot import beachball
from pyrocko.gf.meta import Timing
from pyrocko.gf import LocalEngine, Target, RectangularSource


km = 1e3
r2d = 180. / num.pi
d2r = num.pi / 180.

logger = logging.getLogger(__name__)


def get_azimuthal_targets(
        store_id, radius,
        azi_begin=0., azi_end=360., dazi=1.,
        interpolation='multilinear',
        components='RTZ'):

    assert dazi > 0.
    assert azi_begin < azi_end

    nstations = int((azi_end - azi_begin) // dazi)
    assert nstations > 0

    azimuths = num.linspace(azi_begin, azi_end, nstations)

    coords = num.zeros((2, nstations))
    coords[0, :] = num.cos(azimuths*d2r)
    coords[1, :] = num.sin(azimuths*d2r)
    coords *= radius

    dips = {'R': 0., 'T': 0., 'Z': -90.}
    for comp in components:
        assert comp in dips.keys()

    target_kwargs = dict(
        quantity='displacement',
        interpolation=interpolation,
        store_id=store_id)

    targets = [
        Target(
            north_shift=coords[0, iazi],
            east_shift=coords[1, iazi],
            azimuth={
                'R': azi,
                'T': azi+90.,
                'Z': 0.
            }[channel],
            dip=dips[channel],
            codes=('', 'S%01d' % iazi, '', channel),
            **target_kwargs)
        for iazi, azi in enumerate(azimuths)
        for channel in components]


    for target, azi in zip(targets, azimuths):
        target.azimuth = azi
        target.dazi = dazi

    return targets, azimuths


def get_seismogram_array(response, fmin=None, fmax=None, component='R'):
    resp = response
    assert len(resp.request.sources) == 1, 'more than one source in response'

    ntraces = len(resp.request.targets)
    tmin = None
    tmax = None
    traces = []

    for _, target, tr in response.iter_results():
        if target.codes[-1] != component:
            continue
        assert hasattr(target, 'azimuth')
        assert target.dazi

        if fmin and fmax:
            tr.bandpass(2, fmin, fmax)
        elif fmin:
            tr.highpass(4, fmin)
        elif fmax:
            tr.lowpass(4, fmax)

        tmin = min(tmin, tr.tmin) if tmin else tr.tmin
        tmax = max(tmax, tr.tmax) if tmax else tr.tmax
        traces.append(tr)

    for tr in traces:
        tr.extend(tmin, tmax, fillmethod='repeat')

    data = num.array([tr.get_ydata() for tr in traces])
    nsamples = data.shape[1]
    return data, num.linspace(tmin, tmax, nsamples)


def hillshade(array, azimuth, angle_altitude):
    azimuth = 360.0 - azimuth
    azi = azimuth * r2d
    alt = angle_altitude * r2d

    x, y = num.gradient(array)
    slope = num.pi/2. - num.arctan(num.sqrt(x*x + y*y))
    aspect = num.arctan2(-x, y)

    shaded = num.sin(alt)*num.sin(slope) \
        + num.cos(alt)*num.cos(slope)*num.cos((azi - num.pi/2.) - aspect)

    return (shaded + 1.)/2.


def hillshade_seismogram_array(
        seismogram_array, rgba_map,
        shad_lim=(.4, .98), contrast=1., blend_mode='multiply'):
    assert blend_mode in ('multiply', 'screen'), 'unknown blend mode'
    assert shad_lim[0] < shad_lim[1], 'bad shading limits'
    from scipy.ndimage import convolve as im_conv
    # Light source from somewhere above - psychologically the best choice
    # from upper left
    ramp = num.array([[1., 0.], [0., -1.]]) * contrast

    # convolution of two 2D arrays
    shad = im_conv(seismogram_array, ramp.T).ravel()
    shad *= -1.

    # if there are strong artifical edges in the data, shades get
    # dominated by them. Cutting off the largest and smallest 2% of
    # # shades helps
    percentile2 = num.quantile(shad, 0.02)
    percentile98 = num.quantile(shad, 0.98)
    shad[shad > percentile98] = percentile98
    shad[shad < percentile2] = percentile2

    # # normalize shading
    shad -= num.nanmin(shad)
    shad /= num.nanmax(shad)

    # # reduce range to balance gray color
    shad *= shad_lim[1] - shad_lim[0]
    shad += shad_lim[0]

    if blend_mode == 'screen':
        rgba_map[:, :3] = 1. - ((1. - rgba_map[:, :3]) * (shad[:, num.newaxis]))
    elif blend_mode == 'multiply':
        rgba_map[:, :3] *= shad[:, num.newaxis]

    return rgba_map


def plot_directivity(
        engine, source, store_id,
        distance=300*km, azi_begin=0., azi_end=360., dazi=1.,
        phase_begin='first{stored:any_P}-10%',
        phase_end='last{stored:any_S}+50',
        component='R', fmin=0.01, fmax=.1,
        nthreads=0, hillshade=True, cmap='seismic',
        plot_mt=True, show_annotations=True,
        axes=None):

    if axes is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
    else:
        fig = ax.figure

    targets, azimuths = get_azimuthal_targets(
        store_id, distance, azi_begin, azi_end, dazi, components='R')
    store = engine.get_store(store_id)

    resp = engine.process(rect_source, targets, nthreads=nthreads)
    data, times = get_seismogram_array(resp, fmin, fmax)

    timing_begin = Timing(phase_begin)
    timing_end = Timing(phase_end)

    tbegin = store.t(timing_begin, (source.depth, distance))
    tend = store.t(timing_end, (source.depth, distance))

    tsel = num.logical_and(times >= tbegin, times <= tend)

    data = data[:, tsel].T
    times = times[tsel]
    duration = times[-1] - times[0]

    vmax = num.abs(data).max()
    cmw = ScalarMappable(cmap=cmap)
    cmw.set_clim(-vmax, vmax)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    def r_fmt(v, p):
        if v < tbegin or v > tend:
            return ''
        return '%g s' % v

    ax.yaxis.set_major_formatter(FuncFormatter(r_fmt))
    ax.set_rlim(times[-1] + .3*duration, times[0])

    if plot_mt:
        mt = source.pyrocko_moment_tensor(store=store, target=targets[0])
        beachball.plot_beachball_mpl(
            mt, ax,
            beachball_type='deviatoric', size=.15,
            size_units='axes', color_t='slategray',
            position=(.5, .5), linewidth=1.)

    mesh = ax.pcolormesh(
        azimuths * d2r, times, data,
        cmap='seismic', shading='gouraud', vmin=-vmax, vmax=vmax)

    if hillshade:
        mesh.update_scalarmappable()
        color = mesh.get_facecolor()
        color = hillshade_seismogram_array(
            data, color, shad_lim=(.85, 1.), blend_mode='multiply')
        mesh.set_facecolor(color)

    if show_annotations:
        _phase_begin = Timing(phase_begin)
        _phase_end = Timing(phase_end)

        for p in (_phase_begin, _phase_end):
            p.offset = 0.
            p.offset_is_slowness = False
            p.offset_is_percent = False

        tphase_first = store.t(_phase_begin, (source.depth, distance))
        tphase_last = store.t(_phase_end, (source.depth, distance))

        theta = num.linspace(0, 2*num.pi, 360)
        tfirst = num.full_like(theta, tphase_first)
        tlast = num.full_like(theta, tphase_last)

        ax.plot(theta, tfirst, color='k', alpha=.3, lw=1.)
        ax.plot(theta, tlast, color='k', alpha=.3, lw=1.)

        ax.text(
            num.pi*3/2, tphase_first, '|'.join(_phase_begin.phase_defs),
            ha='left', color='k')

        ax.text(
            num.pi*5/3, tphase_last, '|'.join(_phase_end.phase_defs),
            ha='right', color='k')



    if axes is None:
        plt.show()
    return resp


if __name__ == '__main__':
    engine = LocalEngine(store_superdirs=['.'], use_config=True)

    rect_source = RectangularSource(
        depth=2.6*km,
        strike=240.,
        dip=76.6,
        rake=-.4,
        anchor='top',

        nucleation_x=-.57,
        nucleation_y=-.59,
        velocity=2070.,

        length=27*km,
        width=9.4*km,
        slip=1.4)


    resp = plot_directivity(engine, rect_source, 'crust2_ib', dazi=5, component='T')
