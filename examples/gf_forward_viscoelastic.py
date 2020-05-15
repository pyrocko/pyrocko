'''
An advanced example requiring a viscoelastic static store.
See https://pyrocko.org for detailed instructions.
'''
import logging
import os.path as op
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter

import numpy as num
from pyrocko import gf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('static-viscoelastic')

km = 1e3
d2r = num.pi/180.
store_id = 'static_t'

day = 3600*24

engine = gf.LocalEngine(store_superdirs=['.'])

ngrid = 100

extent = (-75*km, 75*km)
dpx = abs((extent[0] - extent[1]) / ngrid)

easts = num.linspace(*extent, ngrid)
norths = num.linspace(*extent, ngrid)
east_shifts = num.tile(easts, ngrid)
north_shifts = num.repeat(norths, ngrid)

look = 23.
heading = -76.  # descending
npx = int(ngrid**2)

tmax = 1000*day
t_eq = 200*day
nsteps = 200

dt = tmax / nsteps

satellite_target = gf.SatelliteTarget(
    east_shifts=east_shifts,
    north_shifts=north_shifts,
    phi=num.full(npx, d2r*(90.-look)),
    theta=num.full(npx, d2r*(-90.-heading)),
    interpolation='nearest_neighbor')

creep_source = gf.RectangularSource(
    lat=0., lon=0.,
    north_shift=3*km, east_shift=0.,
    depth=25*km,
    width=15*km, length=80*km,
    strike=100., dip=90., rake=0.,
    slip=0.08, anchor='top',
    decimation_factor=4)

coseismic_sources = gf.RectangularSource(
    lat=0., lon=0.,
    north_shift=3*km, east_shift=0.,
    depth=15*km,
    width=10*km, length=60*km,
    strike=100., dip=90., rake=0.,
    slip=.5, anchor='top',
    decimation_factor=4,
    time=t_eq)

targets = []
for istep in range(nsteps):
    satellite_target = gf.SatelliteTarget(
        east_shifts=east_shifts,
        north_shifts=north_shifts,
        phi=num.full(npx, d2r*(90.-look)),
        theta=num.full(npx, d2r*(-90.-heading)),
        interpolation='nearest_neighbor',
        tsnapshot=dt*istep)
    targets.append(satellite_target)


def get_displacement(sources, targets, component='los'):
    result = engine.process(
        sources, targets,
        nthreads=0)

    static_results = result.static_results()
    nres = len(static_results)

    res_arr = num.empty((nres, ngrid, ngrid))

    for ires, res in enumerate(static_results):
        res_arr[ires] = res.result['displacement.%s' % component]\
            .reshape(ngrid, ngrid)
    return res_arr


component = 'los'
fn = 'displacement_%s' % component

# Use cached displacements
if not op.exists('%s.npy' % fn):
    logger.info('Calculating scenario for %s.npy ...', fn)
    displacement_creep = get_displacement(
        creep_source, satellite_target, component)[0]
    displacement_creep /= 365.

    displacement = get_displacement(coseismic_sources, targets, component)

    for istep in range(nsteps):
        displacement[istep] += displacement_creep * (dt / day) * istep

    num.save(fn, displacement)
else:
    logger.info('Loading scenario data from %s.npy ...', fn)
    displacement = num.load('%s.npy' % fn)

if False:
    fig = plt.figure()
    ax = fig.gca()

    for ipx in range(ngrid)[::10]:
        for ipy in range(ngrid)[::10]:
            ax.plot(displacement[:, ipx, ipy], alpha=.3, color='k')

    plt.show()


sample_point = (-20.*km, -27.*km)
sample_idx = (int(sample_point[0] / dpx), int(sample_point[1] / dpx))

fig, (ax_time, ax_u) = plt.subplots(
    2, 1, gridspec_kw={'height_ratios': [1, 4]})
fig.set_size_inches(10, 8)
ax_u = fig.gca()

vrange = num.abs(displacement).max()
colormesh = ax_u.pcolormesh(
    easts, norths, displacement[80],
    cmap='seismic', vmin=-vrange, vmax=vrange, shading='gouraud',
    animated=True)

smpl_point = ax_u.scatter(
    *sample_point, marker='x', color='black', s=30, zorder=30)

time_label = ax_u.text(.95, .05, '0 days', ha='right', va='bottom',
                       alpha=.5, transform=ax_u.transAxes, zorder=20)

# cbar = fig.colorbar(img)
# cbar.set_label('Displacment %s [m]', )

ax_u.set_xlabel('Easting [km]')
ax_u.set_ylabel('Northing [km]')

km_formatter = FuncFormatter(lambda x, pos: x / km)
ax_u.xaxis.set_major_formatter(km_formatter)
ax_u.yaxis.set_major_formatter(km_formatter)

ax_time.set_title('%s Displacement' % component.upper())
urange = (displacement[:, sample_idx[0], sample_idx[1]].min() * 1.05,
          displacement[:, sample_idx[0], sample_idx[1]].max() * 1.05)
ut = ax_time.plot([], [], color='black')[0]
ax_time.axvline(x=t_eq + dt, linestyle='--', color='red', alpha=.5)

day_formatter = FuncFormatter(lambda x, pos: int(x / day))
ax_time.xaxis.set_major_formatter(day_formatter)
ax_time.set_xlim(0., tmax)
ax_time.set_ylim(*urange)

ax_time.set_xlabel('Days')
ax_time.set_ylabel('Displacement [m]')
ax_time.grid(alpha=.3)


def animation_update(frame):
    colormesh.set_array(displacement[frame].ravel())
    time_label.set_text('%d days' % int(frame * (dt / day)))

    ut.set_xdata(num.arange(frame) * dt)
    ut.set_ydata(displacement[:frame, sample_idx[0], sample_idx[1]])

    return colormesh, smpl_point, time_label, ut


ani = FuncAnimation(
    fig, animation_update, frames=nsteps, interval=30, blit=True)
# plt.show()

logger.info('Saving animation...')
ani.save('viscoelastic-response.mp4', writer='ffmpeg')
