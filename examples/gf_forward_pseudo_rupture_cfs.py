'''
Coulomb Failure Stress (CFS) change calculation from pseudo-dynamic rupture.
'''
import os.path as op
import numpy as num

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from pyrocko import gf, orthodrome as pod
from pyrocko.plot import mpl_init, mpl_papersize

# The store we are going extract data from:
store_id = 'gf_abruzzo_nearfield_vmod_Ameri'

# First, download a Greens Functions store. If you already have one that you
# would like to use, you can skip this step and point the *store_superdirs* in
# the next step to that directory.
if not op.exists(store_id):
    gf.ws.download_gf_store(site='kinherd', store_id=store_id)

# We need a pyrocko.gf.Engine object which provides us with the traces
# extracted from the store. In this case we are going to use a local
# engine since we are going to query a local store.
engine = gf.LocalEngine(store_superdirs=['.'])

# The dynamic parameter used for discretization of the PseudoDynamicRupture are
# extracted from the stores config file.
store = engine.get_store(store_id)

# Let's define the source now with its extension, orientation etc.
source = gf.PseudoDynamicRupture(
    lat=37.718, lon=37.487, depth=5.3e3,
    length=213.8e3, width=21.3e3,
    strike=57.,
    dip=82.,
    rake=28.,
    slip=4.03,
    anchor='top',
    nx=6,
    ny=4,
    pure_shear=True,
    smooth_rupture=True)

# Calculate the subfault specific parameters
source.discretize_patches(store)

# Let's now define the target source now with its extension, orientation etc.
target = gf.PseudoDynamicRupture(
    lat=37.992, lon=37.262, depth=4.7e3,
    length=92.9e3, width=17.2e3,
    strike=-92.,
    dip=73.,
    rake=-8.,
    slip=7.07,
    # nucleation_x=-1.,
    # nucleation_y=0.,
    anchor='top',
    nx=6,
    ny=4,
    pure_shear=True,
    smooth_rupture=True)

# Define the receiver point locations, where the CFS will be calculated - here
# as a grid of (northing, easting, depth)
nnorths = 100
neasts = 100
norths = num.linspace(-200., 200., nnorths) * 1e3
easts = num.linspace(-200., 200., neasts) * 1e3
depth_target = 10e3

receiver_points = num.zeros((nnorths * neasts, 3))
receiver_points[:, 0] = num.repeat(norths, neasts)
receiver_points[:, 1] = num.tile(easts, nnorths)
receiver_points[:, 2] = num.ones(nnorths * neasts) * depth_target

# Calculate the Coulomb Failure Stress change (CFS) for the given target plane
strike_target = target.strike
dip_target = target.dip
rake_target = target.rake

cfs = source.get_coulomb_failure_stress(
    receiver_points, friction=0.6, pressure=0.,
    strike=strike_target, dip=dip_target, rake=rake_target, nthreads=2)

# Plot the results as a map
mpl_init(fontsize=12.)
fig, axes = plt.subplots(figsize=mpl_papersize('a5'))

# Plot of the Coulomb Failure Stress changes
mesh = axes.pcolormesh(
    easts / 1e3, norths / 1e3,
    cfs.reshape(neasts, nnorths) / 1e6,
    cmap='RdBu_r',
    shading='gouraud',
    norm=colors.SymLogNorm(
        linthresh=0.03, linscale=0.03, vmin=-1., vmax=1.))

# Plot the source plane as grey shaded area
fn, fe = source.outline(cs='xy').T
axes.fill(
    fe / 1e3, fn / 1e3,
    edgecolor=(0., 0., 0.),
    facecolor='grey',
    alpha=0.7)
axes.plot(fe[0:2] / 1e3, fn[0:2] / 1e3, 'k', linewidth=1.3)

# Plot the target plane as grey shaded area
north_shift, east_shift = pod.latlon_to_ne(
    source.lat, source.lon,
    target.lat, target.lon)
fn, fe = target.outline(cs='xy').T
fn += north_shift
fe += east_shift
axes.fill(
    fe / 1e3, fn / 1e3,
    edgecolor=(0., 0., 0.),
    facecolor='grey',
    alpha=0.7)
axes.plot(fe[0:2] / 1e3, fn[0:2] / 1e3, 'k', linewidth=1.3)

# Plot labeling
axes.set_xlabel('East shift [km]')
axes.set_ylabel('North shift [km]')
axes.set_title(
    f'Target plane: strike {strike_target:.0f}$^\\circ$, ' +
    f'dip {dip_target:.0f}$^\\circ$, ' +
    f'rake {rake_target:.0f}$^\\circ$, depth {depth_target/1e3:.0f} km')

cbar = fig.colorbar(mesh, ax=axes)
cbar.set_label(r'$\Delta$ CFS [MPa]')
cbar_ticks = [-1., -0.5, -0.25, -0.1, 0., 0.1, 0.25, 0.5, 1.]
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels([f'{tick:.2f}' for tick in cbar_ticks])

fig.savefig('gf_forward_pseudo_rupture_cfs.png')

plt.show()
