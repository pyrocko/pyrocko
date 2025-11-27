import numpy as num
import os.path as op

from matplotlib import pyplot as plt, cm, colors

from pyrocko import gf

km = 1e3
d2r = num.pi / 180.
r2d = 180. / num.pi

# The store we are going extract data from:
store_id = 'gf_abruzzo_nearfield_vmod_Ameri'

# First, download a Greens Functions store. If you already have one that you
# would like to use, you can skip this step and point the *store_superdirs* in
# the next step to that directory.
if not op.exists(store_id):
    gf.ws.download_gf_store(site='pyrocko', store_id=store_id)

# We need a pyrocko.gf.Engine object which provides us with the traces
# extracted from the store. In this case we are going to use a local
# engine since we are going to query a local store.
engine = gf.LocalEngine(store_superdirs=['.'])

# The dynamic parameter used for discretization of the PseudoDynamicRupture are
# extracted from the stores config file.
store = engine.get_store(store_id)

# Let's define the source now with its extension, orientation etc.
source_params = dict(
    north_shift=2. * km,
    east_shift=2. * km,
    depth=1.0 * km,
    length=6. * km,
    width=6. * km,
    strike=0.,
    dip=80.,
    rake=45.,
    anchor='top',
    decimation_factor=1)

dyn_rupture = gf.PseudoDynamicRupture(
    nx=5,
    ny=5,
    pure_shear=True,
    **source_params)

# Recalculate slip, that rupture magnitude fits given magnitude
magnitude = 6.0
dyn_rupture.rescale_slip(magnitude=magnitude, store=store)

# Get rake out of slip (can differ from traction rake!)
slip = dyn_rupture.get_slip()
source_params['rake'] = num.arctan2(slip[0, 1], slip[0, 0]) * r2d

# Create similar rectangular source model with rake derivded from slip
rect_rupture = gf.RectangularSource(
    magnitude=magnitude,
    **source_params)

# Define static target grid to extract the surface displacement
ngrid = 40

obs_size = 10. * km
ntargets = ngrid**2

norths = num.linspace(-obs_size, obs_size, ngrid) + \
    source_params['north_shift']
easts = num.linspace(-obs_size, obs_size, ngrid) + \
    source_params['east_shift']

norths2d = num.repeat(norths, len(easts))
easts2d = num.tile(easts, len(norths))

static_target = gf.StaticTarget(
    lats=num.ones(norths2d.size) * dyn_rupture.effective_lat,
    lons=num.ones(norths2d.size) * dyn_rupture.effective_lon,
    north_shifts=norths2d,
    east_shifts=easts2d,
    interpolation='nearest_neighbor',
    store_id=store_id)

# Get static surface displacements for rectangular and pseudo dynamic source
result = engine.process(rect_rupture, static_target)

targets_static = result.request.targets_static
synth_disp_rect = result.results_list[0][0].result

result = engine.process(dyn_rupture, static_target)

targets_static = result.request.targets_static
synth_disp_dyn = result.results_list[0][0].result

# Extract static vertical displacement and plot
down_rect = synth_disp_rect['displacement.d']
down_dyn = synth_disp_dyn['displacement.d']
down_diff = down_rect - down_dyn

vabsmax = num.max(num.abs([down_rect, down_dyn, down_diff]))
vmin = -vabsmax
vmax = vabsmax

fig = plt.figure(figsize=(10, 10))
axes = []
for i in [1, 2, 3]:
    axes.append(fig.add_subplot(2, 2, i, aspect=1.0))

cax = fig.add_axes((0.6, 0.125, 0.02, 0.3))

cmap = 'RdBu_r'
norm = colors.Normalize(vmin=vmin, vmax=vmax)

for ax, (down, label) in zip(
        axes[:3],
        zip((down_rect, down_dyn, down_diff),
            (r'$u_{Z, rect}$', r'$u_{Z, dyn}$', r'$\Delta u_{Z}$'))):

    ax.pcolormesh(
        easts/km, norths/km, down.reshape(ngrid, ngrid),
        cmap=cmap, norm=norm, shading='gouraud')

    ax.set_title(label)

axes[1].set_xlabel('Easting [km]')
axes[2].set_xlabel('Easting [km]')
axes[0].set_ylabel('Northing [km]')
axes[2].set_ylabel('Northing [km]')

axes[0].get_xaxis().set_tick_params(
    bottom=True, labelbottom=False, top=False, labeltop=False)

axes[1].get_yaxis().set_tick_params(
    left=True, right=False, labelleft=False, labelright=False)

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # If not set, an error might be issued

cbar = fig.colorbar(sm, cax=cax)
cbar.ax.set_ylabel('$u$ [m]')

fig.savefig('gf_forward_pseudo_rupture_static.png')

plt.show()
