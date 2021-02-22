import os.path
from pyrocko import gf
import numpy as num

# Download a Greens Functions store, programmatically.
store_id = 'gf_abruzzo_nearfield_vmod_Ameri'
if not os.path.exists(store_id):
    gf.ws.download_gf_store(site='kinherd', store_id=store_id)

# Setup the LocalEngine and point it to the fomosto store you just downloaded.
# *store_superdirs* is a list of directories where to look for GF Stores.
engine = gf.LocalEngine(store_superdirs=['.'])

# We define an extended source, in this case a rectangular geometry
# Centroid UTM position is defined relatively to geographical lat, lon position
# Horizontal closing sill with an N104W azimuth.
# Slip is split to shear and tensile slip where "opening_fraction" determines
# the direction and amount of opening/closing defined from -1, 1
# for a pure shear dislocation "opening_fraction" is 0.

km = 1e3  # for convenience
d2r = num.pi / 180.

rect_source = gf.RectangularSource(
    lat=0., lon=0.,
    north_shift=0., east_shift=0., depth=2.5*km,
    width=4*km, length=8*km,
    dip=0., rake=0., strike=104.,
    slip=3., opening_fraction=-1.)

# We will define a grid of targets
# number in east and north directions, and total
ngrid = 80

# extension from origin in all directions
obs_size = 20.*km
ntargets = ngrid**2

# make regular line vector
norths = num.linspace(-obs_size, obs_size, ngrid)
easts = num.linspace(-obs_size, obs_size, ngrid)

# make regular grid
norths2d = num.repeat(norths, easts.size)
easts2d = num.tile(easts, norths.size)

# We initialize the satellite target and set the line of sight vectors
# direction, example of the Envisat satellite
look = 23.     # angle between the LOS and the vertical
heading = -76  # angle between the azimuth and the east (anti-clock)
theta = num.empty(ntargets)  # vertical LOS from horizontal
theta.fill((90. - look) * d2r)
phi = num.empty(ntargets)  # horizontal LOS from E in anti-clokwise rotation
phi.fill((-90 - heading) * d2r)

satellite_target = gf.SatelliteTarget(
    north_shifts=norths2d,
    east_shifts=easts2d,
    tsnapshot=24. * 3600.,  # one day
    interpolation='nearest_neighbor',
    phi=phi,
    theta=theta,
    store_id=store_id)

# The computation is performed by calling process on the engine
result = engine.process(rect_source, [satellite_target])


def plot_static_los_result(result, target=0):
    '''Helper function for plotting the displacement'''

    import matplotlib.pyplot as plt
    import matplotlib.ticker as tick

    # get target coordinates and displacements from results
    N = result.request.targets[target].coords5[:, 2]
    E = result.request.targets[target].coords5[:, 3]
    synth_disp = result.results_list[0][target].result

    # get the component names of displacements
    components = synth_disp.keys()
    fig, _ = plt.subplots(int(len(components)/2), int(len(components)/2))

    vranges = [(synth_disp[k].max(),
                synth_disp[k].min()) for k in components]

    for comp, ax, vrange in zip(components, fig.axes, vranges):

        lmax = num.abs([num.min(vrange), num.max(vrange)]).max()

        # plot displacements at targets as colored points
        cmap = ax.scatter(E, N, c=synth_disp[comp], s=10., marker='s',
                          edgecolor='face', cmap='seismic',
                          vmin=-1.5*lmax, vmax=1.5*lmax)

        ax.set_title(comp+' [m]')
        ax.set_aspect('equal')
        ax.set_xlim(-obs_size, obs_size)
        ax.set_ylim(-obs_size, obs_size)
        # We plot the modeled fault
        n, e = rect_source.outline(cs='xy').T
        ax.fill(e, n, color=(0.5, 0.5, 0.5), alpha=0.5)

        fig.colorbar(cmap, ax=ax, aspect=5)

        # reduce number of ticks
        yticker = tick.MaxNLocator(nbins=5)
        yax = ax.get_yaxis()
        xax = ax.get_xaxis()
        yax.set_major_locator(yticker)
        xax.set_major_locator(yticker)

    plt.show()


plot_static_los_result(result)
