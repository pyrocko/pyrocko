from pyrocko.gf import LocalEngine, SatelliteTarget, RectangularSource
import numpy as num

# distance in kilometer
km = 1e3

# Ignite the LocalEngine and point it to fomosto stores stored on a
# USB stick, for this example we use a static store with id 'static_store'
engine = LocalEngine(store_superdirs=['/media/usb/stores'])
store_id = 'static_store'

# We define an extended source, in this case a rectangular geometry
# Centroid UTM position is defined relatively to geographical lat, lon position
# Purely lef-lateral strike-slip fault with an N104W azimuth.
rect_source = RectangularSource(
    lat=0., lon=0.,
    north_shift=0., east_shift=0., depth=6.5*km,
    width=5*km, length=8*km,
    dip=90., rake=0., strike=104.,
    slip=1.)

# We will define 1000 randomly distributed targets.
ntargets = 1000

# We initialize the satellite target and set the line of sight vectors
# direction, example of the Envisat satellite
look = 23.     # angle between the LOS and the vertical
heading = -76  # angle between the azimuth and the east (anti-clock)
theta = num.empty(ntargets)  # vertical LOS from horizontal
theta.fill(num.deg2rad(90. - look))
phi = num.empty(ntargets)  # horizontal LOS from E in anti-clokwise rotation
phi.fill(num.deg2rad(-90-heading))

satellite_target = SatelliteTarget(
    north_shifts=(num.random.rand(ntargets)-.5) * 30. * km,
    east_shifts=(num.random.rand(ntargets)-.5) * 30. * km,
    tsnapshot=60,
    interpolation='nearest_neighbor',
    phi=phi,
    theta=theta,
    store_id=store_id)

# The computation is performed by calling process on the engine
result = engine.process(rect_source, [satellite_target])


def plot_static_los_result(result, target=0):
    '''Helper function for plotting the displacement'''

    import matplotlib.pyplot as plt

    N = result.request.targets[target].coords5[:, 2]
    E = result.request.targets[target].coords5[:, 3]
    result = result.results_list[0][target].result

    # get the component names
    components = result.keys()
    fig, _ = plt.subplots(int(len(components)/2), int(len(components)/2))

    vranges = [(result[k].max(),
                result[k].min()) for k in components]

    for dspl, ax, vrange in zip(components, fig.axes, vranges):

        lmax = num.abs([num.min(vrange), num.max(vrange)]).max()
        levels = num.linspace(-lmax, lmax, 50)

        # plot interpolated points in map view with tricontourf
        cmap = ax.tricontourf(E, N, result[dspl],
                              cmap=plt.get_cmap('seismic'), levels=levels)

        ax.set_title(dspl+' [m]')
        ax.set_aspect('equal')

        # We plot the modeled fault
        n, e = rect_source.outline(cs='xy').T
        ax.fill(e, n, color=(0.5, 0.5, 0.5), alpha=0.5)

        fig.colorbar(cmap, ax=ax, aspect=5)

    plt.show()


plot_static_los_result(result)
