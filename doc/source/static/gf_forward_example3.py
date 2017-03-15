from pyrocko.gf import LocalEngine, SatelliteTarget, RectangularSource
import numpy as num

# distance in kilometer
km = 1e3

# Ignite the LocalEngine and point it to fomosto stores stored on a
# USB stick, for this example we use a static store with id 'static_store'
store_id = 'static_store'
engine = LocalEngine(store_superdirs=['/media/usb/gf_stores'])

# We want to reproduce the USGS Solution of the event
d, strike, dip, l, W, rake, slip = 10.5, 90., 40., 10., 10., 90., 5.

# We compute the magnitude of the event
potency = l*km*W*km*slip
m0 = potency*31.5e9
mw = (2./3) * num.log10(m0) - 6.07

# We define an extended source, in this case a rectangular geometry
# horizontal distance
# The centorid north position depends on its dip angle and its width.
n = num.cos(num.deg2rad(dip))*W/2

thrust = RectangularSource(
    north_shift=n*km, east_shift=0.,
    depth=d*km, width=W*km, length=l*km,
    dip=dip, rake=rake, strike=strike,
    slip=slip)

# We define a grid for the targets.
left, right, bottom, top = -15*km, 15*km, -15*km, 15*km
ntargets = 10000

# We initialize the satellite target and set the line of site vectors
# Case example of the Sentinel-1 satellite:
# Heading: -166 (anti clokwise rotation from east)
# Average Look Angle: 36 (from vertical)
heading = -76.
look = 36.
phi = num.empty(ntargets)    # Horizontal LOS from E in anti-clokwise rotation
theta = num.empty(ntargets)  # Vertical LOS from horizontal
phi.fill(num.deg2rad(-90-heading))
theta.fill(num.deg2rad(90.-look))

satellite_target = SatelliteTarget(
    north_shifts=num.random.uniform(bottom, top, ntargets),
    east_shifts=num.random.uniform(left, right, ntargets),
    tsnapshot=60,
    interpolation='nearest_neighbor',
    phi=phi,
    theta=theta,
    store_id=store_id)

# The computation is performed by calling process on the engine
result = engine.process(thrust, [satellite_target])


def plot_static_los_result(result, target=0):
    '''Helper function for plotting the displacement'''
    import matplotlib.pyplot as plt

    # get forward model from engine
    N = result.request.targets[target].coords5[:, 2]
    E = result.request.targets[target].coords5[:, 3]
    result = result.results_list[0][target].result

    fig, _ = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(
        "thrust: depth={:0.2f}, l={}, w={:0.2f},strike={}, "
        "rake={}, dip={}, slip={}\n"
        "heading={}, look angle={}, Mw={:0.3f}"
        .format(d, l, W, strike, rake, dip, slip, heading, look, mw),
        fontsize=14,
        fontweight='bold')

    # Plot unwrapped LOS displacements
    ax = fig.axes[0]
    # We shift the relative LOS displacements
    los = result['displacement.los'] - result['displacement.los'].min()
    losrange = [(los.max(), los.min())]
    losmax = num.abs([num.min(losrange), num.max(losrange)]).max()
    levels = num.linspace(0, losmax, 50)

    cmap = ax.tricontourf(
        E, N, los,
        cmap=plt.get_cmap('seismic'),
        levels=levels)

    ax.set_title('los')
    ax.set_aspect('equal')

    # We plot the fault projection to the surface
    n, e = thrust.outline(cs='xy').T
    ax.fill(e, n, color=(0.5, 0.5, 0.5), alpha=0.5)
    # We underline the tip of the thrust
    ax.plot(e[:2], n[:2], linewidth=2., color='black', alpha=0.5)

    fig.colorbar(cmap, ax=ax, orientation='vertical', aspect=5, shrink=0.5)

    # We plot wrapped phase
    ax = fig.axes[1]
    # We wrap the phase between 0 and 0.028 mm
    wavelenght = 0.028
    wrapped_los = num.mod(los, wavelenght)
    levels = num.linspace(0, wavelenght, 50)

    # ax.tricontour(E, N, wrapped_los,
    #   map='gist_rainbow', levels=levels, colors='k')
    cmap = ax.tricontourf(
        E, N, wrapped_los,
        cmap=plt.get_cmap('gist_rainbow'),
        levels=levels,
        interpolation='bicubic')

    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)

    ax.set_title('wrapped los')
    ax.set_aspect('equal')

    # We plot the fault projection to the surface
    n, e = thrust.outline(cs='xy').T
    ax.fill(e, n, color=(0.5, 0.5, 0.5), alpha=0.5)
    # We underline the tiip of the fault
    ax.plot(e[:2], n[:2], linewidth=2., color='black', alpha=0.5)

    fig.colorbar(cmap, orientation='vertical', shrink=0.5, aspect=5)
    plt.show()

plot_static_los_result(result)
