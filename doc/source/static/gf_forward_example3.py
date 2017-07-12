from pyrocko import gf
import numpy as num

# distance in kilometer
km = 1e3
# many seconds make a day
day= 24.*3600.

# Ignite the LocalEngine and point it to your fomosto store, e.g. stored on a
# USB stick, which for example has the id 'Abruzzo_Ameri_static_nearfield'
# (download at http://kinherd.org:8080/gfws/static/stores/)
engine = gf.LocalEngine(store_superdirs=['/media/usb/stores'])
store_id = 'Abruzzo_Ameri_static_nearfield'

# We want to reproduce the USGS Solution of an event, e.g.
dep, strike, dip, leng, wid, rake, slip = 10.5, 90., 40., 10., 10., 90., .5

# We compute the magnitude of the event
potency = leng*km*wid*km*slip
rigidity = 31.5e9
m0 = potency*rigidity
mw = (2./3) * num.log10(m0) - 6.07

# We define an extended rectangular source
thrust = gf.RectangularSource(
    north_shift=0., east_shift=0.,
    depth=dep*km, width=wid*km, length=leng*km,
    dip=dip, rake=rake, strike=strike,
    slip=slip)

# We will define a grid of targets
# number in east and north directions, and total
ngrid = 90
# extension from origin in all directions
obs_size = 20.*km
ntargets = ngrid**2
# make regular line vector
norths = num.linspace(-obs_size, obs_size, ngrid)
easts  = num.linspace(-obs_size, obs_size, ngrid)
# make regular grid
norths2d = num.repeat(norths, len(easts))
easts2d = num.tile(easts, len(norths))


# We initialize the satellite target and set the line of site vectors
# Case example of the Sentinel-1 satellite:
# Heading: -166 (anti-clockwise rotation from east)
# Average Look Angle: 36 (from vertical)
heading = -76.
look = 36.
phi = num.empty(ntargets)    # Horizontal LOS from E in anti-clockwise rotation
theta = num.empty(ntargets)  # Vertical LOS from horizontal
phi.fill(num.deg2rad(-90-heading))
theta.fill(num.deg2rad(90.-look))

satellite_target = gf.SatelliteTarget(
    north_shifts=norths2d,
    east_shifts=easts2d,
    tsnapshot=1.*day,
    interpolation='nearest_neighbor',
    phi=phi,
    theta=theta,
    store_id=store_id)

# The computation is performed by calling 'process' on the engine
result = engine.process(thrust, [satellite_target])


def plot_static_los_result(result, target=0):
    '''Helper function for plotting the displacement'''
    import matplotlib.pyplot as plt

    # get synthetic displacements and target coordinates from engine's 'result'
    N = result.request.targets[target].coords5[:, 2]
    E = result.request.targets[target].coords5[:, 3]
    synth_disp = result.results_list[0][target].result
    
    # we get the fault projection to the surface for plotting 
    n, e = thrust.outline(cs='xy').T
    
    fig, _ = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(
        "fault: dep={:0.2f}, l={}, w={:0.2f},str={},"
        "rake={}, dip={}, slip={}, Mw={:0.3f}\n"
        "satellite: heading={}, look angle={}"
        .format(dep, leng, wid, strike, rake, dip, slip, heading, look, mw),
        fontsize=14,
        fontweight='bold')

    # Plot unwrapped LOS displacements
    ax = fig.axes[0]
    # We shift the relative LOS displacements
    los = synth_disp['displacement.los']
    losrange = [(los.max(), los.min())]
    losmax = num.abs([num.min(losrange), num.max(losrange)]).max()

    cmap = ax.scatter(
        E, N, c=los,
        s = 10., marker = 's', 
        edgecolor='face',
        cmap=plt.get_cmap('seismic'), 
        vmin= -1.*losmax, vmax=1.*losmax)

    ax.set_title('line-of-sight displacement [m]')
    ax.set_aspect('equal')
    ax.set_xlim(-obs_size, obs_size)
    ax.set_ylim(-obs_size, obs_size)
    # plot fault outline
    ax.fill(e, n, color=(0.5, 0.5, 0.5), alpha=0.5)
    # We underline the tip of the thrust
    ax.plot(e[:2], n[:2], linewidth=2., color='black', alpha=0.5)

    fig.colorbar(cmap, ax=ax, orientation='vertical', aspect=5, shrink=0.5)

    # We plot wrapped phase
    ax = fig.axes[1]
    # We simulate a C-band interferogram for this source
    c_lambda = 0.056
    insar_phase = -num.mod(los, c_lambda/2.)/(c_lambda/2.)*2.*num.pi - num.pi

    cmap = ax.scatter(
        E, N, c= insar_phase,
        s = 10., marker = 's', 
        edgecolor='face',
        cmap=plt.get_cmap('gist_rainbow'))

    ax.set_xlim(-obs_size, obs_size)
    ax.set_ylim(-obs_size, obs_size)
    ax.set_title('simulated interferogram')
    ax.set_aspect('equal')
    
    # plot fault outline
    ax.fill(e, n, color=(0.5, 0.5, 0.5), alpha=0.5)
    # We outline the top edge of the fault with a thick line
    ax.plot(e[:2], n[:2], linewidth=2., color='black', alpha=0.5)

    fig.colorbar(cmap, orientation='vertical', shrink=0.5, aspect=5)
    plt.show()

plot_static_los_result(result)
