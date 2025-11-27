import os.path
import numpy as num
from pyrocko import gf
from pyrocko.guts import List


class CombiSource(gf.Source):
    '''Composite source model.'''

    discretized_source_class = gf.DiscretizedMTSource

    subsources = List.T(gf.Source.T())

    def __init__(self, subsources=[], **kwargs):

        if subsources:

            lats = num.array(
                [subsource.lat for subsource in subsources], dtype=float)
            lons = num.array(
                [subsource.lon for subsource in subsources], dtype=float)

            assert num.all(lats == lats[0]) and num.all(lons == lons[0])
            lat, lon = lats[0], lons[0]

            # if not same use:
            # lat, lon = center_latlon(subsources)

            depth = float(num.mean([p.depth for p in subsources]))
            t = float(num.mean([p.time for p in subsources]))
            kwargs.update(time=t, lat=float(lat), lon=float(lon), depth=depth)

        gf.Source.__init__(self, subsources=subsources, **kwargs)

    def get_factor(self):
        return 1.0

    def discretize_basesource(self, store, target=None):
        dsources = []
        t0 = self.subsources[0].time
        for sf in self.subsources:
            assert t0 == sf.time
            ds = sf.discretize_basesource(store, target)
            ds.m6s *= sf.get_factor()
            dsources.append(ds)

        return gf.DiscretizedMTSource.combine(dsources)


# Download a Greens Functions store, programmatically.
store_id = 'gf_abruzzo_nearfield_vmod_Ameri'
if not os.path.exists(store_id):
    gf.ws.download_gf_store(site='pyrocko', store_id=store_id)

km = 1e3   # distance in kilometer

# We define a grid for the targets.
left, right, bottom, top = -10*km, 10*km, -10*km, 10*km
ntargets = 1000

# Ignite the LocalEngine and point it to fomosto stores stored on a
# USB stick, for this example we use a static store with id 'static_store'
engine = gf.LocalEngine(store_superdirs=['.'])
store_id = 'gf_abruzzo_nearfield_vmod_Ameri'

# We define two finite sources
# The first one is a purely vertical strike-slip fault
strikeslip = gf.RectangularSource(
    north_shift=0., east_shift=0.,
    depth=6*km, width=4*km, length=10*km,
    dip=90., rake=0., strike=90.,
    slip=1.)

# The second one is a ramp connecting to the root of the strike-slip fault
# ramp north shift (n) and width (w) depend on its dip angle and on
# the strike slip fault width
n, w = 2/num.tan(num.deg2rad(45.)), 2.*(2./(num.sin(num.deg2rad(45.))))
thrust = gf.RectangularSource(
    north_shift=n*km, east_shift=0.,
    depth=6*km, width=w*km, length=10*km,
    dip=45., rake=90., strike=90.,
    slip=0.5)

# We initialize the satellite target and set the line of site vectors
# Case example of the Sentinel-1 satellite:
# Heading: -166 (anti clockwise rotation from east)
# Average Look Angle: 36 (from vertical)
heading = -76
look = 36.
phi = num.empty(ntargets)    # Horizontal LOS from E in anti-clockwise rotation
theta = num.empty(ntargets)  # Vertical LOS from horizontal
phi.fill(num.deg2rad(-90. - heading))
theta.fill(num.deg2rad(90. - look))

satellite_target = gf.SatelliteTarget(
    north_shifts=num.random.uniform(bottom, top, ntargets),
    east_shifts=num.random.uniform(left, right, ntargets),
    tsnapshot=24.*3600.,
    interpolation='nearest_neighbor',
    phi=phi,
    theta=theta,
    store_id=store_id)

# We combine the two sources here
patches = [strikeslip, thrust]
sources = CombiSource(subsources=patches)

# The computation is performed by calling process on the engine
result = engine.process(sources, [satellite_target])


def plot_static_los_profile(result, strike, l, w, x0, y0):  # noqa
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    fig, _ = plt.subplots(1, 2, figsize=(8, 4))

    # strike,l,w,x0,y0: strike, length, width, x, and y position
    # of the profile
    strike = num.deg2rad(strike)
    # We define the parallel and perpendicular vectors to the profile
    s = [num.sin(strike), num.cos(strike)]
    n = [num.cos(strike), -num.sin(strike)]

    # We define the boundaries of the profile
    ypmax, ypmin = l/2, -l/2
    xpmax, xpmin = w/2, -w/2

    # We define the corners of the profile
    xpro, ypro = num.zeros((7)), num.zeros((7))
    xpro[:] = x0-w/2*s[0]-l/2*n[0], x0+w/2*s[0]-l/2*n[0], \
        x0+w/2*s[0]+l/2*n[0], x0-w/2*s[0]+l/2*n[0], x0-w/2*s[0]-l/2*n[0], \
        x0-l/2*n[0], x0+l/2*n[0]

    ypro[:] = y0-w/2*s[1]-l/2*n[1], y0+w/2*s[1]-l/2*n[1], \
        y0+w/2*s[1]+l/2*n[1], y0-w/2*s[1]+l/2*n[1], y0-w/2*s[1]-l/2*n[1], \
        y0-l/2*n[1], y0+l/2*n[1]

    # We get the forward model from the engine
    N = result.request.targets[0].coords5[:, 2]
    E = result.request.targets[0].coords5[:, 3]
    result = result.results_list[0][0].result

    # We first plot the surface displacements in map view
    ax = fig.axes[0]
    los = result['displacement.los']
    levels = num.linspace(los.min(), los.max(), 50)

    cmap = ax.tricontourf(E, N, los, cmap=plt.get_cmap('seismic'),
                          levels=levels)

    for sourcess in patches:
        fn, fe = sourcess.outline(cs='xy').T
        ax.fill(fe, fn, color=(0.5, 0.5, 0.5), alpha=0.5)
        ax.plot(fe[:2], fn[:2], linewidth=2., color='black', alpha=0.5)

    # We plot the limits of the profile in map view
    ax.plot(xpro[:], ypro[:], color='black', lw=1.)
    # plot colorbar
    fig.colorbar(cmap, ax=ax, orientation='vertical', aspect=5)
    ax.set_title('Map view')
    ax.set_aspect('equal')

    # We plot displacements in profile
    ax = fig.axes[1]
    # We compute the perpendicular and parallel components in the profile basis
    yp = (E-x0)*n[0]+(N-y0)*n[1]
    xp = (E-x0)*s[0]+(N-y0)*s[1]
    los = result['displacement.los']

    # We select data encompassing the profile
    index = num.nonzero(
        (xp > xpmax) | (xp < xpmin) | (yp > ypmax) | (yp < ypmin))

    ypp, losp = num.delete(yp, index), \
        num.delete(los, index)

    # We associate the same color scale to the scatter plot
    norm = mcolors.Normalize(vmin=los.min(), vmax=los.max())
    m = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('seismic'))
    facelos = m.to_rgba(losp)
    ax.scatter(
        ypp, losp,
        s=0.3, marker='o', color=facelos, label='LOS displacements')

    ax.legend(loc='best')
    ax.set_title('Profile')

    plt.show()


plot_static_los_profile(result, 110., 18*km, 5*km, 0., 0.)
