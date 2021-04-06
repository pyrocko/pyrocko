import numpy as num
import matplotlib.pyplot as plt
from pyrocko import spit, cake
from pyrocko.gf import meta


# Define a list of phases.
phase_defs = [meta.TPDef(id='stored:depth_p', definition='p'),
              meta.TPDef(id='stored:P', definition='P')]

# Load a velocity model. In this example use the default AK135.
mod = cake.load_model()

# Time and space tolerance thresholds defining the accuracy of the
# :py:class:`pyrocko.spit.SPTree`.
t_tolerance = 0.1                           # in seconds
x_tolerance = num.array((500., 500.))       # in meters

# Boundaries of the grid.
xmin = 0.
xmax = 20000
zmin = 0.
zmax = 11000
x_bounds = num.array(((xmin, xmax), (zmin, zmax)))

# In this example the receiver is located at the surface.
receiver_depth = 0.

interpolated_tts = {}

for phase_def in phase_defs:
    v_horizontal = phase_def.horizontal_velocities

    def evaluate(args):
        '''Calculate arrival using source and receiver location
        defined by *args*. To be evaluated by the SPTree instance.'''
        source_depth, x = args

        t = []

        # Calculate arrivals
        rays = mod.arrivals(
            phases=phase_def.phases,
            distances=[x*cake.m2d],
            zstart=source_depth,
            zstop=receiver_depth)

        for ray in rays:
            t.append(ray.t)

        for v in v_horizontal:
            t.append(x/(v*1000.))
        if t:
            return min(t)
        else:
            return None

    # Creat a :py:class:`pyrocko.spit.SPTree` interpolator.
    sptree = spit.SPTree(
        f=evaluate,
        ftol=t_tolerance,
        xbounds=x_bounds,
        xtols=x_tolerance)

    # Store the result in a dictionary which is later used to retrieve an
    # SPTree (value) for each phase_id (key).
    interpolated_tts[phase_def.id] = sptree

    # Dump the sptree for later reuse:
    sptree.dump(filename='sptree_%s.yaml' % phase_def.id.split(':')[1])

# Define a :py:class:`pyrocko.gf.meta.Timing` instance.
timing = meta.Timing('first(depth_p|P)')


# If only one interpolated onset is need at a time you can retrieve
# that value as follows:
# First argument has to be a function which takes a requested *phase_id*
# and returns the associated :py:class:`pyrocko.spit.SPTree` instance.
# Second argument is a tuple of distance and source depth.
z_want = 5000.
x_want = 2000.
one_onset = timing.evaluate(lambda x: interpolated_tts[x],
                            (z_want, x_want))
print('a single arrival: ', one_onset)


# But if you have many locations for which you would like to calculate the
# onset time the following is the preferred way as it is much faster
# on large coordinate arrays.
# x_want is now an array of 1000 distances
x_want = num.linspace(0, xmax, 1000)

# Coords is set up of distances-depth-pairs
coords = num.array((x_want, num.tile(z_want, x_want.shape))).T

# *interpolate_many* then interpolates onset times for each of these
# pairs.
tts = interpolated_tts["stored:depth_p"].interpolate_many(coords)

# Plot distance vs. onset time
plt.plot(x_want, tts, '.')
plt.xlabel('Distance [m]')
plt.ylabel('Travel Time [s]')
plt.show()
