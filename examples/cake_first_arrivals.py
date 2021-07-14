import numpy as num
from matplotlib import pyplot as plt

from pyrocko.plot import mpl_init, mpl_margins, mpl_papersize
from pyrocko import cake

fontsize = 9.
km = 1000.

mod = cake.load_model('ak135-f-continental.m')

phases = cake.PhaseDef.classic('Pg')
source_depth = 20.*km

distances = num.linspace(100.*km, 1000.*km, 100)

data = []
for distance in distances:
    rays = mod.arrivals(
        phases=phases, distances=[distance*cake.m2d], zstart=source_depth)
    for ray in rays[:1]:
        data.append((distance, ray.t))

phase_distances, phase_time = num.array(data, dtype=float).T

# Plot the arrival times
mpl_init(fontsize=fontsize)
fig = plt.figure(figsize=mpl_papersize('a5', 'landscape'))
labelpos = mpl_margins(fig, w=7., h=5., units=fontsize)
axes = fig.add_subplot(1, 1, 1)
labelpos(axes, 2., 1.5)

axes.set_xlabel('Distance [km]')
axes.set_ylabel('Time [s]')

axes.plot(phase_distances/km, phase_time, 'o', ms=3., color='black')

fig.savefig('cake_first_arrivals.pdf')
