import time
import numpy as np
from pyrocko import gf, plot, util

km = 1000.

engine = gf.get_engine()

store: gf.Store = engine.get_store('global_2s_v2')

ndistances = 1000

fontsize = 9.
plt = plot.mpl_init(fontsize=fontsize)
fig = plt.figure(figsize=plot.mpl_papersize('a5', 'landscape'))
labelpos = plot.mpl_margins(fig, w=7., h=5., units=fontsize)
axes = fig.add_subplot(1, 1, 1)
labelpos(axes, 2., 1.5)

for source_depth in reversed(util.arange2(100*km, 700*km, 100*km)):
    distances = np.linspace(10*km, 20000*km, ndistances)  # surface distances

    coordinates = np.zeros((ndistances, 2))
    coordinates[:, 0] = source_depth
    coordinates[:, 1] = distances

    t0 = time.time()
    traveltimes = store.get_stored_phase('P').interpolate_many(coordinates)
    t1 = time.time()
    print(t1 - t0)
    axes.plot(
        distances / km,
        traveltimes,
        label='Source depth: %g km' % (source_depth / km))

fig.suptitle('P phase')
axes.set_ylabel('Traveltime [s]')
axes.set_xlabel('Distance [km]')
axes.legend()
fig.savefig('traveltimes.png')
