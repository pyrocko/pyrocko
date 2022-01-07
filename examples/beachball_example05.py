import random

from pyrocko.plot import beachball
import pyrocko.moment_tensor as mtm
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(4., 4.))
fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)
axes = fig.add_subplot(1, 1, 1)

# Number of available solutions
n_balls = 1000

# Best solution
strike = 135.
dip = 65.
rake = 15.

best_mt = mtm.MomentTensor.from_values((strike, dip, rake))

mts = []
for i in range(n_balls):
    # randomly change the strike by +- 15 deg
    strike_dev = random.random() * 30.0 - 15.0
    mts.append(mtm.MomentTensor.from_values(
        (strike + strike_dev, dip, rake)))

plot_kwargs = {
    'beachball_type': 'full',
    'size': 8,
    'position': (5, 5),
    'color_t': 'black',
    'edgecolor': 'black'
    }

beachball.plot_fuzzy_beachball_mpl_pixmap(mts, axes, best_mt, **plot_kwargs)

# Decorate the axes
axes.set_xlim(0., 10.)
axes.set_ylim(0., 10.)
axes.set_axis_off()

plt.show()
