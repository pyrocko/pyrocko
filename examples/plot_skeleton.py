from matplotlib import pyplot as plt

from pyrocko.plot import mpl_init, mpl_margins, mpl_papersize
# from pyrocko.plot import mpl_labelspace

fontsize = 9.   # in points

# set some Pyrocko style defaults
mpl_init(fontsize=fontsize)

fig = plt.figure(figsize=mpl_papersize('a4', 'landscape'))

# let margins be proportional to selected font size, e.g. top and bottom
# margin are set to be 5*fontsize = 45 [points]
labelpos = mpl_margins(fig, w=7., h=5., units=fontsize)

axes = fig.add_subplot(1, 1, 1)

# positioning of axis labels
# mpl_labelspace(axes)    # either: relative to axis tick labels
labelpos(axes, 2., 1.5)   # or: relative to left/bottom paper edge

axes.plot([0, 1], [0, 9])

axes.set_xlabel('Time [s]')
axes.set_ylabel('Amplitude [m]')

fig.savefig('plot_skeleton.pdf')

plt.show()
