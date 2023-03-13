
import matplotlib.pyplot as plt
from pyrocko.plot import beachball, mpl_color
from pyrocko import moment_tensor as pmt


nrows = ncols = 3

fig = plt.figure(figsize=(6, 6))
axes = fig.add_subplot(1, 1, 1, aspect=1.)
fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)
axes.axison = False
axes.set_xlim(-0.05 - ncols, ncols + 0.05)
axes.set_ylim(-0.05 - nrows, nrows + 0.05)

mt = pmt.as_mt((5., 90, 5.))

for view, irow, icol in [
        ('top', 1, 1),
        ((-90-45, 90.), 0, 0),
        ('north', 0, 1),
        ((-45, 90.), 0, 2),
        ('east', 1, 2),
        ((45., 90.), 2, 2),
        ('south', 2, 1),
        ((90.+45., 90.), 2, 0),
        ('west', 1, 0)]:

    beachball.plot_beachball_mpl(
        mt, axes,
        position=(icol*2-ncols+1, -irow*2+nrows-1),
        size_units='data',
        linewidth=1.0,
        color_t=mpl_color('skyblue2'),
        view=view)

    axes.annotate(
        view,
        xy=(icol*2-ncols+1, -irow*2+nrows-1.75),
        xycoords='data',
        xytext=(0, 0),
        textcoords='offset points',
        verticalalignment='center',
        horizontalalignment='center',
        rotation=0.)

fig.savefig('beachball_example06.png')
plt.show()
