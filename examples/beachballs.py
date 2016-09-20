
from matplotlib import pyplot as plt
from pyrocko import beachball, moment_tensor as pmt

fig = plt.figure(figsize=(10., 10.))
fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)
axes = fig.add_subplot(1, 1, 1)
axes.set_xlim(0., 4.)
axes.set_ylim(0., 3.)
axes.set_axis_off()

for (strike, dip, rake), position, color in [
        ((0., 30., 120.), (1, 1.), 'red'),
        ((0., 30., 150.), (2., 1.), 'black'),
        ((0., 30., 180.), (3., 1), (0.5, 0.3, 0.7)),
        ((0., 30., -180.), (1., 2.), (0.1, 0.6, 0.1)),
        ((0., 30., -150.), (2., 2.), (0.2, 0.7, 1.0)),
        ((0., 30., -120.), (3., 2.), (0.2, 0.3, 0.1))]:

    beachball.plot_beachball_mpl(
        pmt.as_mt((strike, dip, rake)),
        axes,
        beachball_type='full',
        size=100.,
        position=position,
        color_t=color,
        linewidth=1.0)

fig.savefig('beachballs.pdf')

plt.show()
