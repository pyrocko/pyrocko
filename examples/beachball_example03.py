from matplotlib import pyplot as plt
from pyrocko import moment_tensor as pmt
from pyrocko import plot
from pyrocko.plot import beachball


fig = plt.figure(figsize=(4., 2.))
fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)
axes = fig.add_subplot(1, 1, 1)
axes.set_xlim(0., 4.)
axes.set_ylim(0., 2.)
axes.set_axis_off()

for i, beachball_type in enumerate(['full', 'deviatoric', 'dc']):
    beachball.plot_beachball_mpl(
            pmt.as_mt((124654616., 370943136., -6965434.0,
                       553316224., -307467264., 84703760.0)),
            axes,
            beachball_type=beachball_type,
            size=60.,
            position=(i+1, 1),
            color_t=plot.mpl_color('scarletred2'),
            linewidth=1.0)

fig.savefig('beachball-example03.pdf')
plt.show()
