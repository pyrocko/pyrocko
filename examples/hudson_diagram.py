import sys
from matplotlib import pyplot as plt
from pyrocko.plot import hudson, beachball, mpl_init, mpl_color
from pyrocko import moment_tensor as pmt

# a bunch of random MTs
moment_tensors = [pmt.random_mt() for _ in range(200)]

# setup plot layout
fontsize = 10.
markersize = fontsize
mpl_init(fontsize=fontsize)
width = 7.
figsize = (width, width / (4. / 3.))
fig = plt.figure(figsize=figsize)
axes = fig.add_subplot(1, 1, 1)
fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.97)

# draw focal sphere diagrams for the random MTs
for mt in moment_tensors:
    u, v = hudson.project(mt)
    try:
        beachball.plot_beachball_mpl(
            mt, axes,
            beachball_type='full',
            position=(u, v),
            size=markersize,
            color_t=mpl_color('skyblue3'),
            color_p=mpl_color('skyblue1'),
            alpha=1.0,  # < 1 for transparency
            zorder=1,
            linewidth=0.25)

    except beachball.BeachballError as e:
        print(str(e), file=sys.stderr)

# draw the axes and annotations of the hudson plot
hudson.draw_axes(axes)

fig.savefig('hudson_diagram.png', dpi=150)
# plt.show()
