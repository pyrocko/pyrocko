import random
import logging
import sys
from matplotlib import pyplot as plt
from pyrocko import beachball, moment_tensor as pmt
from pyrocko import util

logger = logging.getLogger(sys.argv[0])

util.setup_logging()

fig = plt.figure(figsize=(10., 4.))
fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)
axes = fig.add_subplot(1, 1, 1)

for i in xrange(200):

    # create random moment tensor
    mt = pmt.MomentTensor.random_mt()

    try:
        # create beachball from moment tensor
        beachball.plot_beachball_mpl(
            mt, axes,
            # type of beachball: deviatoric, full or double couple (dc)
            beachball_type='full',
            size=random.random()*120.,
            position=(random.random()*10., random.random()*10.),
            alpha=random.random(),
            linewidth=1.0)

    except beachball.BeachballError, e:
        logger.error('%s for MT:\n%s' % (e, mt))

axes.set_xlim(0., 10.)
axes.set_ylim(0., 10.)
axes.set_axis_off()
fig.savefig('beachball-example01.pdf')

plt.show()
