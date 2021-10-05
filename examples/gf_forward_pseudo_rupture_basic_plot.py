import os.path as op
from pyrocko import gf
from pyrocko.plot.dynamic_rupture import RuptureView

km = 1e3

# The store we are going extract data from:
store_id = 'iceland_reg_v2'

# First, download a Green's functions store. If you already have one that you
# would like to use, you can skip this step and point the *store_superdirs* to
# the containing directory.
if not op.exists(store_id):
    gf.ws.download_gf_store(site='kinherd', store_id=store_id)

# We need a pyrocko.gf.Engine object which provides us with the traces
# extracted from the store. In this case we are going to use a local
# engine since we are going to query a local store.
engine = gf.LocalEngine(store_superdirs=['.'])

# The dynamic parameters used for discretization of the PseudoDynamicRupture
# are extracted from the store's config file.
store = engine.get_store(store_id)

# Let's define the source now with its extension, orientation etc.
rupture = gf.PseudoDynamicRupture(
    # At lat 0. and lon 0. (default)
    north_shift=2.*km,
    east_shift=2.*km,
    depth=3.*km,
    strike=43.,
    dip=89.,
    rake=88.,

    length=15*km,
    width=5*km,

    nx=10,
    ny=5,

    # Relative nucleation between -1. and 1.
    nucleation_x=-.6,
    nucleation_y=.3,
    slip=1.,
    anchor='top',

    # Threads used for modelling
    nthreads=1,

    # Force pure shear rupture
    pure_shear=True)

# Recalculate slip, that rupture magnitude fits given magnitude
rupture.rescale_slip(magnitude=7.0, store=store)

plot = RuptureView(rupture, figsize=(8, 4))
plot.draw_patch_parameter('traction')
plot.draw_time_contour(store)
plot.draw_nucleation_point()
plot.save('dynamic_basic_tractions.png')
# Alternatively plot on screen
# plot.show_plot()
