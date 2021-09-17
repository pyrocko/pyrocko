import os.path as op

from pyrocko import gf

km = 1e3

# The store we are going extract data from:
store_id = 'crust2_mf'

# First, download a Greens Functions store. If you already have one that you
# would like to use, you can skip this step and point the *store_superdirs* in
# the next step to that directory.
if not op.exists(store_id):
    gf.ws.download_gf_store(site='kinherd', store_id=store_id)

# We need a pyrocko.gf.Engine object which provides us with the traces
# extracted from the store. In this case we are going to use a local
# engine since we are going to query a local store.
engine = gf.LocalEngine(store_superdirs=['.'], use_config=True)

# The dynamic parameter used for discretization of the PseudoDynamicRupture are
# extracted from the stores config file.
store = engine.get_store(store_id)

# Let's define the source now with its extension, orientation etc.
dyn_rupture = gf.PseudoDynamicRupture(
    # At lat 0. and lon 0. (default)
    north_shift=2.*km,
    east_shift=2.*km,
    depth=3.*km,
    strike=43.,
    dip=89.,
    rake=88.,

    length=26.*km,
    width=12.*km,

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
dyn_rupture.rescale_slip(magnitude=7.0, store=store)

# Model waveforms for a single station target
waveform_target = gf.Target(
    lat=0.,
    lon=0.,
    east_shift=10.*km,
    north_shift=10.*km,
    store_id=store_id)

result = engine.process(dyn_rupture, waveform_target)
result.snuffle()
