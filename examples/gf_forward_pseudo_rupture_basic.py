import os.path as op
from pyrocko import gf

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
    lat=0.,
    lon=0.,
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

# Recalculate slip, so that the rupture's magnitude fits the given value
rupture.rescale_slip(magnitude=7.0, store=store)

# Create waveform target, where synthetic waveforms are calculated for
waveform_target = gf.Target(
    lat=0.,
    lon=0.,
    east_shift=10*km,
    north_shift=10.*km,
    interpolation='multilinear',
    store_id=store_id)

# Get synthetic waveforms and display them in snuffler
response = engine.process(rupture, waveform_target)
response.snuffle()
