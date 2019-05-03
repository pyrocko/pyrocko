import os.path
from pyrocko import gf
import numpy as num

# Download a Greens Functions store, programmatically.
store_id = 'gf_abruzzo_nearfield_vmod_Ameri'
if not os.path.exists(store_id):
    gf.ws.download_gf_store(site='kinherd', store_id=store_id)

# Setup the LocalEngine and point it to the fomosto store you just downloaded.
# *store_superdirs* is a list of directories where to look for GF Stores.
engine = gf.LocalEngine(store_superdirs=['.'])

# We define an extended source, in this case a rectangular geometry
# Centroid UTM position is defined relatively to geographical lat, lon position
# Purely lef-lateral strike-slip fault with an N104W azimuth.

km = 1e3  # for convenience

rect_source = gf.RectangularSource(
    lat=0., lon=0.,
    north_shift=0., east_shift=0., depth=6.5*km,
    width=5*km, length=8*km,
    dip=90., rake=0., strike=104.,
    slip=1.)

# We will define a grid of targets
# number in east and north directions, and total
ngrid = 80

# extension from origin in all directions
obs_size = 20.*km
ntargets = ngrid**2

# make regular line vector
norths = num.linspace(-obs_size, obs_size, ngrid)
easts = num.linspace(-obs_size, obs_size, ngrid)

# make regular grid
norths2d = num.repeat(norths, len(easts))
easts2d = num.tile(easts, len(norths))

# We initialize the satellite target and set the line of sight vectors
# direction, example of the Envisat satellite
look = 23.     # angle between the LOS and the vertical
heading = -76  # angle between the azimuth and the east (anti-clock)
theta = num.empty(ntargets)  # vertical LOS from horizontal
theta.fill(num.deg2rad(90. - look))
phi = num.empty(ntargets)  # horizontal LOS from E in anti-clokwise rotation
phi.fill(num.deg2rad(-90-heading))

satellite_target = gf.SatelliteTarget(
    north_shifts=norths2d,
    east_shifts=easts2d,
    tsnapshot=24. * 3600.,  # one day
    interpolation='nearest_neighbor',
    phi=phi,
    theta=theta,
    nrows=len(norths),
    ncols=len(easts),
    store_id=store_id)

# The computation is performed by calling process on the engine
result = engine.process(rect_source, [satellite_target])

# We now return a list of kite_scenes from the result.
kite_scenes = result.kite_scenes()
# using Kite we can now display the data, subsample and extract the covariance
kite_scenes[0].spool()
