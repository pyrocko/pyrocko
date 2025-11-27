import os.path
from kite.scene import Scene, FrameConfig
from pyrocko import gf
import numpy as num

km = 1e3
d2r = num.pi/180.

# Download a Greens Functions store
store_id = 'gf_abruzzo_nearfield_vmod_Ameri'
if not os.path.exists(store_id):
    gf.ws.download_gf_store(site='pyrocko', store_id=store_id)

# Setup the modelling LocalEngine
# *store_superdirs* is a list of directories where to look for GF Stores.
engine = gf.LocalEngine(store_superdirs=['.'])

rect_source = gf.RectangularSource(
    # Geographical position [deg]
    lat=0., lon=0.,
    # Relative cartesian offsets [m]
    north_shift=10*km, east_shift=10*km,
    depth=6.5*km,
    # Dimensions of the fault [m]
    width=5*km, length=8*km,
    strike=104., dip=90., rake=0.,
    # Slip in [m]
    slip=1., anchor='top')

# Define the scene's frame
frame = FrameConfig(
    # Lower left geographical reference [deg]
    llLat=0., llLon=0.,
    # Pixel spacing [m] or [degrees]
    spacing='meter', dE=250, dN=250)

# Resolution of the scene
npx_east = 800
npx_north = 800

# 2D arrays for displacement and look vector
displacement = num.empty((npx_east, npx_north))

# Look vectors
# Theta is elevation angle from horizon
theta = num.full_like(displacement, 48.*d2r)
# Phi is azimuth towards the satellite, counter-clockwise from East
phi = num.full_like(displacement, 23.*d2r)

scene = Scene(
    displacement=displacement,
    phi=phi, theta=theta,
    frame=frame)

# Or just load an existing scene!
# scene = Scene.load('my_scene_asc.npy')

satellite_target = gf.KiteSceneTarget(
    scene,
    store_id=store_id)

# Forward model!
result = engine.process(
    rect_source, satellite_target,
    # Use all available cores
    nthreads=0)

kite_scenes = result.kite_scenes()

# Show the synthetic data in spool
# kite_scenes[0].spool()
