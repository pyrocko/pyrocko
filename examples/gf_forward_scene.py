#!/usr/bin/env python3
from pyrocko import gf
from kite import Scene

km = 1e3
engine = gf.LocalEngine(use_config=True)

scene = Scene.load('/home/marius/Development/testing/kite/data/ula-model_data-processed/ula_april_asc.npz')  # noqa

src_lat = 37.08194 + .045
src_lon = 28.45194 + .2

source = gf.RectangularSource(
    lat=src_lat,
    lon=src_lon,
    depth=2*km,
    length=4*km, width=2*km,
    strike=45., dip=60.,
    slip=.5, rake=0.,
    anchor='top')

target = gf.KiteSceneTarget(scene, store_id='ak135_static')

result = engine.process(source, target, nthreads=0)

mod_scene = result.kite_scenes()[0]
mod_scene.spool()
