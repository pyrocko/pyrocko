from pyrocko.plot.directivity import plot_directivity
from pyrocko.gf import LocalEngine, RectangularSource, ws

km = 1e3
store_id = 'crust2_ib'

engine = LocalEngine(store_superdirs=['.'], use_config=True)

try:
    engine.get_store(store_id)
except:
    ws.download_gf_store(site='kinherd', store_id=store_id)

rect_source = RectangularSource(
    depth=2.6*km,
    strike=240.,
    dip=76.6,
    rake=-.4,
    anchor='top',

    nucleation_x=-.57,
    nucleation_y=-.59,
    velocity=2070.,

    length=27*km,
    width=9.4*km,
    slip=1.4)


resp = plot_directivity(
    engine, rect_source, store_id,
    distance=300*km, dazi=5., component='R',
    plot_mt='full', show_annotations=True,
    quantity='displacement')
