import os
import shutil
from pyrocko.plot.directivity import plot_directivity
from pyrocko.gf import LocalEngine, DCSource, Store
from pyrocko.fomosto import ahfullgreen

km = 1e3


def make_homogeneous_gf_store(
        path, store_id, source_depth, receiver_depth, distance):

    if os.path.exists(path):
        shutil.rmtree(path)

    ahfullgreen.init(path, None, config_params=dict(
        id=store_id,
        sample_rate=20.,
        receiver_depth=receiver_depth,
        source_depth_min=source_depth,
        source_depth_max=source_depth,
        distance_min=distance,
        distance_max=distance))

    store = Store(path)
    store.make_travel_time_tables()
    ahfullgreen.build(path)


store_id = 'gf_homogeneous_radpat'
store_path = os.path.join('.', store_id)
distance = 10*km
receiver_depth = 0.0

source = DCSource(
    depth=0.,
    strike=0.,
    dip=90.,
    rake=0.)

make_homogeneous_gf_store(
    store_path, store_id, source.depth, receiver_depth, distance)

engine = LocalEngine(store_dirs=[store_path])

# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(7,5))
# axes = fig.add_subplot(111, polar=True)

resp = plot_directivity(
    engine, source, store_id,
    # axes=axes,
    distance=distance,
    dazi=5.,
    component='R',
    target_depth=receiver_depth,
    plot_mt='full',
    show_phases=True,
    fmin=None,
    fmax=1.0,
    phases={
        'P': '{stored:anyP}-50%',
        'S': '{stored:anyS}+50%'
    },
    interpolation='nearest_neighbor',
    quantity='velocity',
    envelope=False,
    hillshade=False)

# fig.savefig('radiation_pattern.png')
