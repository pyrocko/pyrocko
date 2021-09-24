import os

from pyrocko.gf import LocalEngine, Target, DCSource, ws
from pyrocko import trace
from pyrocko.gui.marker import PhaseMarker

# The store we are going extract data from:
store_id = 'iceland_reg_v2'

# First, download a Greens Functions store. If you already have one that you
# would like to use, you can skip this step and point the *store_superdirs* in
# the next step to that directory.

if not os.path.exists(store_id):
    ws.download_gf_store(site='kinherd', store_id=store_id)

# We need a pyrocko.gf.Engine object which provides us with the traces
# extracted from the store. In this case we are going to use a local
# engine since we are going to query a local store.
engine = LocalEngine(store_superdirs=['.'])

# Define a list of pyrocko.gf.Target objects, representing the recording
# devices. In this case one station with a three component sensor will
# serve fine for demonstation.
channel_codes = 'ENZ'
targets = [
    Target(
        lat=10.,
        lon=10.,
        store_id=store_id,
        codes=('', 'STA', '', channel_code))
    for channel_code in channel_codes]

# Let's use a double couple source representation.
source_dc = DCSource(
    lat=11.,
    lon=11.,
    depth=10000.,
    strike=20.,
    dip=40.,
    rake=60.,
    magnitude=4.)

# Processing that data will return a pyrocko.gf.Reponse object.
response = engine.process(source_dc, targets)

# This will return a list of the requested traces:
synthetic_traces = response.pyrocko_traces()

# In addition to that it is also possible to extract interpolated travel times
# of phases which have been defined in the store's config file.
store = engine.get_store(store_id)

markers = []
for t in targets:
    dist = t.distance_to(source_dc)
    depth = source_dc.depth
    arrival_time = store.t('begin', (depth, dist))
    m = PhaseMarker(tmin=arrival_time,
                    tmax=arrival_time,
                    phasename='P',
                    nslc_ids=(t.codes,))
    markers.append(m)

# Finally, let's scrutinize these traces.
trace.snuffle(synthetic_traces, markers=markers)
