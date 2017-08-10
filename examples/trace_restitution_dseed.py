from pyrocko import io
from pyrocko.io import rdseed
from pyrocko.example import get_example_data

# Note: this example requires *rdseed* available e.g. at:
# https://ds.iris.edu/

# Download example data
get_example_data('station_info.dseed')
get_example_data('traces_italy.mseed')

# read dataless seed file
seed_volume = rdseed.SeedVolumeAccess('station_info.dseed')

# load traces:
traces = io.load('traces_italy.mseed')

out_traces = []

for tr in traces:

    # get response information
    resp = seed_volume.get_pyrocko_response(tr, target='dis')

    # restitute
    displacement = tr.transfer(
        tfade=100.,
        freqlimits=(0.01, 0.02, 5., 10.),
        invert=True,
        transfer_function=resp)

    # change channel id, so we can distinguish the traces in a trace viewer
    displacement.set_codes(channel='D'+tr.channel[-1])
    out_traces.append(displacement)

io.save(out_traces, 'displacement.mseed')
