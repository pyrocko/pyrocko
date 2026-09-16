from pyrocko import io, trace, util
from pyrocko.squirrel import Squirrel
from pyrocko.example import get_example_data

'''
Downsample a whole dataset to a common sampling rate.

For simple cases like this, the same result can also be achieved without
writing any code, using the command line tool `squirrel jackseis`:

    squirrel jackseis --add test.mseed --downsample 0.5 --tinc 3600 \\
        --out-path 'downsampled/%(station)s_%(channel)s_%(wmin)s.mseed'
'''

# Download test file
get_example_data('test.mseed')

sq = Squirrel()
sq.add(['test.mseed'])

tinc = 3600.
target_deltat = 2.0  # test.mseed is sampled at 1 Hz (deltat=1s)
tpad = 50 * target_deltat

# Iterate over the data, with a window length of one hour and padding on
# either side to absorb downsampling filter edge effects. Start time windows
# at full hours.
for batch in sq.chopper_waveforms(tinc=tinc, tpad=tpad, snap_window=True):
    traces = []
    for tr in batch.traces:
        tr.downsample_to(target_deltat, snap=True, demean=False)

        try:
            # remove padding
            tr.chop(batch.tmin, batch.tmax)
            traces.append(tr)
        except trace.NoData:
            # can happen for the trailing, incomplete window
            pass

    if traces:  # the list could be empty due to gaps
        timestring = util.time_to_str(batch.tmin, format='%Y-%m-%d_%H')
        filepath = 'downsampled/%(station)s_%(channel)s_%(mytimestring)s.mseed'
        io.save(traces, filepath, additional={'mytimestring': timestring})


# now look at the result with
#   > squirrel snuffler --add downsampled/
