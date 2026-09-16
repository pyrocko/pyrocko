from pyrocko import io, util
from pyrocko.squirrel import Squirrel
from pyrocko.example import get_example_data

''' Chop a dataset into one-hour windows and reorganize it into hour-files.

For simple cases like this, the same result can also be achieved without
writing any code, using the command line tool `squirrel jackseis`:

    squirrel jackseis --add test.mseed --tinc 3600 \\
        --out-path 'test_hourfiles/hourfile-%(wmin)s.mseed'
'''

# Download test file
get_example_data('test.mseed')

# All data access happens through a Squirrel instance. Could give
# directories or thousands of filenames here, or even add online data
# sources with sq.add_fdsn(...).
sq = Squirrel()
sq.add(['test.mseed'])

# Get timestamp for full hour before first data sample in all selected
# traces.
tmin, _ = sq.get_time_span()
tmin = util.hour_start(tmin)

# Iterate over the data, with a window length of one hour. This will
# automatically connect adjacent traces from separate files as needed.
for batch in sq.chopper_waveforms(tmin=tmin, tinc=3600.):
    if batch.traces:  # the list could be empty due to gaps
        timestring = util.time_to_str(batch.tmin, format='%Y-%m-%d_%H')
        filepath = 'test_hourfiles/hourfile-%s.mseed' % timestring
        io.save(batch.traces, filepath)
