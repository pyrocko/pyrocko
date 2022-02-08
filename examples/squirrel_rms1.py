import numpy as np
from pyrocko.progress import progress
from pyrocko.util import time_to_str as tts, str_to_time_fillup as stt
from pyrocko.squirrel import Squirrel


def rms(data):
    return np.sqrt(np.sum(data**2))


# Time span (Hunga Tonga explosion was on 2022-01-15):
tmin = stt('2022-01-14')
tmax = stt('2022-01-17')

# Filter range:
fmin = 0.01
fmax = 0.05

# Enable progress bars:
progress.set_default_viewer('terminal')

# All data access will happen through a single Squirrel instance:
sq = Squirrel()

# Add local data directories:
# sq.add('data/2022')

# Add online data source:
# Enable access to BGR's FDSN web service and restrict to GR network and
# LH channels only:
sq.add_fdsn('bgr', query_args=dict(network='GR', channel='LH?'))

# Ensure meta-data is up to date for the selected time span:
sq.update(tmin=tmin, tmax=tmax)

# Allow waveform download for station BFO. This does not download anything
# yet, it just enables downloads later, when needed. Omit `tmin`, `tmax`,
# and `codes` to enable download of all everything.
sq.update_waveform_promises(tmin=tmin, tmax=tmax, codes='*.BFO.*.*')
print(sq)

# Iterate window-wise, with some overlap over the data:
for batch in sq.chopper_waveforms(
        tmin=tmin,
        tmax=tmax,
        codes='*.*.*.LHZ',
        tinc=3600.,              # One hour time windows (payload).
        tpad=1.0/fmin,           # Add padding to absorb filter artifacts.
        want_incomplete=False,   # Skip incomplete windows.
        snap_window=True):       # Start all windows at full hours.

    for tr in batch.traces:

        # Filtering will introduce some artifacts in the padding interval.
        tr.highpass(4, fmin)
        tr.lowpass(4, fmax)

        # Cut off the contaminated padding. Trim to the payload interval.
        tr.chop(batch.tmin, batch.tmax)

        # Print channel codes, time-mark and RMS value of the hour.
        print(tr.str_codes, tts(tr.tmin), rms(tr.ydata))
