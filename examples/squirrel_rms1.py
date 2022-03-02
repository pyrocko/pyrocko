import numpy as np
from pyrocko import progress
from pyrocko.util import setup_logging, time_to_str, str_to_time_fillup as stt
from pyrocko.squirrel import Squirrel

# Enable logging and progress bars:
setup_logging('squirrel_rms1.py', 'info')
progress.set_default_viewer('terminal')   # or 'log' to just log progress


def rms(data):
    return np.sqrt(np.sum(data**2))


# Time span (Hunga Tonga explosion was on 2022-01-15):
tmin = stt('2022-01-14')
tmax = stt('2022-01-17')

# Frequency band for restitution:
fmin = 0.01
fmax = 0.05

# All data access will happen through a single Squirrel instance:
sq = Squirrel()

# Uncomment to add local data directories. This will only index the waveform
# headers and not load the actual waveform data at this point. If the files are
# already known and unmodified, it will use cached information.
# sq.add('data/2022')

# Add online data source. Enable access to BGR's FDSN web service and restrict
# to GR network and LH channels only:
sq.add_fdsn('bgr', query_args=dict(network='GR', channel='LH?'))

# Ensure meta-data from online sources is up to date for the selected time
# span:
sq.update(tmin=tmin, tmax=tmax)

# Allow waveform download for station BFO. This does not download anything
# yet, it just enables downloads later, when needed. Omit `tmin`, `tmax`,
# and `codes` to enable download of everything selected in `add_fdsn(...)`.
sq.update_waveform_promises(tmin=tmin, tmax=tmax, codes='*.BFO.*.*')

# Make sure instrument response information is available for the selected data.
sq.update_responses(tmin=tmin, tmax=tmax, codes='*.BFO.*.*')

# Time length for padding (half of overlap).
tpad = 1.0 / fmin

# Iterate window-wise, with some overlap over the data:
for batch in sq.chopper_waveforms(
        tmin=tmin,
        tmax=tmax,
        codes='*.*.*.LHZ',
        tinc=3600.,              # One hour time windows (payload).
        tpad=tpad,               # Add padding to absorb processing artifacts.
        want_incomplete=False,   # Skip incomplete windows.
        snap_window=True):       # Start all windows at full hours.

    for tr in batch.traces:
        resp = sq.get_response(tr).get_effective()

        # Restitution via spectral division. This will introduce artifacts
        # in the padding area in the beginning and end of the trace.
        tr = tr.transfer(
            tpad,                  # Fade in / fade out length.
            (0.5*fmin, fmin,
             fmax, 2.0*fmax),      # Frequency taper.
            resp,                  # Complex frequency response of instrument.
            invert=True,           # Use inverse of instrument response.
            cut_off_fading=False)  # Disable internal trimming.

        # Cut off the contaminated padding. Trim to the payload interval.
        tr.chop(batch.tmin, batch.tmax)

        # Print channel codes, timestamp and RMS value of the hour.
        print(tr.str_codes, time_to_str(batch.tmin), rms(tr.ydata))
