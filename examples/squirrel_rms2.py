import numpy as np
from pyrocko.util import time_to_str as tts
from pyrocko import squirrel


def rms(data):
    return np.sqrt(np.sum(data**2))


sq = squirrel.from_command(
    description='Report hourly RMS values.')

fmin = 0.01
fmax = 0.05

for batch in sq.chopper_waveforms(
        tinc=3600.,
        tpad=1.0/fmin,
        want_incomplete=False,
        snap_window=True):

    for tr in batch.traces:
        tr.highpass(4, fmin)
        tr.lowpass(4, fmax)
        tr.chop(batch.tmin, batch.tmax)
        print(tr.str_codes, tts(tr.tmin), rms(tr.ydata))
