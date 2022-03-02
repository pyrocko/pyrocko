#!/usr/bin/env python3
import numpy as np
from pyrocko.util import time_to_str, str_to_time_fillup as stt
from pyrocko import squirrel


def rms(data):
    return np.sqrt(np.sum(data**2))


tmin = stt('2022-01-14')
tmax = stt('2022-01-17')

fmin = 0.01
fmax = 0.05

# Setup Squirrel instance with waveforms from files given on command line.
# Supports all common options offered by Squirrel tool commands like `squirrel
# scan`:  --help, --loglevel, --progress, --add, --include, --exclude,
# --optimistic, --format, --kind, --persistent, --update, --dataset

parser = squirrel.SquirrelArgumentParser(
    description='Report hourly RMS values.')

parser.add_squirrel_selection_arguments()
args = parser.parse_args()
sq = args.make_squirrel()

sq.update(tmin=tmin, tmax=tmax)
sq.update_waveform_promises(tmin=tmin, tmax=tmax, codes='*.BFO.*.*')
sq.update_responses(tmin=tmin, tmax=tmax, codes='*.BFO.*.*')

tpad = 1.0 / fmin

for batch in sq.chopper_waveforms(
        tmin=tmin,
        tmax=tmax,
        codes='*.*.*.LHZ',
        tinc=3600.,
        tpad=tpad,
        want_incomplete=False,
        snap_window=True):

    for tr in batch.traces:
        resp = sq.get_response(tr).get_effective()
        tr = tr.transfer(
            tpad, (0.5*fmin, fmin, fmax, 2.0*fmax), resp, invert=True,
            cut_off_fading=False)

        tr.chop(batch.tmin, batch.tmax)
        print(tr.str_codes, time_to_str(batch.tmin), rms(tr.ydata))
