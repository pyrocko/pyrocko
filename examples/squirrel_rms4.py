import argparse
import numpy as np
from pyrocko.util import time_to_str, str_to_time_fillup as stt
from pyrocko import squirrel


def rms(data):
    return np.sqrt(np.sum(data**2))


parser = argparse.ArgumentParser()
squirrel.add_squirrel_selection_arguments(parser)

parser.add_argument(
    '--fmin',
    dest='fmin',
    metavar='FLOAT',
    default=0.01,
    type=float,
    help='Corner of highpass [Hz].')

parser.add_argument(
    '--fmax',
    dest='fmax',
    metavar='FLOAT',
    default=0.05,
    type=float,
    help='Corner of lowpass [Hz].')

args = parser.parse_args()

sq = squirrel.squirrel_from_selection_arguments(args)

fmin = args.fmin
fmax = args.fmax

tmin = stt('2022-01-14')
tmax = stt('2022-01-14 01')

tpad = 1.0 / fmin

for batch in sq.chopper_waveforms(
        tmin=tmin,
        tmax=tmax,
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
        print(str(tr.codes), time_to_str(tr.tmin), rms(tr.ydata))
