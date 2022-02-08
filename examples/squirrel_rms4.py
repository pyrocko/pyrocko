import argparse
import numpy as np
from pyrocko.util import time_to_str as tts
from pyrocko.squirrel.tool import common as squirrel_tool_common


def rms(data):
    return np.sqrt(np.sum(data**2))


parser = argparse.ArgumentParser()
squirrel_tool_common.add_selection_arguments(parser)

parser.add_argument(
    '--fmin',
    dest='fmin',
    metavar='FLOAT',
    type=float,
    help='Corner of highpass [Hz].')

parser.add_argument(
    '--fmax',
    dest='fmax',
    metavar='FLOAT',
    type=float,
    help='Corner of lowpass [Hz].')

args = parser.parse_args()

sq = squirrel_tool_common.squirrel_from_selection_arguments(args)

fmin = args.fmin
fmax = args.fmax

for batch in sq.chopper_waveforms(
        tinc=3600.,
        tpad=1.0/fmin if fmin is not None else 0.0,
        want_incomplete=False,
        snap_window=True):

    for tr in batch.traces:

        if fmin is not None:
            tr.highpass(4, fmin)

        if fmax is not None:
            tr.lowpass(4, fmax)

        tr.chop(batch.tmin, batch.tmax)
        print(tr.str_codes, tts(tr.tmin), rms(tr.ydata))
