#!/usr/bin/env python

from pyrocko import io
import sys

for filename in sys.argv[1:]:
    traces = io.load(filename, format='sac')
    if filename.lower().endswith('.sac'):
        out_filename = filename[:-4] + '.mseed'
    else:
        out_filename = filename + '.mseed'

    io.save(traces, out_filename)

