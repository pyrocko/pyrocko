from pyrocko.example import get_example_data

import obspy
from pyrocko import obspy_compat
obspy_compat.plant()

get_example_data('test.mseed')

# Read in MiniSEED data through ObsPy
stream = obspy.read('test.mseed')
stream.snuffle()
