from pyrocko.example import get_example_data

import obspy
from pyrocko import obspy_compat
obspy_compat.plant()

get_example_data('test.mseed')
get_example_data('responses.xml')

# Read in MiniSEED data through ObsPy
inv = obspy.read_inventory('responses.xml')
stream = obspy.read('test.mseed')

# Start the Snuffler
stream.snuffle(inventory=inv)
