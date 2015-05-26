from pyrocko import trace
from math import sqrt
import numpy as num

# Let's create three traces: One trace as the reference (rt) and two as test 
# traces (tt1 and tt2):
ydata1 = num.random.random(1000)
ydata2 = num.random.random(1000)
rt = trace.Trace(station='REF', ydata=ydata1)
candidate1 = trace.Trace(station='TT1', ydata=ydata1)
candidate2 = trace.Trace(station='TT2', ydata=ydata2)

# Define a fader to apply before fft.
taper = trace.CosFader(xfade=5)

# Define a frequency response to apply before performing the inverse fft.
# This can be basically any funtion, as long as it contains a function called
# *evaluate*, which evaluates the frequency response function at a given list
# of frequencies.
# Please refer to the :py:class:`FrequencyResponse` class or its subclasses for
# examples.
# However, we are going to use a butterworth low-pass filter in this example.
bw_filter = trace.ButterworthResponse(corner=2,
                                      order=4,
                                      type='low')

# Combine all information in one misfit setup:
setup = trace.MisfitSetup(description='An Example Setup',
                          norm=2,
                          taper=taper,
                          filter=bw_filter,
                          domain='time_domain')

# Calculate misfits of each candidate against the reference trace:
for candidate in [candidate1, candidate2]:
    misfit = rt.misfit(candidate=candidate, setup=setup)
    print 'misfit: %s, normalization: %s' % misfit

# Finally, dump the misfit setup that has been used as a yaml file for later
# re-use:
setup.dump(filename='my_misfit_setup.txt')
