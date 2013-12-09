from pyrocko import trace
from math import sqrt
import numpy as num

# Let's create three traces: One trace as the reference (rt) and two as test traces (tt1 and tt2):
ydata1 = data = num.random.random(1000)
ydata2 = data = num.random.random(1000)
rt = trace.Trace(station='REF', ydata=ydata1)
tt1 = trace.Trace(station='TT1', ydata=ydata1)
tt2 = trace.Trace(station='TT2', ydata=ydata2)

# the misfit method needs an iterable object containing traces:
test_candidates = [tt1, tt2]

# Define a fader to apply before fft.
taper = trace.CosFader(xfade=5)

# Define a frequency response to apply before performing the inverse fft.
# This can be basically any funtion, as long as it contains a function called *evaluate*, 
# which evaluates the frequency response function at a given list of frequencies.
# Please refer to the :py:class:`FrequencyResponse` class or its subclasses for examples. 
fresponse = trace.FrequencyResponse()        

# Combine all information in one misfit setup:
setup = trace.MisfitSetup(norm=2,
                          taper=taper,
                          domain='time_domain',
                          freqlimits=(1,2,20,40),
                          frequency_response=fresponse)

# Calculate the misfit for each test candidate:
i = 0
for m, n in rt.misfit(candidates=test_candidates, setups=setup):
    M = m/n
    print 'L2 misfit of %s and %s is %s' % (rt.station, test_candidates[i].station, M)
    i += 1 

# Finally, we want to dump the misfit setup that has been used in a yaml file:
f = open('my_misfit_setup.txt', 'w')
f.write(setup.dump())
f.close()
