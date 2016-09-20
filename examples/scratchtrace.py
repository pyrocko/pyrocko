
from pyrocko import trace, util, io
import numpy as num

nsamples = 100
tmin = util.str_to_time('2010-02-20 15:15:30.100')
data = num.random.random(nsamples)
t1 = trace.Trace(
    station='TEST', channel='Z', deltat=0.5, tmin=tmin, ydata=data)
t2 = trace.Trace(
    station='TEST', channel='N', deltat=0.5, tmin=tmin, ydata=data)

# all traces in one file
io.save([t1, t2], 'my_precious_traces.mseed')

# each file one channel
io.save([t1, t2], 'my_precious_trace_%(channel)s.mseed')
