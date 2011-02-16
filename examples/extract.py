from pyrocko import io

traces = io.load('test.mseed')
t = traces[0]

# extract a copy of a part of t
extracted = t.chop(t.tmin+10, t.tmax-10, inplace=False)

# in-place operation modifies t itself
t.chop(t.tmin+10, t.tmax-10)
