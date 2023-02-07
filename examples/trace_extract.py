from pyrocko import io
from pyrocko.example import get_example_data

get_example_data('test.mseed')

traces = io.load('test.mseed')
t = traces[0]
print('original:', t)

# extract a copy of a part of t
extracted = t.chop(t.tmin+10, t.tmax-10, inplace=False)
print('extracted:', extracted)

# in-place operation modifies t itself
t.chop(t.tmin+10, t.tmax-10)
print('modified:', t)
