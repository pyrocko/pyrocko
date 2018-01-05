from pyrocko import io
from pyrocko.example import get_example_data

get_example_data('test.mseed')
traces = io.load('test.mseed')

for it, t in enumerate(traces):
    with open('test-%i.txt' % it, 'w') as f:
        for tim, val in zip(t.get_xdata(), t.get_ydata()):
            f.write('%20f %20g\n' % (tim, val))
