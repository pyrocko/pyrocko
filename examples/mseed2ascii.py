from pyrocko import io

traces = io.load('test.mseed')

for it, t in enumerate(traces):
    f = open('test-%i.txt' % it, 'w')
    
    for tim, val in zip(t.get_xdata(), t.get_ydata()):
        f.write( '%20f %20g\n' % (tim,val) )
    
    f.close()
    