import mseed_ext
from mseed_ext import HPTMODULUS, MSeedError
import trace
import os
from util import reuse

def load(filename, getdata=True):

    mtime = os.stat(filename)[8]
    traces = []
    for tr in mseed_ext.get_traces( filename, getdata ):
        network, station, location, channel = tr[1:5]
        tmin = float(tr[5])/float(HPTMODULUS)
        tmax = float(tr[6])/float(HPTMODULUS)
        deltat = reuse(float(1.0)/float(tr[7]))
        ydata = tr[8]
        
        traces.append(trace.Trace(network, station, location, channel, tmin, tmax, deltat, ydata, mtime=mtime))
    
    return traces
    
def as_tuple(tr):
    itmin = int(round(tr.tmin*HPTMODULUS))
    itmax = int(round(tr.tmax*HPTMODULUS))
    srate = 1.0/tr.deltat
    return (tr.network, tr.station, tr.location, tr.channel, 
            itmin, itmax, srate, tr.get_ydata())

def save(traces, filename_template):
    fn_tr = {}
    for tr in traces:
        fn = tr.fill_template(filename_template)
        if fn not in fn_tr:
            fn_tr[fn] = []
        
        fn_tr[fn].append(tr)
        
    for fn, traces_thisfile in fn_tr.items():
        trtups = []
        traces_thisfile.sort(lambda a,b: cmp(a.full_id, b.full_id))
        for tr in traces_thisfile:
            trtups.append(as_tuple(tr))
        
        try:
            mseed_ext.store_traces(trtups, fn)
        except MSeedError, e:
            raise MSeedError( str(e) + ' (while storing traces to file \'%s\')' % fn)
            
    return fn_tr.keys()
    
    
