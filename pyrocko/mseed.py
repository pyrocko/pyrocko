import mseed_ext
from mseed_ext import HPTMODULUS, MSeedError
import trace
import os, re
from util import reuse, ensuredirs
from struct import unpack
from io_common import FileLoadError

def iload(filename, load_data=True):

    try:
        traces = []
        for tr in mseed_ext.get_traces( filename, load_data ):
            network, station, location, channel = tr[1:5]
            tmin = float(tr[5])/float(HPTMODULUS)
            tmax = float(tr[6])/float(HPTMODULUS)
            try:
                deltat = reuse(float(1.0)/float(tr[7]))
            except ZeroDivisionError, e:
                raise MSeedError('Trace in file %s has a sampling rate of zero.' % filename)
            ydata = tr[8]
            
            traces.append(trace.Trace(network, station, location, channel, tmin, tmax, deltat, ydata))
        
        for tr in traces:
            yield tr
    
    except (OSError, MSeedError), e:
        raise FileLoadError(e)
    
def as_tuple(tr):
    itmin = int(round(tr.tmin*HPTMODULUS))
    itmax = int(round(tr.tmax*HPTMODULUS))
    srate = 1.0/tr.deltat
    return (tr.network, tr.station, tr.location, tr.channel, 
            itmin, itmax, srate, tr.get_ydata())

def save(traces, filename_template, additional={}):
    fn_tr = {}
    for tr in traces:
        fn = tr.fill_template(filename_template, **additional)
        if fn not in fn_tr:
            fn_tr[fn] = []
        
        fn_tr[fn].append(tr)
        
    for fn, traces_thisfile in fn_tr.items():
        trtups = []
        traces_thisfile.sort(lambda a,b: cmp(a.full_id, b.full_id))
        for tr in traces_thisfile:
            trtups.append(as_tuple(tr))
        
        ensuredirs(fn)
        try:
            mseed_ext.store_traces(trtups, fn)
        except MSeedError, e:
            raise MSeedError( str(e) + ' (while storing traces to file \'%s\')' % fn)
            
    return fn_tr.keys()

tcs = {}
def detect(first512):

    if len(first512) < 256:
        return False

    rec = first512
    
    try:
        sequence_number = int(rec[:6])
    except:
        return False
    if sequence_number < 0:
        return False

    type_code = rec[6]
    if type_code in 'DRQM':
        fmt = '>6s1s1s5s2s3s2s10sH2h4Bl2H'
        vals = unpack(fmt, rec[:48])
        fmt_btime = '>HHBBBBH'
        tvals = unpack(fmt_btime, vals[7])
        if tvals[1] < 1 or tvals[1] > 367 or tvals[2] > 23 or tvals[3] > 59 or tvals[4] > 60 or tvals[6] > 9999:
            return False

        #nblockettes = vals[-4]
        #offset_next_blockette = vals[-1]
        #if offset_next_blockette + 8 < len(rec):
        #    fmt_dr_head = '>2H4b'
        #    vals_dr_head = unpack(fmt_dr_head, rec[offset_next_blockette:offset_next_blockette+8])
        #    blockette_type, offset_next_blockette, encoding, word_order, data_record_length_exponent, reserved = vals_dr_head
            
    else:
        if not type_code in 'VAST':
            return False

        continuation_code = rec[7]
        if continuation_code != ' ':
            return False

        blockette_type  = rec[8:8+3]
        if not re.match(r'^\d\d\d$', blockette_type):
            return False

        try:
            blockette_length = int(rec[11:11+4])
        except:
            return False

        if blockette_length < 7:
            return False

    return True
