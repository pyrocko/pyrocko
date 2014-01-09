from struct import unpack
import os, re

from pyrocko import trace
from pyrocko.util import reuse, ensuredirs
from pyrocko.io_common import FileLoadError, FileSaveError

class CodeTooLong(FileSaveError):
    pass

def iload(filename, load_data=True):
    from pyrocko import mseed_ext

    try:
        traces = []
        for tr in mseed_ext.get_traces( filename, load_data ):
            network, station, location, channel = tr[1:5]
            tmin = float(tr[5])/float(mseed_ext.HPTMODULUS)
            tmax = float(tr[6])/float(mseed_ext.HPTMODULUS)
            try:
                deltat = reuse(float(1.0)/float(tr[7]))
            except ZeroDivisionError, e:
                raise FileLoadError('Trace in file %s has a sampling rate of zero.' % filename)
            ydata = tr[8]
            
            traces.append(trace.Trace(network, station, location, channel, tmin, tmax, deltat, ydata))
        
        for tr in traces:
            yield tr
    
    except (OSError, mseed_ext.MSeedError), e:
        raise FileLoadError(str(e))
    
def as_tuple(tr):
    from pyrocko import mseed_ext
    itmin = int(round(tr.tmin*mseed_ext.HPTMODULUS))
    itmax = int(round(tr.tmax*mseed_ext.HPTMODULUS))
    srate = 1.0/tr.deltat
    return (tr.network, tr.station, tr.location, tr.channel, 
            itmin, itmax, srate, tr.get_ydata())



def save(traces, filename_template, additional={}):
    from pyrocko import mseed_ext
    for tr in traces:
        for code, maxlen, val in zip(
                ['network', 'station', 'location', 'channel'],
                [2, 5, 2, 3],
                tr.nslc_id):

            if len(val) > maxlen:
                raise CodeTooLong(
                        '%s code too long to be stored in MSeed file: %s' % 
                        (code, val))
            
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
        except mseed_ext.MSeedError, e:
            raise FileSaveError( str(e) + ' (while storing traces to file \'%s\')' % fn)
            
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
        bads = []
        for sex in '<>':
            bad = False
            fmt = sex + '6s1s1s5s2s3s2s10sH2h4Bl2H'
            vals = unpack(fmt, rec[:48])
            fmt_btime = sex + 'HHBBBBH'
            tvals = unpack(fmt_btime, vals[7])
            if tvals[1] < 1 or tvals[1] > 367 or tvals[2] > 23 or \
                    tvals[3] > 59 or tvals[4] > 60 or tvals[6] > 9999:
                bad = True

            bads.append(bad)

        if all(bads):
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
