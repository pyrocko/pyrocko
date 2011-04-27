'''File IO module for SICK traces format.'''

import os
from struct import unpack, pack

size_record_header = 64
no_hash = '\0' * 20
sick_version = '0000'

record_types = ('trace_header', 'trace_data')

type2store = {
        num.int16: '>i2',
        num.int32: '>i4',
        num.int64: '>i8',
        num.uint16: '>u2',
        num.uint32: '>u4',
        num.uint64: '>u8',
        num.float32: '>f4',
        num.float64: '>f8',
    }

store2type = dict( [ (v,k) for (k,v) in type2store.iteritems() ] )

def serialize(a):
    s = StringIO()
    for x in a:
        s.write(pack('>u', len(x)))
        s.write(x)

    return s.getvalue()

def deserialize(s):
    ipos = 0
    a = []
    while ipos < len(s):
        l = unpack('>u', s[ipos])
        a.append(s[ipos+1:ipos+1+l])
        ipos += 1+l

    return a

def packer(fmt):
    return ((lambda x: pack('>'+fmt, x)), (lambda x: unpack('>'+fmt, x)[0]))

def unpack_array(format, data):
    return num.fromstring(data, dtype=format).astype(store2type[format])

def pack_array(format, data):
    return data.astype(format).tostring()


def array_packer(fmt):
    return ((lambda x pack_array(fmt, x)), (lambda x: unpack_array(fmt, x))) 

def noop(x):
    return x

castings = {
        'i2': packer('h'),
        'i4': packer('i'),
        'i8': packer('q'),
        'u2': packer('H'),
        'u4': packer('I'),
        'u8': packer('Q'),
        'f4': packer('f'),
        'f8': packer('d'),
        'string': (noop, noop),
        'time_string': (util.time_to_str, util.str_to_time),
        '@i2': array_packer('>i2'),
        '@i4': array_packer('>i4'),
        '@i8': array_packer('>i8'),
        '@u2': array_packer('>u2'),
        '@u4': array_packer('>u4'),
        '@u8': array_packer('>u8'),
        '@f4': array_packer('>f4'),
        '@f8': array_packer('>f8'),
    }

def pack_value(type, value):
    try:
        return castings[type][0](value)
    except:
        raise SickError('Packing value failed.')
    
def unpack_value(type, value):
    try:
        return castings[type][1](value)
    except:
        raise SickError('Unpacking value failed.')

class SickError(Exception):
    pass

class NoDataAvailable(Exception):
    pass

class WrongRecordType(Exception):
    pass

def read_record_header(f):
    data = f.read(size_record_header)

    if len(data) == 0:
        raise NoDataAvailable()

    if len(data) != size_record_header:
        raise SickError('Read returned less data than expected.')
    
    sick, version, size_record, size_payload, hash, type = unpack('>4s4sQQ20s20s', data)

    if sick != 'SICK':
        raise SickError('SICK record identifier missing.')
    
    if version != sick_version:
        raise SickError('SICK file version %s not supported.' % version)

    type = type.rstrip()

    return size_record, size_payload, hash, type

def write_record_header(f, size_record, size_payload, hash, type):
    data = pack('>4s4sQQ20s20s', 'SICK', sick_version, size_record, size_payload, hash, type.ljust(20)[:20])
    f.write(data)

def read_record(f, check_hash=False, max_size_record=None, type_wanted=None):
    size_record, size_payload, hash, type = read_record_header(f)
    size_remaining = size_record - size_record_header

    try:
        if size_record - size_record_header < size_payload:
            raise SickError('Record has invalid record payload size.')
       
        if max_size_record is not None and size_record > max_size_record:
            raise SickError('Record size exceeds administrative limitation.')

        if type_wanted is not None and type != type_wanted:
            raise WrongRecordType('Wrong record type')

        data = f.read(size_payload)
        size_remaining = size_remaining - len(data)

        if len(data) != size_payload:
            raise SickError('Read returned less data than expected.')
        
        if check_hash and r.hash != no_hash:
            if sha1.new(data).digest() != r.hash:
                raise SickError('Hash computed from record data does not match value given in header.')

    finally:
        if size_remaining > 0:
            f.seek(size_remaining, os.SEEK_CUR)

    return data, type

def skip_record(f):
    size_record, size_payload, hash, type = read_record_header(f)
    f.seek(size_record - size_record_header, os.SEEK_CUR)

def write_record(f, type, data, min_size_record=None, make_hash=False):
    size_payload = len(data)
    size_record = size_record_header + size_payload

    if min_size_record is not None:
        size_record = max(min_size_record, size_record)

    if make_hash:
        hash = sha1.new(data).digest()
    else:
        hash = no_hash 

    write_record_header(f, size_record, size_payload, hash, type)
    f.write(data)

    size_padding = size_record - size_record_header - size_payload
    if size_padding > 0:
        f.write( '\0' * size_padding )

def unpack_record(data, format):
    vals = deserialize(data)
    d = {}
    for i in xrange(0,len(vals),3):
        key, type, value = vals[i:i+3]
        if key in format and format[key] != type:
            SickError('Record value in unexpected format.')

        d[key] = unpack_value(type, value)

    for key in format:
        if key not in d:
            raise SickError('Missing record value.')
        
    return d

def pack_record(d, format):

    for key in format:
        if key not in d:
            raise MissingRecordValue()
        
    a = []
    for key, value in d.iteritems():
        if key in format:
            a.append(key)
            a.append(type)
            a.append(pack_value(format[key], value))

    return serialize(a)

def check_trace_header(d):
    c = {}
    for k in ('network', 'station', 'location', 'channel'):
        c[k] = d.pop(k, '')

    for k in ('tmin', 'tmax'):
        if k in d:
            c[k] = str_to_date(d.pop(k))
    
    if 'tmin' not in c:
        c['tmin'] = 0.0
    
    if 'tmax' not in c:
        c['tmax'] = None

    c['deltat'] = float(d.pop('deltat'))
    format = d.pop('format', None)
    assert format in store2type 

    return c, format

def unpack_trace_header(data):
    toks = data.split('\0')
    return = check_trace_header(dict( [ (k,v) for (k,v) in zip(toks[0::2], toks[1::2]) ] ))
    
def pack_trace_header(tr):
    
    if tr.ydata is not None:
        assert tr.ydata.dtype in type2store
        format = type2store
    else:
        format = None 

    a = [ 'network', tr.network, 'station', tr.station, 'location', tr.location, 'channel', tr.channel,
            'tmin', date_to_str(tr.tmin), 
            'deltat', '%f' % tr.deltat ]

    if tr.ydata is not None:
        assert tr.ydata.dtype in type2store
        a.extend('format', type2store)
    return '\0'.join(a)


def unpack_trace_data(data, format):
    return num.fromstring(data, dtype=format).astype(store2type[format])

def pack_trace_data(data, format):
    return data.astype(format).tostring()


def load(fn, load_data=True):
   
    traces = []
    while True: 
        try:
            d, format = unpack_trace_header(read_record(f))
        
            if format is not None: 
                if load_data:
                    d['ydata'] = unpack_trace_data(read_record(f), format)
                else:
                    skip_record(f)
                    d['ydata'] = None
            
            tr = trace.Trace(**d)
            traces.append(tr)

        except NoDataAvailable:
            break

    return traces

def save(traces, fn):

    for tr in traces:
        write_record( f, pack_trace_header(tr)

