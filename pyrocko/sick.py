'''File IO module for SICK traces format.'''

import os, sys
from struct import unpack, pack
from cStringIO import StringIO
import numpy as num
try:
    from hashlib import sha1
except:
    from sha import sha as shamod
    sha1 = shamod.new

from pyrocko import util

size_record_header = 64
no_hash = '\0' * 20
sick_version = '0000'

record_types = ('trace_header', 'trace_data')

numtypes = {
        '@i2': (num.int16, '>i2'),
        '@i4': (num.int32, '>i4'),
        '@i8': (num.int64, '>i8'),
        '@u2': (num.uint16, '>u2'),
        '@u4': (num.uint32, '>u4'),
        '@u8': (num.uint64, '>u8'),
        '@f4': (num.float32, '>f4'),
        '@f8': (num.float64, '>f8'),
    }

numtype2type = dict( [ (v[0],k) for (k,v) in numtypes.iteritems() ] )

def serialize(a):
    s = StringIO()
    for x in a:
        s.write(pack('>Q', len(x)))
        s.write(x)

    return s.getvalue()

def deserialize(s):
    ipos = 0
    a = []
    while ipos < len(s):
        l = unpack('>Q', s[ipos:ipos+8])
        a.append(s[ipos+8:ipos+8+l])
        ipos += 8+l

    return a

def packer(fmt):
    return ((lambda x: pack('>'+fmt, x)), (lambda x: unpack('>'+fmt, x)[0]))

def unpack_array(fmt, data):
    return num.fromstring(data, dtype=numtypes[fmt][1]).astype(numtypes[fmt][0])

def pack_array(fmt, data):
    return data.astype(numtypes[fmt][1]).tostring()


def array_packer(fmt):
    return ((lambda x: pack_array(fmt, x)), (lambda x: unpack_array(fmt, x))) 

def encoding_packer(enc):
    return ((lambda x: x.encode(enc)), (lambda x: x.decode(enc)))

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
        '@i2': array_packer('@i2'),
        '@i4': array_packer('@i4'),
        '@i8': array_packer('@i8'),
        '@u2': array_packer('@u2'),
        '@u4': array_packer('@u4'),
        '@u8': array_packer('@u8'),
        '@f4': array_packer('@f4'),
        '@f8': array_packer('@f8'),
    }

def pack_value(type, value):
    try:
        return castings[type][0](value)
    except Exception, e:
        raise SickError('Packing value failed (type=%s, value=%s, error=%s).' % (type,str(value)[:500], e))
    
def unpack_value(type, value):
    try:
        return castings[type][1](value)
    except:
        raise SickError('Unpacking value failed (type=%s, error=%s).' % (type))

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
        
        if check_hash and hash != no_hash:
            if sha1(data).digest() != hash:
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
        hash = sha1(data).digest()
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
        if format is not None and format[key] != type:
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
            a.append(format[key])
            a.append(pack_value(format[key], value))

    return serialize(a)

trace_header_record_format = {
            'network': 'string',
            'station': 'string',
            'location': 'string',
            'channel': 'string',
            'tmin': 'time_string',
            'tmax': 'time_string',
            'deltat': 'f8',
    }

def load(fn, load_data=True):
    f = open(fn, 'r') 
    traces = []
    while True: 
        try:
            d = unpack_record(read_record(f, 'trace_header'), trace_header_record_format)
            if load_data:
                d.update( unpack_trace_data_record(read_record(f)) )
            else:
                skip_record(f)
            
            tr = trace.Trace(**d)
            traces.append(tr)

        except NoDataAvailable:
            break

    return traces

def extract(tr, format):
    d = {}
    for k in format.keys():
        d[k] = getattr(tr,k)
    return d

def pack_trace_header_record(tr):
    format = trace_header_record_format
    return pack_record(extract(tr,format), format)

def pack_trace_data_record(tr):

    format = {}
    type = numtype2type[tr.ydata.dtype.type]
    format['ydata'] = type

    return pack_record(extract(tr, format), format)

def save(traces, fn):
    f = open(fn, 'w')
    for tr in traces:
        write_record(f, 'trace_header', pack_trace_header_record(tr), make_hash=True)
        write_record(f, 'trace_data', pack_trace_data_record(tr), make_hash=True)

    f.close()



