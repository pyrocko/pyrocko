import sys, struct
from collections import namedtuple
import numpy as num
from pyrocko.io_common import FileLoadError
from pyrocko import util, trace

guralp_zero = util.str_to_time('1989-11-17 00:00:00')

class GCFLoadError(FileLoadError):
    pass

class EOF(Exception):
    pass

Header = namedtuple('Header',
        '''
        block_type
        system_id
        stream_id
        instrument_type
        time
        gain
        ttl
        sample_rate
        compression
        nrecords
        '''.split())

def read(f, n, eof_ok=False):
    data = f.read(n)
    if eof_ok and len(data) == 0:
        raise EOF()

    if len(data) != n:
        raise GCFLoadError('Unexpected end of file')
    
    return data

def read_header(f, endianness='>'):
    e = endianness
    data = read(f, 16, eof_ok=True)

    isystem_id, istream_id = struct.unpack(e+'II', data[:8])
    ex = isystem_id & 0x80000000
    if ex:
        ex2 = isystem_id & (1 << 30)
        if ex2:
            system_id = util.base36encode(isystem_id & (2**21 - 1))
        else:
            system_id = util.base36encode(isystem_id & (2**26 - 1))

        instrument_type = (isystem_id >> 26) & 0b1
        gain = [None, 1, 2, 4, 8, 16, 32, 64][(isystem_id >> 27) & 0b111]
    else:

        system_id = util.base36encode(isystem_id)
        instrument_type = None
        gain = None

    stream_id = util.base36encode(istream_id)

    i_day_second = struct.unpack(e+'I', data[8:12])[0]
    iday = i_day_second >> 17
    isecond = i_day_second & 0x7fff
    time = (iday*24*60*60) + guralp_zero + isecond

    ittl, israte, compression, nrecords = struct.unpack(e+'BBBB', data[12:])

    if israte == 0:
        if compression == 4 and stream_id[-2:] == '00':
            block_type = 'status_block'

        elif compression == 4 and stream_id[-2:] == '01':
            block_type = 'unified_status_block'

        elif compression == 4 and stream_id[-2:] == 'SM':
            block_type = 'strong_motion_block'

        elif stream_id[-2:] == 'CD':
            block_type = 'cd_status_block'

        elif compression == 4 and stream_id[-2:] == 'BP':
            block_type = 'byte_pipe_block'
        else:
            block_type = 'unknown_block'
            
        return Header(block_type, system_id, stream_id, instrument_type, 
                      time, gain, ittl, 0.0, compression, nrecords)
    else:
        block_type = 'data_block'

        sample_rate_tab = {
                157: (0.1, None),
                161: (0.125, None),
                162: (0.2, None),
                164: (0.25, None),
                167: (0.5, None),
                171: (400., 8),
                174: (500., 2),
                176: (1000., 4),
                179: (2000., 8),
                181: (4000., 16) }

        if israte in sample_rate_tab:
            sample_rate, tfod = sample_rate_tab[israte]
        else:
            sample_rate = float(israte)
            tfod = None

        if tfod is not None:
            toff = (compression >> 4) / tfod
            compression = compression & 0b1111
            time += toff

        return Header(block_type, system_id, stream_id, instrument_type, 
                      time, gain, ittl, sample_rate, compression, nrecords)

def read_data(f, h, endianness='>'):
    e = endianness
    data = read(f, 1024 - 16)
    first = struct.unpack(e+'i', data[0:4])[0]
    dtype = { 1: e+'i4', 2: e+'i2', 4: e+'i1' }
    if h.compression not in dtype:
        raise GCFLoadError('Unsupported compression code: %i' % h.compression)

    nsamples = h.compression * h.nrecords
    difs = num.fromstring(data[4:4+h.nrecords*4], dtype=dtype[h.compression], 
                          count=nsamples)
    samples = difs.astype(num.int)
    samples[0] += first
    samples = num.cumsum(samples)
    last = struct.unpack(e+'i', data[4+h.nrecords*4:4+h.nrecords*4+4])[0]
    if last != samples[-1]:
        raise GCFLoadError('Checksum error occured')

    return samples

def read_status(f, h):
    data = read(f, 1024 - 16)
    return data[:h.nrecords*4]

def iload(filename, load_data=True):
    traces = {}

    f = open(filename, 'r')
    try:
        p1 = 0
        while True:

            p2 = f.tell()
            h = read_header(f)
            if h.block_type == 'data_block':
                deltat = 1.0 / h.sample_rate
                if load_data:
                    samples = read_data(f, h)
                    tmax = None
                else:
                    f.seek(1024 - 16, 1)
                    samples = None
                    tmax = h.time + h.nrecords * h.compression * deltat

                nslc = ('', h.system_id, '', h.stream_id)

                if nslc in traces:
                    tr = traces[nslc]
                    if abs((tr.tmax + tr.deltat) - h.time) < deltat*0.0001:
                        if samples is not None:
                            tr.append(samples)
                        else:
                            tr.tmax = tmax

                    else:
                        del traces[nslc]
                        yield tr
                        
                if nslc not in traces: 
                    traces[nslc] = trace.Trace(*nslc,
                            tmin = h.time,
                            deltat =  deltat,
                            ydata = samples,
                            tmax = tmax)

            else:
                f.seek(1024 - 16, 1)

    except EOF:
        for tr in traces.values():
            yield tr

if __name__ == '__main__':
    all_traces = []
    for fn in sys.argv[1:]:
        all_traces.extend(iload(fn))

    trace.snuffle(all_traces)




