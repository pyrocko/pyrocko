import sys, os, io
import numpy as num
import util, trace
import struct
import calendar

from pyrocko.io_common import FileLoadError

class SEGYError(Exception):
    pass

class SEGYFile:
    
    nbytes_textual_header = 3200
    nbytes_binary_header = 400
    nbytes_optional_textual_header = 3200
    nbytes_trace_header = 240
    
    def __init__(self, *args, **kwargs):
        if args:
            self.read(*args, **kwargs)
        else:
            self.clear()
            
    def __str__(self):
        pass
            
    def clear(self):
        '''Empty file record.'''
        
        # set the required attributes        
        self.delta = 1.0
        self.npts = 0
        self.b = 0.0
        self.data = [ num.arange(0, dtype=num.int32) ]
        
    def read(self, filename, load_data=True, endianness='>'):
        '''Read SEGY file.
        
           filename -- Name of SEGY file.
           load_data -- If True, the data is read, otherwise only read headers.
        '''
        
        order = endianness
        
        nbth = SEGYFile.nbytes_textual_header
        nbbh = SEGYFile.nbytes_binary_header
        nbthx = SEGYFile.nbytes_optional_textual_header
        nbtrh = SEGYFile.nbytes_trace_header
        
        # read in all data
        f = open(filename,'rb')
        
        # XXX should skip volume label
        
        filedata = f.read()
        f.close()
        
        i = 0
        if True:
            hvals = struct.unpack(order+'24H', filedata[i+3212:i+3212+24*2])
            (ntraces, nauxtraces, deltat_us, deltat_us_orig, nsamples, 
                nsamples_orig, format) = hvals[0:7]
            
            (segy_revision, fixed_length_traces, nextended_headers) = struct.unpack(order+'3H', filedata[3500:3500+3*2])        
        
            formats = { 1: (None,  4, "4-byte IBM floating-point"),
                    2: (order+'i4', 4, "4-byte, two's complement integer"),
                    3: (order+'i4', 2, "2-byte, two's complement integer"),
                    4: (None,  4, "4-byte fixed-point with gain (obolete)"),
                    5: ('f4',  4, "4-byte IEEE floating-point"),
                    6: (None,  0, "not currently used"),
                    7: (None,  0, "not currently used"),
                    8: ('i1',  1, "1-byte, two's complement integer") }
                    
            dtype = formats[format][0]
            sample_size = formats[format][1]
            if dtype is None:
                raise SEGYError('Cannot read SEG-Y files with data in format %i: %s (file=%s)' % (format, formats[format][1], filename))
        
            ipos = nbth+nbbh + nextended_headers*nbthx 
        else:
            
            ipos = 0
            ntraces = 2
            nauxtraces = 0
            fixed_length_traces = False
            sample_size = 4
            dtype = order+'i4'  
        traces = []
        for itrace in xrange(ntraces+nauxtraces):
            trace_header = filedata[ipos:ipos+nbtrh]
            if len(trace_header) != nbtrh:
                raise SEGYError('SEG-Y file incomplete (file=%s)' % filename)
            
            (scoordx,scoordy,gcoordx,gcoordy) = struct.unpack(order+'4f4', trace_header[72:72+4*4])
            (ensemblex,ensembley) = struct.unpack(order+'2f4', trace_header[180:180+2*4])
            (ensemble_num,) = struct.unpack(order+'1I', trace_header[20:24])
            (trensemble_num,) = struct.unpack(order+'1I', trace_header[24:28])
            (trace_number,)= struct.unpack(order+'1I', trace_header[0:4])
            (trace_numbersegy,)= struct.unpack(order+'1I', trace_header[4:8])
            (orfield_num,)= struct.unpack(order+'1I', trace_header[8:12])
            (ortrace_num,)= struct.unpack(order+'1I', trace_header[12:16])
            (nsamples_this, deltat_us_this) = struct.unpack(order+'2H', trace_header[114:114+2*2])
            (year,doy,hour,minute,second) = struct.unpack(order+'5H', trace_header[156:156+2*5])
            try:
                tmin = calendar.timegm((year,1,doy,hour,minute,second))
            except:
                raise SEGYError('Could not get starting date/time for trace %i in SEG-Y file %s.' % (itrace+1, filename))
            
            if fixed_length_traces:
                if (nsamples_this, deltat_us_this) != (nsamples, deltat_us):
                    raise SEGYError('Trace of incorrect length or sampling rate found in SEG-Y file (trace=%i, file=%s)' % (itrace+1, filename))
                
            if load_data:
                datablock = filedata[ipos+nbtrh:ipos+nbtrh+nsamples_this*sample_size]
                if len(datablock) != nsamples_this*sample_size:
                    raise SEGYError('SEG-Y file incomplete (file=%s)' % filename)
                
                data = num.fromstring(datablock, dtype=dtype)
                tmax = None
            else:
                tmax = tmin + deltat_us_this/1000000.*(nsamples_this-1)
                data = None
            tr = trace.Trace(
                '', '%i' % (ensemble_num), '', '%i' %(trace_numbersegy),
                tmin=tmin,
                tmax=tmax,
                deltat=deltat_us_this/1000000.,
                ydata=data.astype(num.int32))

            nlsc=('','%i' %(ensemble_num),'','%i' %(trace_numbersegy))
            traces.append(tr)
            ipos += nbtrh+nsamples_this*sample_size

        self.traces = traces
        
    def get_traces(self):
        return self.traces
                           

def iload(filename, load_data):
    try:
        segyf = SEGYFile(filename, load_data=load_data)
        for tr in segyf.get_traces():
            yield tr

    except (OSError, SEGYError), e:
        raise FileLoadError(e)


