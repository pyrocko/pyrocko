import sys, os
import numpy as num
import util, trace
import struct
import calendar

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
            
            
    def clear(self):
        '''Empty file record.'''
        
        # set the required attributes        
        self.delta = 1.0
        self.npts = 0
        self.b = 0.0
        self.data = [ num.arange(0, dtype=num.int32) ]
        
    def read(self, filename, get_data=True):
        '''Read SAC file.
        
           filename -- Name of SEGY file.
           get_data -- If True, the data is read, otherwise only read headers.
        '''
        nbth = SEGYFile.nbytes_textual_header
        nbbh = SEGYFile.nbytes_binary_header
        nbthx = SEGYFile.nbytes_optional_textual_header
        nbtrh = SEGYFile.nbytes_trace_header
        
        # read in all data
        f = open(filename,'rb')
        
        # XXX should skip volume label
        
        filedata = f.read()
        f.close()
        
        hvals = struct.unpack('>24H', filedata[3212:3212+24*2])
        (ntraces, nauxtraces, deltat_us, deltat_us_orig, nsamples, 
         nsamples_orig, format) = hvals[0:7]
        
        (segy_revision, fixed_length_traces, nextended_headers) = struct.unpack('>3H', filedata[3500:3500+3*2])
        
        formats = { 1: (None,  4, "4-byte IBM floating-point"),
                    2: ('>i4', 4, "4-byte, two's complement integer"),
                    3: ('>i4', 2, "2-byte, two's complement integer"),
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
        traces = []
        for itrace in xrange(ntraces+nauxtraces):
            trace_header = filedata[ipos:ipos+nbtrh]
            if len(trace_header) != nbtrh:
                raise SEGYError('SEG-Y file incomplete (file=%s)' % filename)
            
            (trace_number,) = struct.unpack('>1I', trace_header[0:4])
            (nsamples_this, deltat_us_this) = struct.unpack('>2H', trace_header[114:114+2*2])
            (year,doy,hour,minute,second) = struct.unpack('>5H', trace_header[156:156+2*5])
            
            try:
                tmin = calendar.timegm((year,1,doy,hour,minute,second))
            except:
                raise SEGYError('Could not get starting date/time for trace %i in SEG-Y file %s.' % (itrace+1, filename))
            
            if fixed_length_traces:
                if (nsamples_this, deltat_us_this) != (nsamples, deltat_us):
                    raise SEGYError('Trace of incorrect length or sampling rate found in SEG-Y file (trace=%i, file=%s)' % (itrace+1, filename))
                
            if get_data:
                datablock = filedata[ipos+nbtrh:ipos+nbtrh+nsamples_this*sample_size]
                if len(datablock) != nsamples_this*sample_size:
                    raise SEGYError('SEG-Y file incomplete (file=%s)' % filename)
                
                data = num.fromstring(datablock, dtype=dtype)
                tmax = None
            else:
                tmax = tmin + deltat_us/1000000.*(nsamples_this-1)
                data = None
                
            tr = trace.Trace('','%i' % (itrace+1),'','', tmin=tmin, tmax=tmax, deltat=deltat_us/1000000., ydata=data)
            traces.append(tr)
            
            ipos += nbtrh+nsamples_this*sample_size
        
        self.traces = traces
        
    def get_traces(self):
        return self.traces
                           
if __name__ == '__main__':
    
    segy = SEGYFile(sys.argv[1])
    #print segy.to_trace()
        
