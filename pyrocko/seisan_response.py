import calendar
import util
import numpy as num

class SeisanResponseFileError(Exception):
    pass

def unpack_fixed(format, line, *callargs):
    ipos = 0
    values = []
    icall = 0
    for form in format.split(','):
        optional = form[-1] == '?'
        form = form.rstrip('?')
        typ = form[0]
        l = int(form[1:])
        s = line[ipos:ipos+l]
        cast = {'x': None, 'a': str, 'i': int, 'f': float, '@': 'extra'}[typ]
        if cast == 'extra':
            cast = callargs[icall]
            icall +=1
        
        if cast is not None:
            if optional and s.strip() == '':
                values.append(None)
            else:
                try:
                    values.append(cast(s))
                except:
                    raise SeisanResponseFileError('Invalid cast at position [%i:%i] of line: %s' % (ipos, ipos+1, line))
                
        ipos += l
    
    return values
        
class SeisanResponseFile:
    
    def __init__(self):
        pass
        
    def read(self, filename):
        
        f = open(filename, 'r')
        line = f.readline()
        station, component, century, deltayear, doy, month, day, hr, mi, sec = \
            unpack_fixed('a5,a4,@1,i2,x1,i3,x1,i2,x1,i2,x1,i2,x1,i2,x1,f6', line[0:35],
                lambda s: {' ': 1900, '0': 1900, '1': 2000}[s])
        
        latitude, longitude, elevation, filetype, cf_flag = \
            unpack_fixed('f8?,x1,f9?,x1,f5?,x2,@1,a1', line[50:80],
                lambda s: {' ': 'gains-and-filters', 't': 'tabulated', 'p': 'poles-and-zeros'}[s.lower()])
            
        line = f.readline()
        comment = line.strip()
        tmin = calendar.timegm( (century+deltayear, 1, doy, hr,mi, int(sec) ) ) + sec-int(sec)
        
        if filetype == 'gains-and-filters':
        
            line = f.readline()
            period, damping, sensor_sensitivity, amplifier_gain, digitizer_gain, \
            gain_1hz, filter1_corner, filter1_order, filter2_corner, filter2_order = \
                unpack_fixed('f8,f8,f8,f8,f8,f8,f8,f8,f8,f8', line)
            
            filter_defs = [ filter1_corner, filter1_order, filter2_corner, filter2_order ]
            line = f.readline()
            filter_defs.extend(unpack_fixed('f8,f8,f8,f8,f8,f8,f8,f8,f8,f8', line))
            
            filters = []
            for order, corner in zip(filter_defs[1::2], filter_defs[0::2]):
                if order != 0.0:
                    filters.append((order, corner))
            
        if filetype in ('gains-and-filters', 'tabulated'):
            data = ([],[],[])
            for iy in range(3):
                for ix in range(3):
                    line = f.readline()
                    data[ix].extend(unpack_fixed('f8,f8,f8,f8,f8,f8,f8,f8,f8,f8', line))
            response_table = num.array(data, dtype=num.float)
        
        f.close()
                
        self.station = station
        self.component = component
        self.tmin = tmin
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.filetype = filetype
        self.comment = comment
        self.period = period
        self.damping = damping
        self.sensor_sensitivity = sensor_sensitivity
        self.amplifier_gain = amplifier_gain
        self.digitizer_gain = digitizer_gain
        self.gain_1hz = gain_1hz
        self.filters = filters
        self.response_table = response_table
        
        
        
    def __str__(self):
        return '''Seisan Response File
station: %s
component: %s
start time: %s
latitude: %f
longitude: %f
elevation: %f
filetype: %s
comment: %s
filters: %s
response: %s
'''     % (self.station, self.component, util.gmctime(self.tmin), self.latitude, 
            self.longitude, self.elevation, self.filetype, self.comment, self.filters, self.response_table) 


