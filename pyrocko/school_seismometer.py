import trace

import serial, time, sys, logging
import numpy as num
from scipy import stats


logger = logging.getLogger('pyrocko.school_seismometer')

class SchoolSeismometerError(Exception):
    pass

class SchoolSeismometer:
    
    def __init__(self, port=0, baudrate=9600, timeout=5, buffersize=128,
                       network='', station='TEST', location='', channel='Z',
                       disallow_uneven_sampling_rates=True):
        
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.buffersize = buffersize
        self.ser = None
        self.values = []
        self.times = []
        self.deltat = None
        self.previous = None
        self.disallow_uneven_sampling_rates = disallow_uneven_sampling_rates
        self.network = network
        self.station = station
        self.location = location
        self.channel = channel
        
    def start(self):
        if self.ser is not None:
            self.stop()
            
        try:
            self.ser = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout)
           # self.ser = sys.stdin
            self.buffer = num.zeros(self.buffersize)
        
        except serial.serialutil.SerialException:
            logger.error('Cannot open serial port %i' % self.port)
            raise SchoolSeismometerError('Cannot open serial port %i' % self.port)
            
        self.run()
            
    def stop(self):
        if self.ser is not None:
            self.ser.close()
            self.ser = None
        
            
    def sun_is_shining(self):
        return True
        
    def run(self):
        while self.process():
            pass
        
        
    def process(self):
        if self.ser is None:
            return False
        
        line = self.ser.readline()
        t = time.time()
        
        for tok in line.split():
            try:
                val = float(tok)
            except:
                logger.error('Got something unexpected on serial line')
                continue
            
            self.values.append(val)
            self.times.append(t)
            
            if len(self.values) == self.buffersize:
                self._flush_buffer()
        
        return True
        
    def _flush_buffer(self):
        t = num.array(self.times, dtype=num.float)
        v = num.array(self.values, dtype=num.int) 

        r_deltat, r_tmin, r, tt, stderr = stats.linregress(num.arange(t.size, dtype=num.float), t)
        
        print r_tmin, r_deltat
        
        if self.previous:
            p_tmin, p_deltat, p_size = self.previous
            
            predicted_tmin = p_tmin + p_deltat * p_size   
            err_tmin = predicted_tmin - r_tmin
            print 'tmin error: %g' % err_tmin
            
            if self.deltat is not None:
                continuous_tmin = self.tmin + self.ncontinuous*self.deltat
                if abs(r_tmin - continuous_tmin) > 0.1 * self.deltat:
                    logger.warn('Resynchronizing time')
                self.deltat = None
            
            # detect sampling rate
            if self.deltat is None and err_tmin < p_deltat/10.:
                if self.disallow_uneven_sampling_rates:
                    self.deltat = 1./round(1./p_deltat)
                else:
                    self.deltat = p_deltat
                    
                self.tmin = p_tmin
                self.ncontinuous = p_size
        
        if self.deltat is not None:

            tr = trace.Trace(
                network=self.network, 
                station=self.station,
                location=self.location,
                channel=self.channel, 
                tmin=self.tmin + self.ncontinuous*self.deltat, deltat=self.deltat, ydata=v)
                
                
            self.got_trace(tr)
            self.ncontinuous += v.size
            
        self.previous = r_tmin, r_deltat, t.size

        self.values = []
        self.times = []
    
    def got_trace(self, tr):
        print tr
        logger.info('Completed trace from school seismometer: %s' % tr)
        
        
        
               