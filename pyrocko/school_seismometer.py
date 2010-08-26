import trace

import time, sys, logging
import numpy as num
from scipy import stats

logger = logging.getLogger('pyrocko.school_seismometer')

class Queue:
    def __init__(self, nmax):
        self.nmax = nmax
        self.queue = []
        
    def push_back(self, val):
        self.queue.append(val)
        while len(self.queue) > self.nmax:
            self.queue.pop(0)
        
    def pop_front(self):
        return self.pop(0)
        
    def mean(self):
        return sum(self.queue)/float(len(self.queue))
        
class SchoolSeismometerError(Exception):
    pass

class SchoolSeismometer:
    
    def __init__(self, port=0, baudrate=9600, timeout=5, buffersize=128,
                       network='', station='TEST', location='', channel='Z',
                       disallow_uneven_sampling_rates=True,
                       in_file=None):
        
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.buffersize = buffersize
        self.ser = None
        self.values = []
        self.times = []
        self.deltat = None
        self.tmin = None
        self.previous_deltats = Queue(nmax=10)
        self.previous_tmin_offsets = Queue(nmax=10)
        self.ncontinuous = 0
        self.disallow_uneven_sampling_rates = disallow_uneven_sampling_rates
        self.network = network
        self.station = station
        self.location = location
        self.channel = channel
        self.in_file = in_file    # for testing
        
    def start(self):
        if self.ser is not None:
            self.stop()
        
        if self.in_file is None:
            import serial
            try:
                self.ser = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=self.timeout)
        
            except serial.serialutil.SerialException:
                logger.error('Cannot open serial port %s' % self.port)
                raise SchoolSeismometerError('Cannot open serial port %s' % self.port)
        else:
            self.ser = self.in_file

        self.buffer = num.zeros(self.buffersize)

        self.run()
            
    def stop(self):
        if self.ser is not None:
            if self.in_file is None:
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
        
    def _regression(self,t):
        toff = t[0]
        t = t-toff
        i = num.arange(t.size, dtype=num.float)
        r_deltat, r_tmin, r, tt, stderr = stats.linregress(i, t)
        print 'XXX', r_deltat, r_tmin, r, tt, stderr
        for ii in range(2):
            t_fit = r_tmin+r_deltat*i
            quickones = num.where(t < t_fit)
            i = i[quickones]
            t = t[quickones]
            r_deltat, r_tmin, r, tt, stderr = stats.linregress(i, t)
            print 'YYY', r_deltat, r_tmin, r, tt, stderr
        
        return r_deltat, r_tmin+toff
        
    def _flush_buffer(self):
        t = num.array(self.times, dtype=num.float)
        v = num.array(self.values, dtype=num.int) 
        r_deltat, r_tmin = self._regression(t)
        if self.disallow_uneven_sampling_rates:
            r_deltat = 1./round(1./r_deltat)

        # check if deltat is consistent with expectations        
        self.previous_deltats.push_back(r_deltat)
        
        if self.deltat is not None:
            p_deltat = self.previous_deltats.mean()
            if self.disallow_uneven_sampling_rates and 1./p_deltat - 1./self.deltat > 0.5:
                self.deltat = None
            
            elif not self.disallow_uneven_sampling_rates and (self.deltat - p_deltat)/self.deltat > 0.01:
                self.deltat = None
                
        # check if onset has drifted / jumped
        if self.tmin is not None and self.tmin is not None:        
            continuous_tmin = self.tmin + self.ncontinuous*self.deltat
            
            tmin_offset = r_tmin - continuous_tmin
            self.previous_tmin_offsets.push_back(tmin_offset)
            toffset = self.previous_tmin_offsets.mean()
            if toffset > self.deltat*0.6:
                logger.warn('drift detected')
            
        # detect sampling rate
        if self.deltat is None:
            self.deltat = r_deltat

        if self.tmin is None:
            self.tmin = r_tmin
            

        tr = trace.Trace(
                network=self.network, 
                station=self.station,
                location=self.location,
                channel=self.channel, 
                tmin=self.tmin + self.ncontinuous*self.deltat, deltat=self.deltat, ydata=v)
                
                
        self.got_trace(tr)
        self.ncontinuous += v.size
        
        self.values = []
        self.times = []
    
    def got_trace(self, tr):
        logger.info('Completed trace from school seismometer: %s' % tr)
        
        
        
               