import serial, re, string, sys, struct, collections, time, logging
import numpy as num
from pyrocko import trace, util
from scipy import stats

logger = logging.getLogger('pyrocko.edl')

MINBLOCKSIZE = 192

block_def = {}

block_def['MOD\0'] = '''
        identifier 4s
        size_of_mod I
        device_id 12s
        version 6s
        space1 1s
        serial_no 4s
        space2 1s
        test_pattern 8s
        block_count I
        ncomps H
        sample_rate H
        bytes_per_sample H
        filter_type 2s
        decimation3 H
        decimation4 H
        plldata h
        gain_g 1s
        gain B
        gain_component1 I
        gain_component2 I
        gain_component3 I
        offset_component1 I
        offset_component2 I
        offset_component3 I
        supply_voltage H
        supply_current H
        temp_sensor_voltage H
        supply_voltage_remote H
        user_input1 H
        user_input2 H
        user_input3 H
        not_used H
        coupling 2s
        reserved1 I
        reserved2 I
        reserved3 I
        gps_status_flags H
        gps_message_block_count I
        gps_message 72s
'''.split()

block_def['MDE\0'] = '''
        identifier 4s
        size_of_mde I
        serial_no 8s
        decimation5 H
        decimation6 H
        decimation7 H
        gain_component4 I
        gain_component5 I
        gain_component6 I
        offset_component4 I
        offset_component5 I
        offset_component6 I
        temperature1 H
        temperature2 H
        temperature3 H
        temperature4 H
        temperature5 H
        temperature6 H
        gps_message 129s
        pad 1s
'''.split()

block_def['SUM\0'] = '''
        identifier 4s
        size_of_sum I
        reserved H
        checksum_lo B
        checksum_hi B
'''.split()

block_def['DAT\0'] = '''
        identifier 4s
        size_of_dat I
'''.split()


Blocks = {}
for k in block_def.keys():
    fmt = '<'+''.join(block_def[k][1::2])
    fmt_len = struct.calcsize(fmt)
    Block = collections.namedtuple('Block'+k[:3], block_def[k][::2])
    Blocks[k] = (fmt, fmt_len, Block)

def portnames():
    try:
        # needs serial >= v2.6
        from serial.tools.list_ports import comports
        names = sorted( x[0] for x in comports() )

    except:
        # may only work on some linuxes
        from glob import glob
        names = sorted( glob('/dev/ttyS*') + glob('/dev/ttyUSB*') )

    return names

def unpack_block(data):
    block_type = data[:4]

    if block_type not in Blocks:
        return None

    fmt, fmt_len, Block = Blocks[block_type]

    if len(data) < fmt_len:
        raise EDLError('block size too short')

    return Block(*struct.unpack(fmt, data[:fmt_len])), data[fmt_len:]

def unpack_values(ncomps, bytes_per_sample, data):
    if bytes_per_sample == 4:
        return num.fromstring(data, dtype=num.dtype('<i4'))

    elif bytes_per_sample == 3:
        b1 = num.fromstring(data, dtype=num.dtype('<i1'))
        b4 = num.zeros(len(data)/4, dtype=num.dtype('<i4'))
        b4.view(dtype='<i2')[::2] = b1.view(dtype='<i2')
        b4.view(dtype='<i1')[2::4] = b1[i::3]
        return b4.astype(num.int32)

    else:
        raise


class TimeEstimator:
    def __init__(self, nlookback):
        self._nlookback = nlookback
        self._queue = []
        self._t0 = None
        self._n = 0
        self._deltat = None
        
    def insert(self, deltat, nadd, t):
        if self._deltat is None or self._deltat != deltat:
            self.reset()
            self._deltat = deltat

        if self._t0 is None:
            self._t0 = int(round(t/self._deltat))*self._deltat
        

        self._queue.append((self._n, t))
        self._n += nadd
        while len(self._queue) > self._nlookback:
            self._queue.pop()

        ns, ts = num.array(self._queue, dtype=num.float).T

        tpredicts = self._t0 + ns * self._deltat

        terrors = ts - tpredicts
        mterror = num.median(terrors)
        if num.abs(mterror) > 0.7*deltat and len(self._queue) == self._nlookback:
            deltat, t0, r, tt, stderr = stats.linregress(ns, ts)
            self._queue[:] = []
            self._t0 = t0
        
        return self._t0 + (self._n-nadd)*self._deltat
            
    def reset(self):
        self._queue[:] = []
        self._t0 = None

    def __len__(self):
        return len(self._queue)

class Record:
    def __init__(self, approx_time, mod, mde, dat, sum, values):
        self.approx_time = approx_time
        self.mod = mod
        self.mde = mde
        self.dat = dat
        self.sum = sum
        self.values = values

    def get_time(self):
        return self.approx_time

    def get_traces(self):
        traces = []
        for i in range(self.mod.ncomps):
            tr = trace.Trace('', 'ed', '', 'p%i' % i, 
                    deltat=num.float(self.mod.ncomps)/self.mod.sample_rate, tmin=self.get_time(), ydata=self.values[i::3])
            traces.append(tr)

        return traces

class NotAquiring(Exception):
    pass

class ReadTimeout(Exception):
    pass

class Reader:

    def __init__(self, port=0, timeout=3., baudrate=115200, lookback=10):
        if isinstance(port, int):
            self._port = portnames()[port]
        else:
            self._port = str(port)

        self._timeout = float(timeout)
        self._baudrate = int(baudrate)
        self._serial = None
        self._buffer = ''
        self._irecord = 0
        
        self._time_estimator = TimeEstimator(lookback)

    def running(self):
        return self._serial is not None

    def assert_running(self):
        if not self.running():
            raise NotAquiring()

    def start(self):
        self.stop()

        self._serial = serial.Serial(port=self._port, baudrate=self._baudrate, timeout=self._timeout)

        self._sync_on_mod()
    
    def _sync_on_mod(self):
        self._fill_buffer(MINBLOCKSIZE)

        while self._buffer[:4] != 'MOD\0':
            imark = self._buffer.find('MOD\0')
            if imark != -1:
                self._buffer = self._buffer[imark:]
            else:
                self._buffer = self._buffer[-4:]

            self._fill_buffer(MINBLOCKSIZE) 

    def _fill_buffer(self, minlen):
        if len(self._buffer) >= minlen:
            return

        nread = minlen-len(self._buffer)
        data = self._serial.read(nread)
        if len(data) != nread:
            self.stop()
            raise ReadTimeout()
        self._buffer += data

    def _read_block(self):
        self.assert_running()
        self._fill_buffer(8)
        block_type, block_len = struct.unpack('<4sI', self._buffer[:8])
        block_len += 8
        self._fill_buffer(block_len)
        block_data = self._buffer
        self._buffer = ''
        return unpack_block(block_data)

    def read_record(self):
        self._irecord += 1
        mod, _ = self._read_block()
        measured_system_time = time.time()
        mde, _ = self._read_block()
        dat, values_data = self._read_block()
        sum, _ = self._read_block()
        values = unpack_values(mod.ncomps, mod.bytes_per_sample, values_data)
        deltat = 1./mod.sample_rate * mod.ncomps
        approx_time = self._time_estimator.insert(deltat, values.size/mod.ncomps, measured_system_time)
        r = Record(approx_time, mod, mde, dat, sum, values)
        return r

    def stop(self):
        if not self.running():
            return

        self._serial.close()
        self._serial = None
        self._buffer = ''


class EDLHamster:
    def __init__(self, *args, **kwargs):
        self.reader = Reader(*args, **kwargs)

    def acquisition_start(self):
        self.reader.start()

    def acquisition_stop(self):
        self.reader.stop()

    def process(self):
        try:
            rec = self.reader.read_record()
            for tr in rec.get_traces():
                self.got_trace(tr)

            return True

        except ReadTimeout:
            return False

    def got_trace(self, tr):
        logger.info('Got trace from EDL: %s' % tr)

