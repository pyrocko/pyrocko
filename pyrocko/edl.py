import serial, re, string, sys, struct, collections, time, logging
import numpy as num
from pyrocko import trace

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

class NotAquiring(Exception):
    pass

class ReadTimeout(Exception):
    pass

class Reader:

    def __init__(self, port=0, timeout=3., baudrate=115200):
        if isinstance(port, int):
            self._port = portnames()[port]
        else:
            self._port = str(port)

        self._timeout = float(timeout)
        self._baudrate = int(baudrate)
        self._serial = None
        self._buffer = ''

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
        mod, _ = self._read_block()
        mde, _ = self._read_block()
        dat, values_data = self._read_block()
        sum, _ = self._read_block()
        values = unpack_values(mod.ncomps, mod.bytes_per_sample, values_data)
        return (mod, mde, dat, sum, values)

    def stop(self):
        if not self.running():
            return

        self._serial.close()
        self._serial = None
        self._buffer = ''


class EDLHamster:
    def __init__(self, *args, **kwargs):
        self.reader = Reader(*args, **kwargs)
        self._i = 0

    def acquisition_start(self):
        self.reader.start()

    def acquisition_stop(self):
        self.reader.stop()

    def process(self):
        try:
            mod, mde, dat, sum, values = self.reader.read_record()

            traces = [
                trace.Trace('','Test','','X', deltat=0.005, tmin=float(self._i), ydata=values[::3]),
                trace.Trace('','Test','','Y', deltat=0.005, tmin=float(self._i), ydata=values[1::3]),
                trace.Trace('','Test','','Z', deltat=0.005, tmin=float(self._i), ydata=values[2::3])
            ]
            self._i += 1

            for tr in traces:
                self.got_trace(tr)

            return True

        except ReadTimeout:
            return False

    def got_trace(self, tr):
        logger.info('Got trace from EDL: %s' % tr)

