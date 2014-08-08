

import math
import calendar
import collections

import numpy as num
from pyrocko import trace, util

gps_fmt = {}
# gps altitude message
gps_fmt['AL'] = '''
    type 2 str
    time_of_day 5 float
    altitude 6 float
    vertical_velocity 4 float
    source_flag 1 int
    age_flag 1 int
'''

# gps position velocity message
gps_fmt['PV'] = '''
    type 2 str
    time_of_day 5 float
    latitude 8 lat_float
    longitude 9 lon_float
    speed_mph 3 float
    heading 3 float
    source_flag 1 int
    age_flag 1 int
'''

# gps status message
gps_fmt['ST'] = '''
    type 2 str
    tracking_status_code 2 hex_int
    nibble1 1 hex_int
    nibble2 1 hex_int
    machine_id 2 hex_int
    nibble3 1 hex_int
    nibble4 1 hex_int
    reserved 2 str
'''

# gps time date message
gps_fmt['TM'] = '''
    type 2 str
    hours 2 int
    minutes 2 int
    seconds 5 seconds_float
    day 2 int
    month 2 int
    year 4 int
    gps_utc_time_offset 2 int
    current_fix_source 1 int
    number_usable_svs 2 int
    gps_utc_offset_flag 1 int
    reserved 5 str
'''


def time_from_gps(tm):
    if not tm.gps_utc_offset_flag:
        # time is GPS time
        offset = tm.gps_utc_time_offset
    else:
        # time is UTC time
        offset = 0.

    return calendar.timegm((tm.year, tm.month, tm.day, tm.hours, tm.minutes,
                            tm.seconds)) + offset


def latlon_float(s):
    return int(s) / 100000.


def seconds_float(s):
    return int(s) / 1000.


def hex_int(s):
    return int(s, 16)

convert_functions = {
    'int': int,
    'float': float,
    'lat_float': latlon_float,
    'lon_float': latlon_float,
    'seconds_float': seconds_float,
    'str': str,
    'hex_int': hex_int}


class GPSFormat_:
    def __init__(self, name, fmt):

        nn = 0
        names = []
        items = []
        for line in fmt.strip().splitlines():
            toks = line.split()
            n = int(toks[1])
            items.append((nn, nn+n, convert_functions[toks[2]]))
            names.append(toks[0])
            nn += n

        self.items = items
        self.Message = collections.namedtuple('GPSMessage'+k, names)

    def unpack(self, s):
        return self.Message(*(converter(s[begin:end]) for
                            (begin, end, converter) in self.items))


GPSFormat = {}
for k in gps_fmt.keys():
    GPSFormat[k] = GPSFormat_(k, gps_fmt[k])


class EOF(Exception):
    pass


class CubeReaderError(Exception):
    pass


class CubeReader(object):

    header_keys = {
        str: 'GIPP_V DEV_NO E_NAME GPS_PO S_TIME S_DATE DAT_NO'.split(),
        int: '''P_AMPL CH_NUM S_RATE D_FILT C_MODE A_CHOP F_TIME GPS_TI GPS_OF
                A_FILT A_PHAS GPS_ON ACQ_ON V_TCXO D_VOLT E_VOLT'''.split()}

    all_header_keys = header_keys[str] + header_keys[int]

    def __init__(self, f, load_data=True):
        self._f = f
        self._maxbuf = 1000000
        self._buf = ''
        self._traces_complete = []
        self._traces = []
        self._at_eof = False
        self._load_data = load_data

    def read_to(self, sep):
        while True:
            data = self._f.read(1024)
            if len(data) != 1024:
                raise CubeReaderError('premature end of file')

            if len(self._buf) + len(data) > self._maxbuf:
                raise CubeReaderError('buffer limit exceeded')

            self._buf += data

            ipos = self._buf.find(sep)
            if ipos != -1:
                data = self._buf[:ipos]
                self._buf = self._buf[ipos:]
                return data

    def read(self, n=None):
        if n is None:
            out = self._buf + self._f.read()
            self._buf = ''
            return out

        if len(self._buf) == 0:
            return self._f.read(n)

        if len(self._buf) < n:
            data = self._f.read(n-len(self._buf))
            self._buf += data

        out = self._buf[:n]
        self._buf = self._buf[n:]
        return out

    def read_header(self):
        s = self.read_to(chr(128))
        s = s.replace(chr(240), '')
        s = s.replace(';', ' ')
        s = s.replace('=', ' ')
        kvs = s.split(' ')
        d = dict((k, v) for (k, v) in zip(kvs[0::2], kvs[1::2]))
        d2 = {}
        for t, ks in self.header_keys.iteritems():
            for k in ks:
                d2[k] = t(d.get(k, {int: -1, str: ''}[t]))

        self.nchannels = d2['CH_NUM']
        self.deltat = 1.0 / d2['S_RATE']
        self.recording_unit = d2['DEV_NO']

        self.firmware_version = d2['GIPP_V']
        self.tdelay = d2['D_FILT'] * self.deltat
        self._time_start0 = util.str_to_time(d2['S_DATE'] + d2['S_TIME'],
                                             format='%y/%m/%d%H:%M:%S')
        self._ipos = 0
        self._ipos_pps = None
        self._time_start1 = None
        self._toffsets = []
        self._igps = 0

        self._traces = [None] * self.nchannels

    def _init_trace(self, ic, tmin):
        if self._load_data:
            ydata = num.array([], dtype=num.int32)
            tmax = None
        else:
            ydata = None
            tmax = tmin - self.deltat

        return trace.Trace('', self.recording_unit, '', 'p%i' % ic,
                           tmin=tmin, tmax=tmax, deltat=self.deltat,
                           ydata=ydata)

    def read_data_samples(self, first_blocktype):
        nr = 4 * self.nchannels + 1
        nc = self.nchannels
        s = self.read(512 * nr)

        d = num.fromstring(s, dtype=num.uint8)
        next_ = d[nr-1::nr]
        try:
            # get next blocktype which is not 8 (sample without pps)
            # or 9 (with pps)
            nsamples = num.where((next_ >> 5) != 4)[0][0] + 1
        except IndexError:
            nsamples = len(s) / nr

        # get latest pps sample position (first is not included in next_)
        if first_blocktype == 9:
            self._ipos_pps = self._ipos

        try:
            self._ipos_pps = self._ipos + 1 + num.where(
                next_[:nsamples] == 9)[0][-1]
        except IndexError:
            pass

        self._buf = s[nsamples*nr-1:]

        if self._load_data:
            d = d[:nsamples*nr]
            d = d.reshape((d.size/nr, nr))
            d = d[:, :nr-1].reshape((nsamples*nc, 4)).astype(num.int32)
            values = d[:, 0] << 17
            values += d[:, 1] << 10
            values += d[:, 2] << 3
            values += d[:, 3]
            values -= (values & 2**23) << 1

        for ic in xrange(nc):
            if self._traces[ic] and self._traces[ic].data_len() > 1000000:
                self._traces_complete.append(self._traces[ic])
                self._traces[ic] = None

            if self._traces[ic] is None:
                if self._time_start1 is not None:
                    t = self._time_start1 + self._ipos * self.deltat
                else:
                    t = self._time_start0 + self._ipos * self.deltat

                self._traces[ic] = self._init_trace(ic, t)

            tr = self._traces[ic]

            if self._load_data:
                tr.append(values[ic::nc])
            else:
                tr.tmax = tr.tmin + (nsamples + tr.data_len() - 1) * self.deltat
                self._traces[ic]._update_ids()

        self._ipos += nsamples

    def read_gps_block(self):
        data = self.read(79)

        m = {}
        for line in data[:76].split('<'):
            if line:
                k = line[2:4]
                m[k] = GPSFormat[k].unpack(line[2:])

        b = data[76]
        if (ord(b) >> 4) != 11:
            raise CubeReaderError('no t shift after GPS string')

        b2 = data[77:]
        xshift = float(ord(b2[0])*128 + ord(b2[1])) * 2.44140625

        tm = m['TM']
        tgps = math.floor(time_from_gps(tm)) + xshift / 1000000.0 - self.tdelay

        self._igps += 1

        if self._igps >= 20 and self._time_start1 is None and \
                self._ipos_pps is not None:
            self._time_start1 = tgps - self.deltat * self._ipos_pps
            tshift = self._time_start1 - self._time_start0
            for tr in self._traces_complete + self._traces:
                if tr is not None:
                    tr.shift(tshift)

        if self._time_start1 is not None:
            toffset = tgps - (self._time_start1 + self.deltat * self._ipos_pps)
            self._toffsets.append(toffset)
            toffset = num.median(self._toffsets[-20:])
            if abs(toffset) > 0.6 * self.deltat:
                toffset = round(toffset / self.deltat) * self.deltat
                self._traces_complete.extend(self._traces)
                self._traces = [None] * self.nchannels
                self._time_start1 += toffset
                self._toffsets = []

    def read_end_block(self):
        self.read()
        raise EOF()

    def read_blocktype(self):
        s = self.read(1)
        if len(s) == 0:
            raise EOF()

        return ord(s) >> 4

    def read_next(self):
        t = self.read_blocktype()
        if t in (8, 9):
            self.read_data_samples(t)
        elif t == 10:
            self.read_gps_block()
        elif t == 14:
            self.read_end_block()
        elif t == 15:
            self.read_header()
        elif t == 12:  # empty ?
            pass
        elif t == 13:  # info block ascii ?
            pass
        elif t == 0:   # huh?
            pass
        else:
            raise CubeReaderError('unknown block type %i' % t)

    def __iter__(self):
        return self

    def next(self):
        while not self._traces_complete and not self._at_eof:
            try:
                self.read_next()
            except EOF:
                self._traces_complete.extend(self._traces)
                self._traces = None
                self._at_eof = True

        if self._traces_complete:
            return self._traces_complete.pop(0)
        else:
            if self._at_eof:
                raise StopIteration()


def iload(fn, load_data=True):
    with open(fn, 'r') as f:
        r = CubeReader(f, load_data=load_data)
        for tr in r:
            yield tr


def detect(first512):

    s = first512
    if len(s) < 512:
        return False

    if ord(s[0]) >> 4 != 15:
        return False

    n = s.find(chr(128))
    if n == -1:
        n = len(s)

    s = s[1:n]
    s = s.replace(chr(240), '')
    s = s.replace(';', ' ')
    s = s.replace('=', ' ')
    kvs = s.split(' ')

    if len([x for x in CubeReader.all_header_keys if x in kvs]) == 0:
        return False

    return True
