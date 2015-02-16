'''Module to read and write GSE2.0, GSE2.1, and IMS1.0 files.'''

import sys
import re

from pyrocko import util
from pyrocko.io_common import FileLoadError, FileSaveError
from pyrocko.guts import (
    Object, String, StringChoice, Timestamp, Int, Float, List, Bool, Complex,
    ValidationError)


g_versions = ('GSE2.0', 'GSE2.1', 'IMS1.0')
g_dialects = ('NOR_NDC', 'USA_DMC')


class SerializeError(Exception):
    pass


class DeserializeError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args)
        self._line_number = None
        self._line = None
        self._position = kwargs.get('position', None)
        self._format = kwargs.get('format', None)

    def set_context(self, line_number, line, version_dialect):
        self._line_number = line_number
        self._line = line
        self._version_dialect = version_dialect

    def __str__(self):
        l = [Exception.__str__(self)]
        if self._version_dialect is not None:
            l.append('format version: %s' % self._version_dialect[0])
            l.append('dialect: %s' % self._version_dialect[1])
        if self._line_number is not None:
            l.append('line number: %i' % self._line_number)
        if self._line is not None:
            l.append('line content:\n%s' % (self._line or
                                            '*** line is empty ***'))

        if self._position is not None:
            if self._position[1] is None:
                length = max(1, len(self._line or '') - self._position[0])
            else:
                length = self._position[1]

            l.append(' ' * self._position[0] + '^' * length)

        if self._format is not None:
            i = 0
            f = []
            j = 1
            for element in self._format:
                if element.length != 0:
                    f.append(' ' * (element.position - i))
                    if element.length is not None:
                        f.append(str(j % 10) * element.length)
                        i = element.position + element.length
                    else:
                        f.append(str(j % 10) + '...')
                        i = element.position + 4

                    j += 1

            l.append(''.join(f))

        return '\n'.join(l)


def float_or_none(x):
    if x.strip():
        return float(x)
    else:
        return None


def int_or_none(x):
    if x.strip():
        return int(x)
    else:
        return None


def float_to_string(fmt):
    ef = fmt[0]
    assert ef in 'ef'
    l, d = map(int, fmt[1:].split('.'))
    pfmts = ['%%%i.%i%s' % (l, dsub, ef) for dsub in range(d, -1, -1)]
    blank = ' ' * l

    def func(v):
        if v is None:
            return blank

        for pfmt in pfmts:
            s = pfmt % v
            if len(s) == l:
                return s

        raise SerializeError('format="%s", value=%s' % (pfmt, repr(v)))

    return func


def int_to_string(fmt):
    assert fmt[0] == 'i'
    pfmt = '%'+fmt[1:]+'i'
    l = int(fmt[1:])
    blank = ' ' * l

    def func(v):
        if v is None:
            return blank

        s = pfmt % v
        if len(s) == l:
            return s
        else:
            raise SerializeError('format="%s", value=%s' % (pfmt, repr(v)))

    return func


def deserialize_string(fmt):
    if fmt.endswith('?'):
        def func(s):
            if s.strip():
                return s.rstrip()
            else:
                return None
    else:
        def func(s):
            return s.rstrip()

    return func


def serialize_string(fmt):
    if fmt.endswith('+'):
        more_ok = True
    else:
        more_ok = False

    fmt = fmt.rstrip('?+')

    assert fmt[0] == 'a'
    l = int(fmt[1:])

    def func(v):
        if v is None:
            v = ''

        v = str(v)
        s = v.ljust(l)
        if more_ok or len(s) == l:
            return s
        else:
            raise SerializeError('max string length: %i, value="%s"' % l, v)

    return func


def rstrip_string(v):
    return v.rstrip()


def x_fixed(expect):
    def func():
        def parse(s):
            if s != expect:
                raise DeserializeError(
                    'expected="%s", value="%s"' % (expect, s))
            return s

        def string(s):
            return expect

        return parse, string

    func.width = len(expect)
    func.help_type = 'Keyword: %s' % expect
    return func


def x_substitute(value):
    def func():
        def parse(s):
            assert s == ''
            return value

        def string(s):
            return ''

        return parse, string

    func.width = 0
    func.help_type = 'Not present in this file version.'
    return func


def x_date_time(fmt='%Y/%m/%d %H:%M:%S.3FRAC'):
    def parse(s):
        try:
            return util.str_to_time(s, format=fmt)

        except:
            raise DeserializeError('expected date, value="%s"' % s)

    def string(s):
        return util.time_to_str(s, format=fmt)

    return parse, string

x_date_time.width = 23
x_date_time.help_type = 'YYYY/MM/DD HH:MM:SS.FFF'


def x_date():
    return x_date_time(fmt='%Y/%m/%d')

x_date.width = 10
x_date.help_type = 'YYYY/MM/DD'


def x_date_iris():
    return x_date_time(fmt='%m/%d/%Y')

x_date_iris.width = 10
x_date_iris.help_type = 'MM/DD/YYYY'


def x_date_time_no_seconds():
    return x_date_time(fmt='%Y/%m/%d %H:%M')

x_date_time_no_seconds.width = 16
x_date_time_no_seconds.help_type = 'YYYY/MM/DD HH:MM'


def x_yesno():
    def parse(s):
        if s == 'y':
            return True
        elif s == 'n':
            return False
        else:
            raise DeserializeError('"y" on "n" expected')

    def string(b):
        return 'ny'[int(b)]

    return parse, string

x_yesno.width = 1
x_yesno.help_type = 'yes/no'


def optional(x_func):

    def func():
        parse, string = x_func()

        def parse_optional(s):
            if s.strip():
                return parse(s)
            else:
                return None

        def string_optional(s):
            if s is None:
                return ' ' * x_func.width
            else:
                return string(s)

        return parse_optional, string_optional

    func.width = x_func.width
    func.help_type = 'optional %s' % x_func.help_type

    return func


class E(object):
    def __init__(self, begin, end, fmt, dummy=False):
        self.advance = 1
        if dummy:
            self.advance = 0

        self.position = begin - 1
        if end is not None:
            self.length = end - begin + 1
        else:
            self.length = None

        self.end = end

        if isinstance(fmt, basestring):
            t = fmt[0]
            if t in 'ef':
                self.parse = float_or_none
                self.string = float_to_string(fmt)
                l = int(fmt[1:].split('.')[0])
                self.help_type = 'float'
            elif t == 'a':
                self.parse = deserialize_string(fmt)
                self.string = serialize_string(fmt)
                l = int(fmt[1:].rstrip('+?'))
                self.help_type = 'string'
            elif t == 'i':
                self.parse = int_or_none
                self.string = int_to_string(fmt)
                l = int(fmt[1:])
                self.help_type = 'integer'
            else:
                assert False, 'invalid format: %s' % t

            assert self.length is None or l == self.length, \
                'inconsistent length for pos=%i, fmt=%s' \
                % (self.position, fmt)

        else:
            self.parse, self.string = fmt()
            self.help_type = fmt.help_type


def end_section(line, extra=None):
    if line is None:
        return True

    ul = line.upper()
    return ul.startswith('DATA_TYPE') or ul.startswith('STOP') or \
        (extra is not None and ul.startswith(extra))


class Section(Object):
    handlers = {}  # filled after section have been defined below

    @classmethod
    def read(cls, reader):
        datatype = DataType.read(reader)
        reader.pushback()
        return Section.handlers[datatype.type.upper()].read(reader)

    def write_datatype(self, writer):
        datatype = DataType(
            type=self.keyword,
            format=writer.version_dialect[0])
        datatype.write(writer)

    @classmethod
    def read_table(cls, reader, expected_header, block_cls, end=end_section):

        header = reader.readline()
        if not header.upper().startswith(expected_header.upper()):
            raise DeserializeError('invalid table header line, expected:\n'
                                   '%s' % expected_header)

        while True:
            line = reader.readline()
            reader.pushback()
            if end(line):
                break

            yield block_cls.read(reader)

    def write_table(self, writer, header, blocks):
        writer.writeline(header)
        for block in blocks:
            block.write(writer)


def get_versioned(x, version_dialect):
    if isinstance(x, dict):
        for v in (tuple(version_dialect), version_dialect[0], None):
            if v in x:
                return x[v]
    else:
        return x


class Block(Object):

    def values(self):
        return list(self.T.ivals(self))

    @classmethod
    def format(cls, version_dialect):
        return get_versioned(cls._format, version_dialect)

    def serialize(self, version_dialect):
        ivalue = 0
        out = []
        values = self.values()
        for element in self.format(version_dialect):
            if element.length != 0:
                out.append((element.position, element.string(values[ivalue])))
            ivalue += element.advance

        out.sort()

        i = 0
        slist = []
        for (position, s) in out:
            slist.append(' ' * (position - i))
            slist.append(s)
            i = position + len(s)

        return ''.join(slist)

    @classmethod
    def deserialize_values(cls, line, version_dialect):
        values = []
        for element in cls.format(version_dialect):
            try:
                val = element.parse(
                    line[element.position:element.end])

                if element.advance != 0:
                    values.append(val)
            except:
                raise DeserializeError(
                    'Cannot parse %s' % (
                        element.help_type),
                    position=(element.position, element.length),
                    version=version_dialect[0],
                    dialect=version_dialect[1],
                    format=cls.format(version_dialect))

        return values

    @classmethod
    def validated(cls, *args, **kwargs):
        obj = cls(*args, **kwargs)
        try:
            obj.validate()
        except ValidationError, e:
            raise DeserializeError(str(e))

        return obj

    @classmethod
    def regularized(cls, *args, **kwargs):
        obj = cls(*args, **kwargs)
        try:
            obj.regularize()
        except ValidationError, e:
            raise DeserializeError(str(e))

        return obj

    @classmethod
    def deserialize(cls, line, version_dialect):
        values = cls.deserialize_values(line, version_dialect)
        return cls.validated(**dict(zip(cls.T.propnames, values)))

    @classmethod
    def read(cls, reader):
        line = reader.readline()
        return cls.deserialize(line, reader.version_dialect)

    def write(self, writer):
        s = self.serialize(writer.version_dialect)
        writer.writeline(s)


class FreeFormatLine(Block):
    @classmethod
    def deserialize_values(cls, line, version_dialect):
        format = cls.format(version_dialect)
        values = line.split(None, len(format)-1)

        values_weeded = []
        for x, v in zip(format, values):
            if isinstance(x, basestring):
                if v.upper() != x:
                    raise DeserializeError(
                        'expected keyword: %s, found %s' % (x, v.upper()))

            else:
                if isinstance(x, tuple):
                    x, (parse, _) = x
                    v = parse(v)

                values_weeded.append((x, v))

        values_weeded.sort()
        return [xv[1] for xv in values_weeded]

    @classmethod
    def deserialize(cls, line, version_dialect):
        values = cls.deserialize_values(line, version_dialect)
        propnames = cls.T.propnames
        stuff = dict(zip(propnames, values))
        return cls.regularized(**stuff)

    def serialize(self, version_dialect):
        names = self.T.propnames
        props = self.T.properties
        out = []
        for x in self.format(version_dialect):
            if isinstance(x, basestring):
                out.append(x)
            else:
                if isinstance(x, tuple):
                    x, (_, string) = x
                    v = string(getattr(self, names[x-1]))
                else:
                    v = getattr(self, names[x-1])

                if v is None:
                    break

                out.append(props[x-1].to_save(v))

        return ' '.join(out)


class DataType(Block):
    type = String.T()
    subtype = String.T(optional=True)
    format = String.T()
    subformat = String.T(optional=True)

    @classmethod
    def deserialize(cls, line, version_dialect):
        pat = r'DATA_TYPE +([^ :]+)(:([^ :]+))? +([^ :]+)(:([^ :]+))?'
        m = re.match(pat, line)
        if not m:
            raise DeserializeError('invalid DATA_TYPE line')

        return cls.validated(
            type=m.group(1),
            subtype=m.group(3),
            format=m.group(4),
            subformat=m.group(6))

    def serialize(self, version_dialect):
        s = self.type
        if self.subtype:
            s += ':' + self.subtype

        f = self.format
        if self.subformat:
            f += ':' + self.subformat

        return 'DATA_TYPE %s %s' % (s, f)

    @classmethod
    def read(cls, reader):
        line = reader.readline()
        datatype = cls.deserialize(line, reader.version_dialect)
        reader.version_dialect[0] = datatype.format
        return datatype

    def write(self, writer):
        s = self.serialize(writer.version_dialect)
        writer.version_dialect[0] = self.format
        writer.writeline(s)


class FTPFile(FreeFormatLine):
    _format = ['FTP_FILE', 1, 2, 3, 4]

    net_address = String.T()
    login_mode = StringChoice.T(choices=('USER', 'GUEST'))
    directory = String.T()
    file = String.T()


class WaveformSubformat(StringChoice):
    choices = ['INT', 'CM6', 'CM8', 'AU6', 'AU8']


class WID2(Block):
    _format = [
        E(1, 4, x_fixed('WID2'), dummy=True),
        E(6, 28, x_date_time),
        E(30, 34, 'a5'),
        E(36, 38, 'a3'),
        E(40, 43, 'a4'),
        E(45, 47, 'a3'),
        E(49, 56, 'i8'),
        E(58, 68, 'f11.6'),
        E(70, 79, 'e10.2'),
        E(81, 87, 'f7.3'),
        E(89, 94, 'a6?'),
        E(96, 100, 'f5.1'),
        E(102, 105, 'f4.1')
    ]

    time = Timestamp.T()
    station = String.T()
    channel = String.T()
    aux_id = String.T(default='')
    sub_format = WaveformSubformat.T(default='CM6')
    samps = Int.T(default=0)
    samprate = Float.T(default=1.0)
    calib = Float.T(optional=True)
    calper = Float.T(optional=True)
    instype = String.T(optional=True)
    hang = Float.T(optional=True)
    vang = Float.T(optional=True)


class OUT2(Block):
    _format = [
        E(1, 4, x_fixed('OUT2'), dummy=True),
        E(6, 28, x_date_time),
        E(30, 34, 'a5'),
        E(36, 38, 'a3'),
        E(40, 43, 'a4'),
        E(45, 55, 'f11.3')
    ]

    time = Timestamp.T()
    station = String.T()
    channel = String.T()
    aux_id = String.T()
    duration = Float.T()


class DLY2(Block):
    _format = [
        E(1, 4, x_fixed('DLY2'), dummy=True),
        E(6, 28, x_date_time),
        E(30, 34, 'a5'),
        E(36, 38, 'a3'),
        E(40, 43, 'a4'),
        E(45, 55, 'f11.3')
    ]

    time = Timestamp.T()
    station = String.T()
    channel = String.T()
    aux_id = String.T()
    queuetim = Float.T()


class DAT2(Block):
    _format = [
        E(1, 4, x_fixed('DAT2'), dummy=True)
    ]

    raw_data = List.T(String.T())

    @classmethod
    def read(cls, reader):
        line = reader.readline()
        dat2 = cls.deserialize(line, reader.version_dialect)
        while True:
            line = reader.readline()
            if line.upper().startswith('CHK2 '):
                reader.pushback()
                break
            else:
                if reader._load_data:
                    dat2.raw_data.append(line.strip())

        return dat2


class STA2(Block):
    _format = [
        E(1, 4, x_fixed('STA2'), dummy=True),
        E(6, 14, 'a9'),
        E(16, 24, 'f9.5'),
        E(26, 35, 'f10.5'),
        E(37, 48, 'a12'),
        E(50, 54, 'f5.3'),
        E(56, 60, 'f5.3')
    ]

    network = String.T()
    lat = Float.T()
    lon = Float.T()
    coordsys = String.T(default='WGS-84')
    elev = Float.T()
    edepth = Float.T()


class CHK2(Block):
    _format = [
        E(1, 4, x_fixed('CHK2'), dummy=True),
        E(6, 13, 'i8')
    ]

    checksum = Int.T()


class EID2(Block):
    _format = [
        E(1, 4, x_fixed('EID2'), dummy=True),
        E(6, 13, 'a8'),
        E(15, 23, 'a9'),
    ]

    event_id = String.T()
    bull_type = String.T()


class BEA2(Block):
    _format = [
        E(1, 4, x_fixed('BEA2'), dummy=True),
        E(6, 17, 'a12'),
        E(19, 23, 'f5.1'),
        E(25, 29, 'f5.1')]

    beam_id = String.T()
    azimuth = Float.T()
    slowness = Float.T()


class Network(Block):
    _format = [
        E(1, 9, 'a9'),
        E(11, None, 'a64+')]

    network = String.T()
    description = String.T()


class Station(Block):
    _format = {
        None: [
            E(1, 9, 'a9'),
            E(11, 15, 'a5'),
            E(17, 20, 'a4'),
            E(22, 30, 'f9.5'),
            E(32, 41, 'f10.5'),
            E(43, 54, 'a12'),
            E(56, 60, 'f5.3'),
            E(62, 71, x_date),
            E(73, 82, optional(x_date))
        ],
        'GSE2.0': [
            E(0, -1, x_substitute('')),
            E(1, 5, 'a5'),
            E(7, 10, 'a4'),
            E(12, 20, 'f9.5'),
            E(22, 31, 'f10.5'),
            E(32, 31, x_substitute('WGS-84')),
            E(33, 39, 'f7.3'),
            E(41, 50, x_date),
            E(52, 61, optional(x_date))]}

    _format['IMS1.0', 'USA_DMC'] = list(_format[None])
    _format['IMS1.0', 'USA_DMC'][-2:] = [
        E(62, 71, x_date_iris),
        E(73, 82, optional(x_date_iris))]

    network = String.T()
    sta = String.T()
    statype = String.T()
    lat = Float.T()
    lon = Float.T()
    coordsys = String.T(default='WGS-84')
    elev = Float.T()
    ondate = Timestamp.T()
    offdate = Timestamp.T(optional=True)


class Channel(Block):
    _format = {
        None: [
            E(1, 9, 'a9'),
            E(11, 15, 'a5'),
            E(17, 19, 'a3'),
            E(21, 24, 'a4'),
            E(26, 34, 'f9.5'),
            E(36, 45, 'f10.5'),
            E(47, 58, 'a12'),
            E(60, 64, 'f5.3'),
            E(66, 70, 'f5.3'),
            E(72, 77, 'f6.1'),
            E(79, 83, 'f5.1'),
            E(85, 95, 'f11.6'),
            E(97, 102, 'a6'),
            E(105, 114, x_date),
            E(116, 125, optional(x_date))],
        'GSE2.0': [
            E(0, -1, x_substitute('')),
            E(1, 5, 'a5'),
            E(7, 9, 'a3'),
            E(11, 14, 'a4'),
            E(16, 24, 'f9.5'),
            E(26, 35, 'f10.5'),
            E(32, 31, x_substitute('WGS-84')),
            E(37, 43, 'f7.3'),
            E(45, 50, 'f6.3'),
            E(52, 57, 'f6.1'),
            E(59, 63, 'f5.1'),
            E(65, 75, 'f11.6'),
            E(77, 83, 'a7'),
            E(85, 94, x_date),
            E(96, 105, optional(x_date))]}

    # norsar plays its own game...
    _format['GSE2.0', 'NOR_NDC'] = list(_format['GSE2.0'])
    _format['GSE2.0', 'NOR_NDC'][-2:] = [
        E(85, 100, x_date_time_no_seconds),
        E(102, 117, optional(x_date_time_no_seconds))]

    # also iris plays its own game...
    _format['IMS1.0', 'USA_DMC'] = list(_format[None])
    _format['IMS1.0', 'USA_DMC'][-2:] = [
        E(105, 114, x_date_iris),
        E(116, 125, optional(x_date_iris))]

    network = String.T()
    sta = String.T()
    chan = String.T()
    aux_id = String.T()
    lat = Float.T(optional=True)
    lon = Float.T(optional=True)
    coordsys = String.T(default='WGS-84')
    elev = Float.T(optional=True)
    edepth = Float.T(optional=True)
    hang = Float.T(optional=True)
    vang = Float.T(optional=True)
    samprate = Float.T()
    inst = String.T()
    ondate = Timestamp.T()
    offdate = Timestamp.T(optional=True)


class BeamGroup(Block):
    _format = [
        E(1, 8, 'a8'),
        E(10, 14, 'a5'),
        E(16, 18, 'a3'),
        E(20, 23, 'a4'),
        E(25, 27, 'i3'),
        E(29, 37, 'f9.5')]

    bgroup = String.T()
    sta = String.T()
    chan = String.T()
    aux_id = String.T()
    wgt = Int.T(optional=True)
    delay = Float.T(optional=True)


class BeamType(StringChoice):
    choices = ['inc', 'coh']


class FilterType(StringChoice):
    choices = ['BP', 'LP', 'HP', 'BR']


class BeamParameters(Block):
    _format = [
        E(1, 12, 'a12'),
        E(14, 21, 'a8'),
        E(23, 25, 'a3'),
        E(27, 27, x_yesno),
        E(29, 33, 'f5.1'),
        E(35, 39, 'f5.3'),  # standard says f5.1 -999.0 is vertical beam
        E(41, 48, 'a8'),
        E(50, 55, 'f6.2'),
        E(57, 62, 'f6.2'),
        E(64, 65, 'i2'),
        E(67, 67, x_yesno),
        E(69, 70, 'a2'),
        E(72, 81, x_date),
        E(83, 92, optional(x_date))]

    beam_id = String.T()
    bgroup = String.T()
    type = BeamType.T()
    rot = Bool.T()
    azimuth = Float.T()
    slowness = Float.T()
    phase = String.T()
    flo = Float.T()
    fhi = Float.T()
    ford = Int.T()
    zero_phase = Bool.T()
    ftype = FilterType.T()
    ondate = Timestamp.T()
    offdate = Timestamp.T(optional=True)


class OutageReportPeriod(Block):
    _format = [
        E(1, 18, x_fixed('Report period from'), dummy=True),
        E(20, 42, x_date_time),
        E(44, 45, x_fixed('to'), dummy=True),
        E(47, 69, x_date_time)]

    start_time = Timestamp.T()
    end_time = Timestamp.T()


class Outage(Block):
    _format = [
        E(1, 9, 'a9'),
        E(11, 15, 'a5'),
        E(17, 19, 'a3'),
        E(21, 24, 'a4'),
        E(26, 48, x_date_time),
        E(50, 72, x_date_time),
        E(74, 83, 'f10.3'),
        E(85, None, 'a48+')]

    network = String.T()
    sta = String.T()
    chan = String.T()
    aux_id = String.T()
    start_time = Timestamp.T()
    end_time = Timestamp.T()
    duration = Float.T()
    comment = String.T()


class CAL2(Block):
    '''Calibration Identification Line'''

    _format = {
        None: [
            E(1, 4, x_fixed('CAL2'), dummy=True),
            E(6, 10, 'a5'),
            E(12, 14, 'a3'),
            E(16, 19, 'a4'),
            E(21, 26, 'a6'),
            E(28, 42, 'e15.8'),  # standard: e15.2
            E(44, 50, 'f7.3'),
            E(52, 62, 'f11.5'),
            E(64, 79, x_date_time_no_seconds),
            E(81, 96, optional(x_date_time_no_seconds))],
        'GSE2.0': [
            E(1, 4, x_fixed('CAL2'), dummy=True),
            E(6, 10, 'a5'),
            E(12, 14, 'a3'),
            E(16, 19, 'a4'),
            E(21, 26, 'a6'),
            E(28, 37, 'e10.4'),
            E(39, 45, 'f7.3'),
            E(47, 56, 'f10.5'),
            E(58, 73, x_date_time_no_seconds),
            E(75, 90, optional(x_date_time_no_seconds))]}

    sta = String.T(help='station code')
    chan = String.T(help='channel code')
    aux_id = String.T(help='location code (aux. identification code)')
    insttype = String.T(help='instrument type')
    calib = Float.T(
        help='system sensitivity (nm/count) at reference period (calper)')
    calper = Float.T(help='calibration reference period [s]')
    samprate = Float.T(help='system output sample rate [Hz]')
    ontime = Timestamp.T(help='effective start date and time')
    offtime = Timestamp.T(optional=True, help='effective end date and time')
    comments = List.T(String.T(optional=True))

    @classmethod
    def read(cls, reader):
        line = reader.readline()
        obj = cls.deserialize(line, reader.version_dialect)

        while True:
            line = reader.readline()
            if line is None:
                reader.pushback()
                break

            m = re.match(r' ?\((.*)\)\s*$', line)
            if m:
                obj.comments.append(m.group(1))

            else:
                reader.pushback()
                break

        return obj

    def write(self, writer):
        s = self.serialize(writer.version_dialect)
        writer.writeline(s)
        for c in self.comments:
            writer.writeline(' (%s)' % c)


class Units(StringChoice):
    choices = ['V', 'A', 'C']


class Stage(Block):
    snum = Int.T(help='stage sequence number')

    @classmethod
    def read(cls, reader):
        line = reader.readline()
        obj = cls.deserialize(line, reader.version_dialect)

        while True:
            line = reader.readline()
            if line is None:
                reader.pushback()
                break

            m = re.match(r' ?\((.*)\)\s*$', line)
            if m:
                obj.comments.append(m.group(1))
            elif line.startswith(' '):
                obj.append_dataline(line, reader.version_dialect)

            else:
                reader.pushback()
                break

        return obj

    def write(self, writer):
        line = self.serialize(writer.version_dialect)
        writer.writeline(line)
        self.write_datalines(writer)
        for c in self.comments:
            writer.writeline(' (%s)' % c)

    def write_datalines(self, writer):
        pass


class PAZ2Data(Block):
    _format = [
        E(2, 16, 'e15.8'),
        E(18, 32, 'e15.8')]

    real = Float.T()
    imag = Float.T()


class PAZ2(Stage):
    '''Poles and Zeros Section'''

    _format = {
        None: [
            E(1, 4, x_fixed('PAZ2'), dummy=True),
            E(6, 7, 'i2'),
            E(9, 9, 'a1'),
            E(11, 25, 'e15.8'),
            E(27, 30, 'i4'),
            E(32, 39, 'f8.3'),
            E(41, 43, 'i3'),
            E(45, 47, 'i3'),
            E(49, None, 'a25+')],
        ('IMS1.0', 'USA_DMC'): [
            E(1, 4, x_fixed('PAZ2'), dummy=True),
            E(6, 7, 'i2'),
            E(9, 9, 'a1'),
            E(11, 25, 'e15.8'),
            E(27, 30, 'i4'),
            E(32, 39, 'f8.3'),
            E(40, 42, 'i3'),
            E(44, 46, 'i3'),
            E(48, None, 'a25+')]}

    ounits = Units.T(help='output units code (V=volts, A=amps, C=counts)')
    sfactor = Float.T(help='scale factor')
    deci = Int.T(optional=True, help='decimation')
    corr = Float.T(optional=True, help='group correction applied [s]')
    npole = Int.T(help='number of poles')
    nzero = Int.T(help='number of zeros')
    descrip = String.T(help='description')

    poles = List.T(Complex.T())
    zeros = List.T(Complex.T())

    comments = List.T(String.T(optional=True))

    def append_dataline(self, line, version_dialect):
        d = PAZ2Data.deserialize(line, version_dialect)
        v = complex(d.real, d.imag)
        i = len(self.poles) + len(self.zeros)

        if i < self.npole:
            self.poles.append(v)
        elif i < self.npole + self.nzero:
            self.zeros.append(v)
        else:
            raise DeserializeError(
                'more poles and zeros than expected')

    def write_datalines(self, writer):
        for pole in self.poles:
            PAZ2Data(real=pole.real, imag=pole.imag).write(writer)
        for zero in self.zeros:
            PAZ2Data(real=zero.real, imag=zero.imag).write(writer)


class FAP2Data(Block):
    _format = [
        E(2, 11, 'f10.5'),
        E(13, 27, 'e15.8'),
        E(29, 32, 'i4')]

    freq = Float.T()
    amp = Float.T()
    phase = Float.T()


class FAP2(Stage):
    '''Frequency, Amplitude, and Phase Section'''

    _format = [
        E(1, 4, x_fixed('FAP2'), dummy=True),
        E(6, 7, 'i2'),
        E(9, 9, 'a1'),
        E(11, 14, 'i4'),
        E(16, 23, 'f8.3'),
        E(25, 27, 'i3'),
        E(29, 53, 'a25')]

    ounits = Units.T(help='output units code (V=volts, A=amps, C=counts)')
    deci = Int.T(optional=True, help='decimation')
    corr = Float.T(help='group correction applied [s]')
    ntrip = Int.T(help='number of frequency, amplitude, phase triplets')
    descrip = String.T(help='description')

    freqs = List.T(Float.T(), help='frequency [Hz]')
    amps = List.T(Float.T(), help='amplitude [input untits/output units]')
    phases = List.T(Float.T(), help='phase delay [degrees]')

    comments = List.T(String.T(optional=True))

    def append_dataline(self, line, version_dialect):
        d = FAP2Data.deserialize(line, version_dialect)
        self.freqs.append(d.freq)
        self.amps.append(d.amp)
        self.phases.append(d.phase)

    def write_datalines(self, writer):
        for freq, amp, phase in zip(self.freqs, self.amps, self.phases):
            FAP2Data(freq=freq, amp=amp, phase=phase).write(writer)


class GEN2Data(Block):
    _format = [
        E(2, 12, 'f11.5'),
        E(14, 19, 'f6.3')]

    cfreq = Float.T()
    slope = Float.T()


class GEN2(Stage):
    '''Generic Response Section'''

    _format = [
        E(1, 4, x_fixed('GEN2'), dummy=True),
        E(6, 7, 'i2'),
        E(9, 9, 'a1'),
        E(11, 25, 'e15.8'),
        E(27, 33, 'f7.3'),
        E(35, 38, 'i4'),
        E(40, 47, 'f8.3'),
        E(49, 51, 'i3'),
        E(53, 77, 'a25')]

    ounits = Units.T(help='output units code (V=volts, A=amps, C=counts)')
    calib = Float.T(
        help='system sensitivity (nm/count) at reference period (calper)')
    calper = Float.T(help='calibration reference period [s]')
    deci = Int.T(optional=True, help='decimation')
    corr = Float.T(help='group correction applied [s]')
    ncorner = Int.T('number of corners')
    descrip = String.T(help='description')

    cfreqs = List.T(Float.T(), help='corner frequencies [Hz]')
    slopes = List.T(Float.T(), help='slopes above corners [dB/decade]')

    comments = List.T(String.T(optional=True))

    def append_dataline(self, line, version_dialect):
        d = GEN2Data.deserialize(line, version_dialect)
        self.cfreqs.append(d.cfreq)
        self.slopes.append(d.slope)

    def write_datalines(self, writer):
        for cfreq, slope in zip(self.cfreqs, self.slopes):
            GEN2Data(cfreq=cfreq, slope=slope).write(writer)


class DIG2(Stage):
    '''Digitizer Response Section'''

    _format = [
        E(1, 4, x_fixed('DIG2'), dummy=True),
        E(6, 7, 'i2'),
        E(9, 23, 'e15.8'),
        E(25, 35, 'f11.5'),
        E(37, None, 'a25+')]

    sensitivity = Float.T('sensitivity [counts/input unit]')
    samprate = Float.T('digitizer sample rate [Hz]')
    descrip = String.T('description')

    comments = List.T(String.T(optional=True))


class SymmetryFlag(StringChoice):
    choices = ['A', 'B', 'C']


class FIR2Data(Block):
    _format = [
        E(2, 16, 'e15.8'),
        E(18, 32, 'e15.8'),
        E(34, 48, 'e15.8'),
        E(50, 64, 'e15.8'),
        E(66, 80, 'e15.8')]

    factors = List.T(Float.T())

    def values(self):
        return self.factors + [None]*(5-len(self.factors))

    @classmethod
    def deserialize(cls, line, version_dialect):
        factors = [v for v in cls.deserialize_values(line, version_dialect)
                   if v is not None]
        return cls.validated(factors=factors)


class FIR2(Stage):
    '''Finite Impulse Response Section'''

    _format = [
        E(1, 4, x_fixed('FIR2'), dummy=True),
        E(6, 7, 'i2'),
        E(9, 18, 'e10.2'),
        E(20, 23, 'i4'),
        E(25, 32, 'f8.3'),
        E(34, 34, 'a1'),
        E(36, 39, 'i4'),
        E(41, None, 'a25+')]

    gain = Float.T(help='filter gain (relative factor, not in dB)')
    deci = Int.T(optional=True, help='decimation')
    corr = Float.T(help='group correction applied [s]')
    symflag = SymmetryFlag.T(
        help='symmetry flag (A=asymmetric, B=symmetric (odd), '
             'C=symmetric (even))')
    nfactor = Int.T(help='number of factors')
    descrip = String.T('description')

    comments = List.T(String.T(optional=True))

    factors = List.T(Float.T())

    def append_dataline(self, line, version_dialect):
        d = FIR2Data.deserialize(line, version_dialect)
        self.factors.extend(d.factors)

    def write_datalines(self, writer):
        i = 0
        while i < len(self.factors):
            FIR2Data(factors=self.factors[i:i+5]).write(writer)
            i += 5


class Begin(FreeFormatLine):
    _format = ['BEGIN', 1]
    version = String.T()

    @classmethod
    def read(cls, reader):
        line = reader.readline()
        obj = cls.deserialize(line, reader.version_dialect)
        reader.version_dialect[0] = obj.version
        return obj

    def write(self, writer):
        FreeFormatLine.write(self, writer)
        writer.version_dialect[0] = self.version


class MessageType(StringChoice):
    choices = ['REQUEST', 'DATA', 'SUBSCRIPTION']


class MsgType(FreeFormatLine):
    _format = ['MSG_TYPE', 1]
    type = MessageType.T()


class MsgID(FreeFormatLine):
    _format = ['MSG_ID', 1, 2]
    msg_id_string = String.T()
    msg_id_source = String.T(optional=True)

    @classmethod
    def read(cls, reader):
        line = reader.readline()
        obj = cls.deserialize(line, reader.version_dialect)
        if obj.msg_id_source in g_dialects:
            reader.version_dialect[1] = obj.msg_id_source

        return obj

    def write(self, writer):
        FreeFormatLine.write(self, writer)
        if self.msg_id_source in g_dialects:
            writer.version_dialect[1] = self.msg_id_source


class RefID(FreeFormatLine):
    _format = {
        None: ['REF_ID', 1, 2, 'PART', 3, 'OF', 4],
        'GSE2.0': ['REF_ID', 1]}

    msg_id_string = String.T()
    msg_id_source = String.T(optional=True)
    sequence_number = Int.T(optional=True)
    total_number = Int.T(optional=True)

    def serialize(self, version_dialect):
        out = ['REF_ID', self.msg_id_string]
        if self.msg_id_source:
            out.append(self.msg_id_source)
            i = self.sequence_number
            n = self.total_number
            if i is not None and n is not None:
                out.extend(['PART', str(i), 'OF', str(n)])

        return ' '.join(out)


class LogSection(Section):
    keyword = 'LOG'
    lines = List.T(String.T())

    @classmethod
    def read(cls, reader):
        DataType.read(reader)
        lines = []
        while True:
            line = reader.readline()
            if end_section(line):
                reader.pushback()
                break
            else:
                lines.append(line)

        return cls(lines=lines)

    def write(self, writer):
        self.write_datatype(writer)
        for line in self.lines:
            ul = line.upper()
            if ul.startswith('DATA_TYPE') or ul.startswith('STOP'):
                line = ' ' + line

            writer.writeline(line)


class ErrorLogSection(LogSection):
    keyword = 'ERROR_LOG'


class FTPLogSection(Section):
    keyword = 'FTP_LOG'
    ftp_file = FTPFile.T()

    @classmethod
    def read(cls, reader):
        DataType.read(reader)
        ftp_file = FTPFile.read(reader)
        return cls(ftp_file=ftp_file)

    def write(self, writer):
        self.write_datatype(writer)
        self.ftp_file.write(writer)


class WID2Section(Section):

    wid2 = WID2.T()
    sta2 = STA2.T(optional=True)
    eid2s = List.T(EID2.T())
    bea2 = BEA2.T(optional=True)
    dat2 = DAT2.T()
    chk2 = CHK2.T()

    @classmethod
    def read(cls, reader):
        blocks = dict(eid2s=[])
        expect = [('WID2 ', WID2, 1)]

        if reader.version_dialect[0] == 'GSE2.0':
            # should not be there in GSE2.0, but BGR puts it there
            expect.append(('STA2 ', STA2, 0))
        else:
            expect.append(('STA2 ', STA2, 1))

        expect.extend([
            ('EID2 ', EID2, 0),
            ('BEA2 ', BEA2, 0),
            ('DAT2', DAT2, 1),
            ('CHK2 ', CHK2, 1)])

        for k, handler, required in expect:
            line = reader.readline()
            reader.pushback()

            if line is None:
                raise DeserializeError('incomplete waveform section')

            if line.upper().startswith(k):
                block = handler.read(reader)
                if k == 'EID2 ':
                    blocks['eid2s'].append(block)
                else:
                    blocks[k.lower().rstrip()] = block
            else:
                if required:
                    raise DeserializeError('expected %s block' % k)
                else:
                    continue

        return cls(**blocks)

    def write(self, writer):
        self.wid2.write(writer)
        if self.sta2:
            self.sta2.write(writer)
        for eid2 in self.eid2s:
            eid2.write(writer)
        if self.bea2:
            self.bea2.write(writer)

        self.dat2.write(writer)
        self.chk2.write(writer)

    def pyrocko_trace(self):
        from pyrocko import gse2_ext, trace

        raw_data = self.dat2.raw_data
        nsamples = self.wid2.samps
        deltat = 1.0 / self.wid2.samprate
        tmin = self.wid2.time
        if self.sta2:
            net = self.sta2.network
        else:
            net = ''
        sta = self.wid2.station
        loc = self.wid2.aux_id
        cha = self.wid2.channel

        if raw_data:
            ydata = gse2_ext.decode_m6(''.join(raw_data), nsamples)
            tmax = None
        else:
            tmax = tmin + (nsamples - 1) * deltat
            ydata = None

        return trace.Trace(
            net, sta, loc, cha, tmin=tmin, tmax=tmax,
            deltat=deltat,
            ydata=ydata)


class OUT2Section(Section):

    out2 = OUT2.T()
    sta2 = STA2.T()

    @classmethod
    def read(cls, reader):
        out2 = OUT2.read(reader)
        line = reader.readline()
        reader.pushback()
        if line.startswith('STA2'):
            # the spec sais STA2 is mandatory but in practice, it is not
            # always there...
            sta2 = STA2.read(reader)
        else:
            sta2 = None

        return cls(out2=out2, sta2=sta2)

    def write(self, writer):
        self.out2.write(writer)
        if self.sta2 is not None:
            self.sta2.write(writer)


class DLY2Section(Section):

    dly2 = DLY2.T()
    sta2 = STA2.T()

    @classmethod
    def read(cls, reader):
        dly2 = DLY2.read(reader)
        sta2 = STA2.read(reader)
        return cls(dly2=dly2, sta2=sta2)

    def write(self, writer):
        self.dly2.write(writer)
        self.sta2.write(writer)


class WaveformSection(Section):
    keyword = 'WAVEFORM'

    datatype = DataType.T()

    @classmethod
    def read(cls, reader):
        datatype = DataType.read(reader)
        return cls(datatype=datatype)

    def write(self, writer):
        self.datatype.write(writer)


class TableSection(Section):

    @classmethod
    def read(cls, reader):
        DataType.read(reader)
        ts = cls.table_setup

        header = get_versioned(ts['header'], reader.version_dialect)
        blocks = list(cls.read_table(reader, header, ts['cls']))
        return cls(**{ts['attribute']: blocks})

    def write(self, writer):
        self.write_datatype(writer)
        ts = self.table_setup
        header = get_versioned(ts['header'], writer.version_dialect)
        self.write_table(writer, header, getattr(self, ts['attribute']))


class NetworkSection(TableSection):
    keyword = 'NETWORK'
    table_setup = dict(
        header='Net       Description',
        attribute='networks',
        cls=Network)

    networks = List.T(Network.T())


class StationSection(TableSection):
    keyword = 'STATION'
    table_setup = dict(
        header={
            None: (
                'Net       Sta   Type  Latitude  Longitude Coord '
                'Sys     Elev   On Date   Off Date'),
            'GSE2.0': (
                'Sta   Type  Latitude  Longitude    Elev   On Date   '
                'Off Date')},
        attribute='stations',
        cls=Station)

    stations = List.T(Station.T())


class ChannelSection(TableSection):
    keyword = 'CHANNEL'
    table_setup = dict(
        header={
            None: (
                'Net       Sta  Chan Aux   Latitude Longitude  Coord Sys'
                '       Elev   Depth   Hang  Vang Sample Rate Inst      '
                'On Date    Off Date'),
            'GSE2.0': (
                'Sta  Chan Aux   Latitude  Longitude    '
                'Elev  Depth   Hang  Vang Sample_Rate Inst       '
                'On Date   Off Date')},
        attribute='channels',
        cls=Channel)

    channels = List.T(Channel.T())


class BeamSection(Section):
    keyword = 'BEAM'
    beam_group_header = 'Bgroup   Sta  Chan Aux  Wgt     Delay'
    beam_parameters_header = 'BeamID       Bgroup Btype R  Azim  Slow '\
                             'Phase       Flo    Fhi  O Z F    '\
                             'On Date    Off Date'
    group = List.T(BeamGroup.T())
    parameters = List.T(BeamParameters.T())

    @classmethod
    def read(cls, reader):
        DataType.read(reader)
        end = lambda line: line.upper().startswith('BEAMID')
        group = list(cls.read_table(reader, cls.beam_group_header, BeamGroup,
                                    end))

        parameters = list(cls.read_table(reader, cls.beam_parameters_header,
                                         BeamParameters))

        return cls(group=group, parameters=parameters)

    def write(self, writer):
        self.write_datatype(writer)
        self.write_table(writer, self.beam_group_header, self.group)
        writer.writeline('')
        self.write_table(writer, self.beam_parameters_header, self.parameters)


class CAL2Section(Section):
    cal2 = CAL2.T()
    stages = List.T(Stage.T())

    @classmethod
    def read(cls, reader):
        cal2 = CAL2.read(reader)
        stages = []
        handlers = {
            'PAZ2': PAZ2,
            'FAP2': FAP2,
            'GEN2': GEN2,
            'DIG2': DIG2,
            'FIR2': FIR2}

        while True:
            line = reader.readline()
            reader.pushback()
            if end_section(line, 'CAL2'):
                break

            k = line[:4].upper()
            if k in handlers:
                stages.append(handlers[k].read(reader))
            else:
                raise DeserializeError('unexpected line')

        return cls(cal2=cal2, stages=stages)

    def write(self, writer):
        self.cal2.write(writer)
        for stage in self.stages:
            stage.write(writer)


class ResponseSection(Section):
    keyword = 'RESPONSE'

    datatype = DataType.T()

    @classmethod
    def read(cls, reader):
        datatype = DataType.read(reader)
        return cls(datatype=datatype)

    def write(self, writer):
        self.datatype.write(writer)


class OutageSection(Section):
    keyword = 'OUTAGE'
    outages_header = 'NET       Sta  Chan Aux      Start Date Time'\
                     '          End Date Time        Duration Comment'
    report_period = OutageReportPeriod.T()
    outages = List.T(Outage.T())

    @classmethod
    def read(cls, reader):
        DataType.read(reader)
        report_period = OutageReportPeriod.read(reader)
        outages = []
        outages = list(cls.read_table(reader, cls.outages_header,
                                      Outage))

        return cls(
            report_period=report_period,
            outages=outages)

    def write(self, writer):
        self.write_datatype(writer)
        self.report_period.write(writer)
        self.write_table(writer, self.outages_header, self.outages)

for sec in (
        LogSection, ErrorLogSection, FTPLogSection, WaveformSection,
        NetworkSection, StationSection, ChannelSection, BeamSection,
        ResponseSection, OutageSection):

    Section.handlers[sec.keyword] = sec


class MessageHeader(Section):
    version = String.T()
    type = String.T()
    msg_id = MsgID.T(optional=True)
    ref_id = RefID.T(optional=True)

    @classmethod
    def read(cls, reader):
        handlers = {
            'BEGIN': Begin,
            'MSG_TYPE': MsgType,
            'MSG_ID': MsgID,
            'REF_ID': RefID}

        blocks = {}
        while True:
            line = reader.readline()
            reader.pushback()
            ok = False
            for k in handlers:
                if line.upper().startswith(k):
                    blocks[k] = handlers[k].read(reader)
                    ok = True

            if not ok:
                break

        return MessageHeader(
            type=blocks['MSG_TYPE'].type,
            version=blocks['BEGIN'].version,
            msg_id=blocks['MSG_ID'],
            ref_id=blocks['REF_ID'])

    def write(self, writer):
        Begin(version=self.version).write(writer)
        MsgType(type=self.type).write(writer)
        self.msg_id.write(writer)
        self.ref_id.write(writer)


def parse_ff_date_time(s):
    toks = s.split()
    if len(toks) == 2:
        sdate, stime = toks
    else:
        sdate, stime = toks[0], ''

    stime += '00:00:00.000'[len(stime):]
    util.str_to_time(sdate + ' ' + stime, format='%Y/%m/%d %H:%M:%S.3FRAC')


def string_ff_date_time(t):
    return util.time_to_str(t, format='%Y/%m/%d %H:%M:%S.3FRAC')


class TimeStamp(FreeFormatLine):
    _format = ['TIME_STAMP', 1]

    value = Timestamp.T()

    @classmethod
    def deserialize(cls, line, version_dialect):
        (s,) = cls.deserialize_values(line, version_dialect)
        return cls(value=parse_ff_date_time(s))

    def serialize(self, line, version_dialect):
        return 'TIME_STAMP %s' % string_ff_date_time(self.value)


class Stop(FreeFormatLine):
    _format = ['STOP']

    dummy = String.T(optional=True)


class XW01(FreeFormatLine):
    _format = ['XW01']

    dummy = String.T(optional=True)


class Reader(object):
    def __init__(self, f, load_data=True, version=None, dialect=None):
        self._f = f
        self._load_data = load_data
        self._current_fpos = None
        self._current_lpos = None  # "physical" line number
        self._current_line = None
        self._readline_count = 0
        self._pushed_back = False
        self._handlers = {
            'DATA_TYPE ': Section,
            'WID2 ': WID2Section,
            'OUT2 ': OUT2Section,
            'DLY2 ': DLY2Section,
            'CAL2 ': CAL2Section,
            'BEGIN': MessageHeader,
            'STOP': Stop,
            'XW01': XW01,  # for compatibility with BGR dialect
        }
        self._comment_lines = []
        self._time_stamps = []
        self.version_dialect = [version, dialect]  # main version, dialect
        self._in_garbage = True

    def tell(self):
        return self._current_fpos

    def readline(self):
        if self._pushed_back:
            self._pushed_back = False
            return self._current_line

        while True:
            self._current_fpos = self._f.tell()
            self._current_lpos = self._readline_count + 1
            l = self._f.readline()
            self._readline_count += 1
            if not l:
                self._current_line = None
                return None

            lines = [l.rstrip('\n\r')]
            while lines[-1].endswith('\\'):
                lines[-1] = lines[-1][:-1]
                l = self._f.readline()
                self._readline_count += 1
                lines.append(l.rstrip('\n\r'))

            self._current_line = ''.join(lines)
            if not self._current_line.strip():
                pass
            elif self._current_line.startswith('%'):
                self._comment_lines.append(
                    (self._current_lpos, self._current_line))
            elif self._current_line[:10].upper() == 'TIME_STAMP':
                self._time_stamps.append(
                    TimeStamp.deserialize(
                        self._current_line, self.version_dialect))
            else:
                return self._current_line

    def pushback(self):
        assert not self._pushed_back
        self._pushed_back = True

    def __iter__(self):
        return self

    def next(self):
        try:
            while True:
                line = self.readline()
                if line is None:
                    raise StopIteration()

                for k in self._handlers:
                    if line.upper().startswith(k):
                        self.pushback()
                        sec = self._handlers[k].read(self)
                        if isinstance(sec, Stop):
                            self._in_garbage = True
                        else:
                            self._in_garbage = False
                        return sec

                if not self._in_garbage:
                    raise DeserializeError('unexpected line')

        except DeserializeError, e:
            e.set_context(
                self._current_lpos,
                self._current_line,
                self.version_dialect)
            raise


class Writer(object):

    def __init__(self, f):
        self._f = f
        self.version_dialect = ['GSE2.1', None]

    def write(self, section):
        section.write(self)

    def writeline(self, line):
        self._f.write(line.rstrip())
        self._f.write('\n')


def write_string(sections):
    from cStringIO import StringIO
    f = StringIO()
    w = Writer(f)
    for section in sections:
        w.write(section)

    return f.getvalue()


def iload_fh(f):
    '''Load IMS/GSE2 records from open file handle.'''
    try:
        r = Reader(f)
        for section in r:
            yield section

    except DeserializeError, e:
        raise FileLoadError(e)


def iload_string(s):
    from cStringIO import StringIO
    f = StringIO(s)
    return iload_fh(f)


iload_filename, iload_dirname, iload_glob, iload = util.make_iload_family(
    iload_fh, 'IMS/GSE2', ':py:class:`Section`')


def dump_fh(sections, f):
    '''Dump IMS/GSE2 records to open file handle.'''
    try:
        w = Writer(f)
        for section in sections:
            w.write(section)

    except SerializeError, e:
        raise FileSaveError(e)


def dump_string(sections):
    from cStringIO import StringIO
    f = StringIO()
    dump_fh(sections, f)
    return f.getvalue()


if __name__ == '__main__':
    from optparse import OptionParser

    usage = 'python -m pyrocko.ims'

    description = '''
    Read and print IMS/GSE2 records.
    '''

    parser = OptionParser(
        usage=usage,
        description=description,
        formatter=util.BetterHelpFormatter())

    parser.add_option(
        '--version',
        dest='version',
        choices=g_versions,
        help='inial guess for version')

    parser.add_option(
        '--dialect',
        dest='dialect',
        choices=g_dialects,
        help='inial guess for dialect')

    (options, args) = parser.parse_args(sys.argv[1:])

    for fn in args:
        with open(fn, 'r') as f:
            r = Reader(f, load_data=False,
                       version=options.version, dialect=options.dialect)
            for sec in r:
                print sec
