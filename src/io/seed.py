from __future__ import print_function, absolute_import

import sys
import calendar
import logging

from . import util

logger = logging.getLogger('pyrocko.seed')


class NotABlockette(Exception):
    pass


class UnknownBlocketteType(Exception):
    pass


class Field(object):
    def __init__(self, name, v=None):
        self.name = name
        self.repeats = False
        if v is not None:
            self.min_version, self.max_version = v
        else:
            self.min_version, self.max_version = None, None

    def format(self, value):
        return str(value)


class A(Field):
    def __init__(self, name, length, v=None):
        Field.__init__(self, name, v)
        self.length = length

    def extract(self, data, pos):
        return self.interprete(data[pos:pos+self.length]), self.length

    def interprete(self, string):
        return string


class D(A):
    def __init__(self, name, length, mask, v=None):
        A.__init__(self, name, length, v)
        self.length = length
        self.mask = mask

    def interprete(self, string):
        return float(string)


class DInt(D):
    def __init__(self, name, length, mask, v=None):
        D.__init__(self, name, length, mask, v=v)

    def interprete(self, string):
        return int(string)


class DPow2(DInt):
    def interprete(self, string):
        return 2**int(string)


class V(Field):
    def __init__(self, name, min_length=None, max_length=None, v=None):
        Field.__init__(self, name, v)
        self.min_length = min_length
        self.max_length = max_length

    def extract(self, data, pos):
        end = data.index('~', pos)
        return self.interprete(data[pos:end]), end-pos + 1

    def interprete(self, string):
        return string


class VTime(V):
    def interprete(self, string):
        if len(string) == 0:
            return None

        toks = string.split('.')
        toks2 = toks[0].replace(':', ',').split(',')
        tt = [toks2[0], 1] + toks2[1:]
        tt = tt + [0] * (6-len(tt))
        t = calendar.timegm(tuple([int(x) for x in tt]))
        if len(toks) == 2:
            t += float('.'+toks[1])

        return t

    def format(self, value):
        return util.gmctime(value)


class Repeat(object):
    def __init__(self, repetitions_key, fields):
        self.repetitions_key = repetitions_key
        self.fields = fields
        for field in self.fields:
            field.repeats = True


class Definition(object):
    def __init__(self, btype, name, control_header, fields):
        self.btype = btype
        self.name = name
        self.control_header = control_header
        self.fields = [
            DInt('Blockette type',            3, '###'),
            DInt('Length of blockette',       4, '####')] + fields

    def get_fields(self):
        return list(self.fields)


blockette_definitions = [

    Definition(10, 'Volume Identifier Blockette', 'V', [
        D('Version of format',            4, '##.#'),
        DPow2('Logical record length',     2, '##'),
        VTime('Beginning time',        1, 22),
        VTime('End time',              1, 22),
        VTime('Volume Time',           1, 22, v=('V2.3', None)),
        V('Originating Organization',  1, 80, v=('V2.3', None)),
        V('Label',                     1, 80, v=('V2.3', None))
    ]),

    Definition(11, 'Volume Station Header Index Blockette', 'V', [
        DInt('Number of stations',                3, '###'),
        Repeat('Number of stations', [
            A('Station code',                         5),
            DInt('Sequence number of station header', 6, '######'),
        ]),
    ]),

    Definition(30, 'Data Format Dictionary Blockette', 'A', [
        V('Short descriptive name',               1, 50),
        DInt('Data format identifier code',       4, '####'),
        DInt('Data family type',                  3, '###'),
        DInt('Number of decoder keys',            2, '##'),
        Repeat('Number of decoder keys', [
            V('Decoder keys'),
        ]),
    ]),

    Definition(33, 'Generic Abbreviation Blockette', 'A', [
        DInt('Abbreviation lookup code',          3, '###'),
        V('Abbreviation description',             1, 50),
    ]),

    Definition(34, 'Units Abbreviations Blockette', 'A', [
        DInt('Unit lookup code',                  3, '###'),
        V('Unit name',                            1, 20),
        V('Unit description',                     0, 50),
    ]),

    Definition(50, 'Channel Identifier Blockette', 'S', [
        A('Station code',                         5),
        D('Latitude',                            10, '-##.######'),
        D('Longitude',                           11, '-###.######'),
        D('Elevation',                            7, '-####.#'),
        DInt('Number of channels',                4, '####'),
        DInt('Number of station comments',        3, '###'),
        V('Site name',                            1, 60),
        DInt('Network identifier code',           3, '###'),
        DInt('Bit word order 32',                 4, '####'),
        DInt('Bit word order 16',                 2, '##'),
        VTime('Start effective date',             1, 22),
        VTime('End effective date',               0, 22),
        A('Update flag',                          1),
        A('Network code',                         2),
    ]),
]

blockette_definitions_dict = dict(
    [(v.btype, v) for v in blockette_definitions])


def blockette_definition(key):
    if key not in blockette_definitions_dict:
        raise UnknownBlocketteType(key)
    else:
        return blockette_definitions_dict[key]


def ident(string):
    return string.lower().replace(' ', '_')


class BlocketteContent(object):
    def set(self, k, v):
        setattr(self, k, v)

    def add(self, k, v):
        if not hasattr(self, k):
            setattr(self, k, [])

        getattr(self, k).append(v)


class Blockette(object):

    def __init__(self, btype):
        self.btype = btype
        self.content = BlocketteContent()
        self.definition = blockette_definition(self.btype)

    def fields_packed(self):
        for field in self.definition.get_fields():
            if isinstance(field, Repeat):
                for rfield in field.fields:
                    yield rfield
                continue
            else:
                yield field

    def value(self, name):
        return getattr(self.content, ident(name))

    def unpack(self, data):
        pos = 0
        fields = self.definition.get_fields()
        while True:
            try:
                field = fields.pop(0)
            except IndexError:
                break
            if isinstance(field, Repeat):
                repetitions = self.value(field.repetitions_key)
                fields.extend(field.fields * repetitions)
                continue

            value, length = field.extract(data, pos)
            if field.repeats:
                self.content.add(ident(field.name), value)
            else:
                self.content.set(ident(field.name), value)

            pos += length

        if pos != len(data):
            logger.warning('Blockette of incorrect length found')

    def __str__(self):
        s = []
        npad = 0
        for field in self.fields_packed():
            npad = max(npad, len(field.name))

        for field in self.fields_packed():
            if field.repeats:
                v = ' '.join([field.format(v) for v in self.value(field.name)])
            else:
                v = field.format(self.value(field.name))
            s.append(('%-'+str(npad+1)+'s %s') % (field.name+':', v))

        return '\n'.join(s)


class DatalessSeedReader(object):

    def __init__(self):
        pass

    def read(self, filename):
        f = open(filename, 'r')
        data = f.read(256)      # 256 is the minimal blockette size

        sequence_number, type_code, continuation_code = \
            self._unpack_control_header(data)

        blockette_type, blockette_length = \
            self._peek_into_blockette(data[8:8+7])

        assert blockette_length < 256-8
        assert blockette_type == 10
        assert continuation_code == ' '

        volumeheader = Blockette(blockette_type)
        volumeheader.unpack(data[8:8+blockette_length])
        print(volumeheader)
        print('---')

        self.logical_record_length = volumeheader.content.logical_record_length
        data += f.read(self.logical_record_length-256)
        pos = 8+blockette_length
        while True:
            try:
                blockette_type, blockette_length = \
                    self._peek_into_blockette(data[pos:pos+7])
            except NotABlockette:
                data = f.read(self.logical_record_length)
                if len(data) == 0:
                    break
                sequence_number, type_code, continuation_code = \
                    self._unpack_control_header(data)
                pos = 8
                blockette_type, blockette_length = \
                    self._peek_into_blockette(data[pos:pos+7])

            while len(data) < pos+blockette_length:
                moredata = f.read(self.logical_record_length)
                assert len(moredata) == self.logical_record_length
                sequence_number, type_code, continuation_code = \
                    self._unpack_control_header(moredata)
                assert continuation_code == '*'
                data += moredata[8:]

            if pos > self.logical_record_length:
                data = data[pos:]
                pos = 0

            try:
                x = Blockette(blockette_type)
                x.unpack(data[pos:pos+blockette_length])
                print(x)
                print('---')

            except UnknownBlocketteType as e:
                print('unknown blockette type found: %s' % e)

            pos += blockette_length

    def _unpack_control_header(self, data):
        sequence_number = int(data[0:6])
        type_code = data[6]
        continuation_code = data[7]

        return sequence_number, type_code, continuation_code

    def _peek_into_blockette(self, data):
        if len(data) != 7:
            raise NotABlockette()
        s = data[0:3]
        if s == '   ':
            raise NotABlockette()
        blockette_type = int(s)
        s = data[3:7]
        if s == '    ':
            raise NotABlockette()
        blockette_length = int(s)
        return blockette_type, blockette_length


if __name__ == '__main__':
    dsr = DatalessSeedReader()

    dsr.read(sys.argv[1])
