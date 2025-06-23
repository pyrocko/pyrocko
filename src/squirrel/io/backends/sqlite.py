
import struct

# from pyrocko.io.io_common import get_stats, touch  # noqa
from ... import model


def unpack_application_id(b):
    return struct.unpack('>i', b)[0]


ID_TO_NAME = dict(
    (unpack_application_id(name.encode('ascii')), name)
    for name in ['ELK1'])


def provided_formats():
    return ['sqlite-' + name for name in sorted(ID_TO_NAME.values())]


def detect(first512):
    if first512.startswith(b'SQLite format 3\x00'):
        application_id = unpack_application_id(first512[68:68+4])
        if application_id in ID_TO_NAME:
            return 'sqlite-%s' % ID_TO_NAME[application_id]

    return None


def iload(format, file_path, segment, content):
    print('in iload', format, file_path, segment, content)

    if format == 'sqlite-ELK1':
        nut = model.make_elk_nut(
            file_segment=segment,
            file_element=0,
            codes=model.CodesX(''))

        yield nut


def get_stats(file_path):
    return 0.0, 0


def touch(file_path):
    pass
