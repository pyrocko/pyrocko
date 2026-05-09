# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Squirrel IO adaptor to :py:mod:`pyrocko.io.yaff`.
'''

from pyrocko.io.io_common import get_stats, touch  # noqa
from ... import model


def provided_formats():
    return ['yaff']


def detect(first512):
    from pyrocko.io import yaff

    if yaff.detect(first512):
        return 'yaff'
    else:
        return None


def iload(format, file_path, segment, content):
    assert format == 'yaff'
    from pyrocko.io import yaff

    load_data = 'waveform' in content

    if segment is None:
        offset = 0
        nsegments = 0
    else:
        offset = segment
        nsegments = 1

    for tr in yaff.iload(
            file_path, load_data=load_data,
            offset=offset, nsegments=nsegments):

        nut = model.make_waveform_nut(
            file_segment=tr.meta['offset'],
            file_element=0,
            codes=tr.codes,
            tmin=tr.tmin,
            tmax=tr.tmin + tr.deltat * tr.data_len(),
            deltat=tr.deltat)

        if 'waveform' in content:
            nut.content = tr

        yield nut
