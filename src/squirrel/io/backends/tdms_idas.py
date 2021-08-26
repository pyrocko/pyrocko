from __future__ import absolute_import, print_function

from pyrocko.io.io_common import get_stats, touch  # noqa
from ... import model


def provided_formats():
    return ['tdms_idas']


def detect(first512):
    from pyrocko.io import tdms_idas

    if tdms_idas.detect(first512):
        return 'tdms_idas'
    else:
        return None


def iload(format, file_path, segment, content):
    assert format == 'tdms_idas'

    from pyrocko.io import tdms_idas

    load_data = 'waveform' in content

    for itr, tr in enumerate(tdms_idas.iload(file_path, load_data=load_data)):

        nsamples = int(round((tr.tmax - tr.tmin) / tr.deltat)) + 1

        nut = model.make_waveform_nut(
            file_segment=0,
            file_element=itr,
            agency='',
            network=tr.network,
            station=tr.station,
            location=tr.location,
            channel=tr.channel,
            tmin=tr.tmin,
            tmax=tr.tmin + tr.deltat * nsamples,
            deltat=tr.deltat)

        if 'waveform' in content:
            nut.content = tr

        yield nut
