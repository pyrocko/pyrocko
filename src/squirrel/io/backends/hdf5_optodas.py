# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Squirrel IO adaptor to :py:mod:`pyrocko.io.tdms_idas`.
'''
from __future__ import annotations

from typing import Any, Generator

from ... import model

from pyrocko.io.io_common import get_stats, touch 

SEGMENT_SIZE = 1024*1024

try:
    import simpledas
except ImportError:
    simpledas = None


def provided_formats() -> list[str]:
    return ["hdf5_optodas"]


def detect(first512: bytes) -> bool:
    from pyrocko.io import hdf5_optodas
    if hdf5_optodas.detect(first512):
        return 'hdf5_optodas'
    else:
        return None


def iload(format: str, file_path: str, segment: int, content: tuple[str,...]) -> Generator[model.Nut, Any, None]:
    assert format == 'hdf5_optodas'
    from pyrocko.io import hdf5_optodas

    load_data = 'waveform' in content

    if segment is None:
        offset = 0
        nsegments = 0
    else:
        offset = segment
        nsegments = 1

    file_segment = None
    itr = 0
    for tr in hdf5_optodas.iload(
            file_path, load_data=load_data):

        # if file_segment != tr.meta['offset_start']:
        #     itr = 0
        #     file_segment = tr.meta['offset_start']

        nsamples = int(round((tr.tmax - tr.tmin) / tr.deltat)) + 1
        nut = model.make_waveform_nut(
            file_segment=file_segment,
            file_element=itr,
            codes=tr.codes,
            tmin=tr.tmin,
            tmax=tr.tmin + tr.deltat * nsamples,
            deltat=tr.deltat)

        if 'waveform' in content:
            nut.content = tr

        yield nut
        itr += 1
