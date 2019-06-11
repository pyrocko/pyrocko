# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
from __future__ import division, absolute_import
from builtins import zip

from struct import unpack
import os
import re
import logging

from pyrocko import trace
from pyrocko.util import reuse, ensuredirs
from .io_common import FileLoadError, FileSaveError

logger = logging.getLogger('pyrocko.io.mseed')


class CodeTooLong(FileSaveError):
    pass


def iload(filename, load_data=True):
    from pyrocko import mseed_ext

    have_zero_rate_traces = False
    try:
        traces = []
        for tr in mseed_ext.get_traces(filename, load_data):
            network, station, location, channel = tr[1:5]
            tmin = float(tr[5])/float(mseed_ext.HPTMODULUS)
            tmax = float(tr[6])/float(mseed_ext.HPTMODULUS)
            try:
                deltat = reuse(1.0/float(tr[7]))
            except ZeroDivisionError:
                have_zero_rate_traces = True
                continue

            ydata = tr[8]

            traces.append(trace.Trace(
                network, station, location, channel, tmin, tmax,
                deltat, ydata))

        for tr in traces:
            yield tr

    except (OSError, mseed_ext.MSeedError) as e:
        raise FileLoadError(str(e)+' (file: %s)' % filename)

    if have_zero_rate_traces:
        logger.warning(
            'Ignoring traces with sampling rate of zero in file %s '
            '(maybe LOG traces)' % filename)


def as_tuple(tr):
    from pyrocko import mseed_ext
    itmin = int(round(tr.tmin*mseed_ext.HPTMODULUS))
    itmax = int(round(tr.tmax*mseed_ext.HPTMODULUS))
    srate = 1.0/tr.deltat
    return (tr.network, tr.station, tr.location, tr.channel,
            itmin, itmax, srate, tr.get_ydata())


def save(traces, filename_template, additional={}, overwrite=True):
    from pyrocko import mseed_ext

    fn_tr = {}
    for tr in traces:
        for code, maxlen, val in zip(
                ['network', 'station', 'location', 'channel'],
                [2, 5, 2, 3],
                tr.nslc_id):

            if len(val) > maxlen:
                raise CodeTooLong(
                    '%s code too long to be stored in MSeed file: %s' %
                    (code, val))

        fn = tr.fill_template(filename_template, **additional)
        if not overwrite and os.path.exists(fn):
            raise FileSaveError('File exists: %s' % fn)

        if fn not in fn_tr:
            fn_tr[fn] = []

        fn_tr[fn].append(tr)

    for fn, traces_thisfile in fn_tr.items():
        trtups = []
        traces_thisfile.sort(key=lambda a: a.full_id)
        for tr in traces_thisfile:
            trtups.append(as_tuple(tr))

        ensuredirs(fn)
        try:
            mseed_ext.store_traces(trtups, fn)
        except mseed_ext.MSeedError as e:
            raise FileSaveError(
                str(e) + ' (while storing traces to file \'%s\')' % fn)

    return list(fn_tr.keys())


tcs = {}


def detect(first512):

    if len(first512) < 256:
        return False

    rec = first512

    try:
        sequence_number = int(rec[:6])
    except Exception:
        return False
    if sequence_number < 0:
        return False

    type_code = rec[6:7]
    if type_code in b'DRQM':
        bads = []
        for sex in '<>':
            bad = False
            fmt = sex + '6s1s1s5s2s3s2s10sH2h4Bl2H'
            vals = unpack(fmt, rec[:48])
            fmt_btime = sex + 'HHBBBBH'
            tvals = unpack(fmt_btime, vals[7])
            if tvals[1] < 1 or tvals[1] > 367 or tvals[2] > 23 or \
                    tvals[3] > 59 or tvals[4] > 60 or tvals[6] > 9999:
                bad = True

            bads.append(bad)

        if all(bads):
            return False

    else:
        if type_code not in b'VAST':
            return False

        continuation_code = rec[7:8]
        if continuation_code != b' ':
            return False

        blockette_type = rec[8:8+3].decode()
        if not re.match(r'^\d\d\d$', blockette_type):
            return False

        try:
            blockette_length = int(rec[11:11+4])
        except Exception:
            return False

        if blockette_length < 7:
            return False

    return True
