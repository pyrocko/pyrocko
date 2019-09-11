# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
# python 2/3
from __future__ import absolute_import

import os
import math
import numpy as num
from builtins import str as newstr

from pyrocko import trace, util, plot
from pyrocko.guts import Object, Int, String, Timestamp

from . import io_common

N_GPS_TAGS_WANTED = 200  # must match definition in datacube_ext.c


def color(c):
    c = plot.color(c)
    return tuple(x/255. for x in c)


class DataCubeError(io_common.FileLoadError):
    pass


def make_control_point(ipos_block, t_block, tref, deltat):

    # reduce time (no drift would mean straight line)
    tred = (t_block - tref) - ipos_block*deltat

    # first round, remove outliers
    q25, q75 = num.percentile(tred, (25., 75.))
    iok = num.where(num.logical_and(q25 <= tred, tred <= q75))[0]

    # detrend
    slope, offset = num.polyfit(ipos_block[iok], tred[iok], 1)
    tred2 = tred - (offset + slope * ipos_block)

    # second round, remove outliers based on detrended tred, refit
    q25, q75 = num.percentile(tred2, (25., 75.))
    iok = num.where(num.logical_and(q25 <= tred2, tred2 <= q75))[0]
    slope, offset = num.polyfit(ipos_block[iok], tred[iok], 1)

    ic = ipos_block[ipos_block.size//2]
    tc = offset + slope * ic

    return ic, tc + ic * deltat + tref


def analyse_gps_tags(header, gps_tags, offset, nsamples):

    ipos, t, fix, nsvs = gps_tags
    deltat = 1.0 / int(header['S_RATE'])

    tquartz = offset + ipos * deltat

    toff = t - tquartz
    toff_median = num.median(toff)

    n = t.size

    dtdt = (t[1:n] - t[0:n-1]) / (tquartz[1:n] - tquartz[0:n-1])

    ok = abs(toff_median - toff) < 10.

    xok = num.abs(dtdt - 1.0) < 0.00001

    ok[0] = False
    ok[1:n] &= xok
    ok[0:n-1] &= xok
    ok[n-1] = False

    ipos = ipos[ok]
    t = t[ok]
    fix = fix[ok]
    nsvs = nsvs[ok]

    blocksize = N_GPS_TAGS_WANTED // 2

    if ipos.size < blocksize:
        tmin = util.str_to_time(header['S_DATE'] + header['S_TIME'],
                                format='%y/%m/%d%H:%M:%S') + offset * deltat

        tmax = tmin + (nsamples - 1) * deltat
        icontrol, tcontrol = None, None
        return tmin, tmax, icontrol, tcontrol

    else:
        j = 0
        control_points = []
        tref = num.median(t - ipos*deltat)
        while j < ipos.size - blocksize:
            ipos_block = ipos[j:j+blocksize]
            t_block = t[j:j+blocksize]
            ic, tc = make_control_point(ipos_block, t_block, tref, deltat)
            control_points.append((ic, tc))
            j += blocksize

        ipos_last = ipos[-blocksize:]
        t_last = t[-blocksize:]
        ic, tc = make_control_point(ipos_last, t_last, tref, deltat)
        control_points.append((ic, tc))

        i0, t0 = control_points[0]
        i1, t1 = control_points[1]
        i2, t2 = control_points[-2]
        i3, t3 = control_points[-1]
        if len(control_points) == 2:
            tmin = t0 - i0 * deltat - offset * deltat
            tmax = t3 + (nsamples - i3 - 1) * deltat
        else:
            tmin = t0 + (offset - i0) * (t1 - t0) / (i1 - i0)
            tmax = t2 + (offset + nsamples - 1 - i2) * (t3 - t2) / (i3 - i2)

        if offset < i0:
            control_points[0:0] = [(offset, tmin)]

        if offset + nsamples - 1 > i3:
            control_points.append((offset + nsamples - 1, tmax))

        icontrol = num.array([x[0] for x in control_points], dtype=num.int64)
        tcontrol = num.array([x[1] for x in control_points], dtype=num.float)

        return tmin, tmax, icontrol, tcontrol


def plot_timeline(fns):
    from matplotlib import pyplot as plt

    fig = plt.figure()
    axes = fig.gca()

    h = 3600.

    if isinstance(fns, (str, newstr)):
        ipos, t, fix, nsvs, header, offset, nsamples = \
            get_extended_timing_context(fns)

    else:
        ipos, t, fix, nsvs, header, offset, nsamples = \
            get_timing_context(fns)

    deltat = 1.0 / int(header['S_RATE'])

    tref = num.median(t - ipos * deltat)
    tref = round(tref / deltat) * deltat

    x = ipos*deltat
    y = (t - tref) - ipos*deltat

    ifix = num.where(fix != 0)
    inofix = num.where(fix == 0)

    axes.plot(x[ifix]/h, y[ifix], '+', ms=5, color=color('chameleon3'))
    axes.plot(x[inofix]/h, y[inofix], 'x', ms=5, color=color('scarletred1'))

    tmin, tmax, icontrol, tcontrol = analyse_gps_tags(
        header, (ipos, t, fix, nsvs), offset, nsamples)

    tred = tcontrol - icontrol*deltat - tref
    axes.plot(icontrol*deltat/h, tred, color=color('aluminium6'))
    axes.plot(icontrol*deltat/h, tred, 'o', ms=5, color=color('aluminium6'))

    ymin = math.floor(tred.min() / deltat) * deltat - 0.1 * deltat
    ymax = math.ceil(tred.max() / deltat) * deltat + 0.1 * deltat

    # axes.set_ylim(ymin, ymax)
    if ymax - ymin < 100 * deltat:
        ygrid = math.floor(tred.min() / deltat) * deltat
        while ygrid < ymax:
            axes.axhline(ygrid, color=color('aluminium4'))
            ygrid += deltat

    xmin = icontrol[0]*deltat/h
    xmax = icontrol[-1]*deltat/h
    xsize = xmax - xmin
    xmin -= xsize * 0.1
    xmax += xsize * 0.1
    axes.set_xlim(xmin, xmax)

    axes.set_xlabel('Uncorrected (quartz) time [h]')
    axes.set_ylabel('Relative time correction [s]')

    plt.show()


g_dir_contexts = {}


class DirContextEntry(Object):
    path = String.T()
    tstart = Timestamp.T()
    ifile = Int.T()


class DirContext(Object):
    path = String.T()
    mtime = Timestamp.T()
    entries = DirContextEntry.T()

    def get_entry(self, fn):
        path = os.path.abspath(fn)
        for entry in self.entries:
            if entry.path == path:
                return entry

        raise Exception('entry not found')

    def iter_entries(self, fn, step=1):
        current = self.get_entry(fn)
        by_ifile = dict(
            (entry.ifile, entry) for entry in self.entries
            if entry.tstart == current.tstart)

        icurrent = current.ifile
        while True:
            icurrent += step
            try:
                yield by_ifile[icurrent]

            except KeyError:
                break


def context(fn):
    from pyrocko import datacube_ext

    dpath = os.path.dirname(os.path.abspath(fn))
    mtimes = [os.stat(dpath)[8]]

    dentries = sorted([os.path.join(dpath, f) for f in os.listdir(dpath)
                       if os.path.isfile(os.path.join(dpath, f))])
    for dentry in dentries:
        fn2 = os.path.join(dpath, dentry)
        mtimes.append(os.stat(fn2)[8])

    mtime = float(max(mtimes))

    if dpath in g_dir_contexts:
        dir_context = g_dir_contexts[dpath]
        if dir_context.mtime == mtime:
            return dir_context

        del g_dir_contexts[dpath]

    entries = []
    for dentry in dentries:
        fn2 = os.path.join(dpath, dentry)
        if not os.path.isfile(fn2):
            continue

        with open(fn2, 'rb') as f:
            first512 = f.read(512)
            if not detect(first512):
                continue

        with open(fn2, 'rb') as f:
            try:
                header, data_arrays, gps_tags, nsamples, _ = \
                        datacube_ext.load(f.fileno(), 3, 0, -1, None)

            except datacube_ext.DataCubeError as e:
                e = DataCubeError(str(e))
                e.set_context('filename', fn)
                raise e

        header = dict(header)
        entries.append(DirContextEntry(
            path=os.path.abspath(fn2),
            tstart=util.str_to_time(
                '20' + header['S_DATE'] + ' ' + header['S_TIME'],
                format='%Y/%m/%d %H:%M:%S'),
            ifile=int(header['DAT_NO'])))

    dir_context = DirContext(mtime=mtime, path=dpath, entries=entries)

    return dir_context


def get_time_infos(fn):
    from pyrocko import datacube_ext

    with open(fn, 'rb') as f:
        try:
            header, _, gps_tags, nsamples, _ = datacube_ext.load(
                f.fileno(), 1, 0, -1, None)

        except datacube_ext.DataCubeError as e:
            e = DataCubeError(str(e))
            e.set_context('filename', fn)
            raise e

    return dict(header), gps_tags, nsamples


def get_timing_context(fns):
    joined = [[], [], [], []]
    ioff = 0
    for fn in fns:
        header, gps_tags, nsamples = get_time_infos(fn)

        ipos = gps_tags[0]
        ipos += ioff

        for i in range(4):
            joined[i].append(gps_tags[i])

        ioff += nsamples

    ipos, t, fix, nsvs = [num.concatenate(x) for x in joined]

    nsamples = ioff
    return ipos, t, fix, nsvs, header, 0, nsamples


def get_extended_timing_context(fn):
    c = context(fn)

    header, gps_tags, nsamples_base = get_time_infos(fn)

    ioff = 0
    aggregated = [gps_tags]

    nsamples_total = nsamples_base

    if num.sum(gps_tags[2]) == 0:

        ioff = nsamples_base
        for entry in c.iter_entries(fn, 1):

            _, gps_tags, nsamples = get_time_infos(entry.path)

            ipos = gps_tags[0]
            ipos += ioff

            aggregated.append(gps_tags)
            nsamples_total += nsamples

            if num.sum(gps_tags[2]) > 0:
                break

            ioff += nsamples

        ioff = 0
        for entry in c.iter_entries(fn, -1):

            _, gps_tags, nsamples = get_time_infos(entry.path)

            ioff -= nsamples

            ipos = gps_tags[0]
            ipos += ioff

            aggregated[0:0] = [gps_tags]

            nsamples_total += nsamples

            if num.sum(gps_tags[2]) > 0:
                break

    ipos, t, fix, nsvs = [num.concatenate(x) for x in zip(*aggregated)]

#    return ipos, t, fix, nsvs, header, ioff, nsamples_total
    return ipos, t, fix, nsvs, header, 0, nsamples_base


def iload(fn, load_data=True, interpolation='sinc'):
    from pyrocko import datacube_ext
    from pyrocko import signal_ext

    if interpolation not in ('sinc', 'off'):
        raise NotImplementedError(
            'no such interpolation method: %s' % interpolation)

    with open(fn, 'rb') as f:
        if load_data:
            loadflag = 2
        else:
            if interpolation == 'off':
                loadflag = 0
            else:
                # must get correct nsamples if interpolation is off
                loadflag = 1

        try:
            header, data_arrays, gps_tags, nsamples, _ = datacube_ext.load(
                f.fileno(), loadflag, 0, -1, None)

        except datacube_ext.DataCubeError as e:
            e = DataCubeError(str(e))
            e.set_context('filename', fn)
            raise e

    header = dict(header)
    deltat = 1.0 / int(header['S_RATE'])
    nchannels = int(header['CH_NUM'])

    ipos, t, fix, nsvs, header_, offset_, nsamples_ = \
        get_extended_timing_context(fn)

    tmin, tmax, icontrol, tcontrol = analyse_gps_tags(
        header_, (ipos, t, fix, nsvs), offset_, nsamples_)

    tmin_ip = round(tmin / deltat) * deltat
    if interpolation != 'off':
        tmax_ip = round(tmax / deltat) * deltat
    else:
        tmax_ip = tmin_ip + (nsamples-1) * deltat

    nsamples_ip = int(round((tmax_ip - tmin_ip)/deltat)) + 1
    # to prevent problems with rounding errors:
    tmax_ip = tmin_ip + (nsamples_ip-1) * deltat

    leaps = num.array(
        [x[0] + util.gps_utc_offset(x[0]) for x in util.read_leap_seconds2()],
        dtype=num.float)

    for i in range(nchannels):
        if load_data:
            arr = data_arrays[i]
            assert arr.size == nsamples

            if interpolation == 'sinc' and icontrol is not None:
                ydata = num.empty(nsamples_ip, dtype=num.float)
                signal_ext.antidrift(
                    icontrol, tcontrol,
                    arr.astype(num.float), tmin_ip, deltat, ydata)

                ydata = num.round(ydata).astype(arr.dtype)
            else:
                ydata = arr

            tr_tmin = tmin_ip
            tr_tmax = None
        else:
            ydata = None
            tr_tmin = tmin_ip
            tr_tmax = tmax_ip

        tr = trace.Trace('', header['DEV_NO'], '', 'p%i' % i, deltat=deltat,
                         ydata=ydata, tmin=tr_tmin, tmax=tr_tmax, meta=header)

        bleaps = num.logical_and(tmin_ip <= leaps, leaps < tmax_ip)

        if num.any(bleaps):
            assert num.sum(bleaps) == 1
            tcut = leaps[bleaps][0]

            for tmin_cut, tmax_cut in [
                    (tr.tmin, tcut), (tcut, tr.tmax+tr.deltat)]:

                try:
                    tr_cut = tr.chop(tmin_cut, tmax_cut, inplace=False)
                    tr_cut.shift(
                        util.utc_gps_offset(0.5*(tr_cut.tmin+tr_cut.tmax)))
                    yield tr_cut

                except trace.NoData:
                    pass

        else:
            tr.shift(util.utc_gps_offset(0.5*(tr.tmin+tr.tmax)))
            yield tr


header_keys = {
    str: b'GIPP_V DEV_NO E_NAME GPS_PO S_TIME S_DATE DAT_NO'.split(),
    int: b'''P_AMPL CH_NUM S_RATE D_FILT C_MODE A_CHOP F_TIME GPS_TI GPS_OF
            A_FILT A_PHAS GPS_ON ACQ_ON V_TCXO D_VOLT E_VOLT'''.split()}

all_header_keys = header_keys[str] + header_keys[int]


def detect(first512):
    s = first512

    if len(s) < 512:
        return False

    if ord(s[0:1]) >> 4 != 15:
        return False

    n = s.find(b'\x80')
    if n == -1:
        n = len(s)

    s = s[1:n]
    s = s.replace(b'\xf0', b'')
    s = s.replace(b';', b' ')
    s = s.replace(b'=', b' ')
    kvs = s.split(b' ')

    if len([k for k in all_header_keys if k in kvs]) == 0:
        return False
    return True


if __name__ == '__main__':
    import sys
    fns = sys.argv[1:]
    if len(fns) > 1:
        plot_timeline(fns)
    else:
        plot_timeline(fns[0])
