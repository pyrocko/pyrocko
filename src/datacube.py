
import math
import numpy as num
from pyrocko import trace, util, plot

N_GPS_TAGS_WANTED = 200  # must match definition in datacube_ext.c


def color(c):
    c = plot.color(c)
    return tuple(x/255. for x in c)


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

    ic = ipos_block[ipos_block.size/2]
    tc = offset + slope * ic

    return ic, tc + ic * deltat + tref


def analyse_gps_tags(header, gps_tags, nsamples):

    ipos, t, fix, nsvs = gps_tags
    blocksize = N_GPS_TAGS_WANTED / 2
    deltat = 1.0 / int(header['S_RATE'])

    if ipos.size < blocksize:
        tmin = util.str_to_time(header['S_DATE'] + header['S_TIME'],
                                format='%y/%m/%d%H:%M:%S')
        tmax = tmin + (nsamples-1) * deltat
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
            tmin = t0 - i0 * deltat
            tmax = t3 + (nsamples - i3 - 1) * deltat
        else:
            tmin = t0 + (0 - i0) * (t1 - t0) / (i1 - i0)
            tmax = t2 + (nsamples - 1 - i2) * (t3 - t2) / (i3 - i2)

        control_points[0:0] = [(0, tmin)]
        control_points.append((nsamples-1, tmax))

        icontrol = num.array([x[0] for x in control_points], dtype=num.int64)
        tcontrol = num.array([x[1] for x in control_points], dtype=num.float)

        return tmin, tmax, icontrol, tcontrol


def plot_timeline(fns):
    from matplotlib import pyplot as plt
    from pyrocko import datacube_ext

    h = 3600.

    joined = [[], [], [], []]
    ioff = 0
    for fn in fns:
        with open(fn, 'r') as f:
            header, _, gps_tags, nsamples, _ = datacube_ext.load(
                f.fileno(), 1, 0, -1, None)

        header = dict(header)
        deltat = 1.0 / int(header['S_RATE'])

        ipos = gps_tags[0]
        ipos += ioff

        for i in range(4):
            joined[i].append(gps_tags[i])

        ioff += nsamples

    ipos, t, fix, nsvs = [num.concatenate(x) for x in joined]

    tref = num.median(t - ipos * deltat)
    tref = round(tref / deltat) * deltat

    x = ipos*deltat
    y = (t - tref) - ipos*deltat

    ifix = num.where(fix == 1)
    inofix = num.where(fix == 0)

    plt.plot(x[ifix]/h, y[ifix], '+', ms=5, color=color('chameleon3'))
    plt.plot(x[inofix]/h, y[inofix], 'x', ms=5, color=color('scarletred1'))

    tmin, tmax, icontrol, tcontrol = analyse_gps_tags(
        header, (ipos, t, fix, nsvs), ioff)

    tred = tcontrol - icontrol*deltat - tref
    plt.plot(icontrol*deltat/h, tred, color=color('aluminium6'))
    plt.plot(icontrol*deltat/h, tred, 'o', ms=5, color=color('aluminium6'))

    ymin = math.floor(tred.min() / deltat) * deltat - 0.1 * deltat
    ymax = math.ceil(tred.max() / deltat) * deltat + 0.1 * deltat

    plt.ylim(ymin, ymax)
    ygrid = math.floor(tred.min() / deltat) * deltat
    while ygrid < ymax:
        plt.axhline(ygrid, color=color('aluminium4'))
        ygrid += deltat

    xmin = icontrol[0]*deltat/h
    xmax = icontrol[-1]*deltat/h
    xsize = xmax - xmin
    xmin -= xsize * 0.1
    xmax += xsize * 0.1
    plt.xlim(xmin, xmax)

    plt.xlabel('Uncorrected (quartz) time [h]')
    plt.ylabel('Relative time correction [s]')

    plt.show()


def iload(fn, load_data=True, interpolation='sinc'):
    from pyrocko import datacube_ext
    from pyrocko import signal_ext

    if interpolation not in ('sinc', 'off'):
        raise NotImplemented(
            'no such interpolation method: %s' % interpolation)

    with open(fn, 'r') as f:
        if load_data:
            loadflag = 2
        else:
            if interpolation == 'off':
                loadflag = 0
            else:
                # must get correct nsamples if interpolation is off
                loadflag = 1

        header, data_arrays, gps_tags, nsamples, _ = datacube_ext.load(
            f.fileno(), loadflag, 0, -1, None)

    header = dict(header)
    deltat = 1.0 / int(header['S_RATE'])
    nchannels = int(header['CH_NUM'])

    tmin, tmax, icontrol, tcontrol = analyse_gps_tags(
        header, gps_tags, nsamples)

    tmin_ip = round(tmin / deltat) * deltat
    if interpolation != 'off':
        tmax_ip = round(tmax / deltat) * deltat
    else:
        tmax_ip = tmin_ip + (nsamples-1) * deltat

    nsamples_ip = int(round((tmax_ip - tmin_ip)/deltat)) + 1
    # to prevent problems with rounding errors:
    tmax_ip = tmin_ip + (nsamples_ip-1) * deltat

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

        toff = util.gps_utc_offset(tmin_ip)
        tr_tmin -= toff
        if tr_tmax is not None:
            tr_tmax -= toff

        tr = trace.Trace('', header['DEV_NO'], '', 'p%i' % i, deltat=deltat,
                         ydata=ydata, tmin=tr_tmin, tmax=tr_tmax, meta=header)

        yield tr


header_keys = {
    str: 'GIPP_V DEV_NO E_NAME GPS_PO S_TIME S_DATE DAT_NO'.split(),
    int: '''P_AMPL CH_NUM S_RATE D_FILT C_MODE A_CHOP F_TIME GPS_TI GPS_OF
            A_FILT A_PHAS GPS_ON ACQ_ON V_TCXO D_VOLT E_VOLT'''.split()}

all_header_keys = header_keys[str] + header_keys[int]


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

    if len([x for x in all_header_keys if x in kvs]) == 0:
        return False

    return True

if __name__ == '__main__':
    import sys
    plot_timeline(sys.argv[1:])
