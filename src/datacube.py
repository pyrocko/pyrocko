
import numpy as num
from pyrocko import trace, util


def iload(fn, load_data=True):
    from pyrocko import datacube_ext

    blocksize = 60

    with open(fn, 'r') as f:
        if load_data:
            loadflag = 2
        else:
            loadflag = 0
        header, data_arrays, gps_tags, nsamples, _ = datacube_ext.load(
            f.fileno(), loadflag, 0, -1)

    h = dict(header)
    deltat = 1.0 / int(h['S_RATE'])
    nchannels = int(h['CH_NUM'])

    ipos, t, fix, nsvs = gps_tags

    if ipos.size >= blocksize:
        j = 0
        control_points = []
        while j < ipos.size - blocksize:
            ipos_block = ipos[j:j+blocksize]
            t_block = t[j:j+blocksize]
            tcontrol = t_block[0] + num.median(
                (t_block-t_block[0])-(ipos_block-ipos_block[0])*deltat)

            control_points.append((ipos_block[0], tcontrol))
            j += blocksize

        ipos_last = ipos[-blocksize:]
        t_last = t[-blocksize:]
        tcontrol = t_last[-1] + num.median(
            (t_last-t_last[-1])-(ipos_last-ipos_last[-1])*deltat)

        control_points.append((ipos_last[-1], tcontrol))
        i0, t0 = control_points[0]
        i1, t1 = control_points[-1]
        tmin = t0 - i0 * deltat
        tmax = t1 + (nsamples - i1 - 1) * deltat
        control_points[0:0] = [(0, tmin)]
        control_points.append((nsamples-1, tmax))

        if load_data:
            icontrol, tcontrol = num.array(control_points).T
            tsamples = num.interp(num.arange(nsamples), icontrol, tcontrol)

    else:
        tmin = util.str_to_time(h['S_DATE'] + h['S_TIME'],
                                format='%y/%m/%d%H:%M:%S')
        tmax = tmin + (nsamples-1) * deltat
        if load_data:
            tsamples = num.linspace(tmin, tmax, nsamples)

    tmin_ip = round(tmin / deltat) * deltat
    tmax_ip = round(tmax / deltat) * deltat
    nsamples_ip = int(round((tmax_ip - tmin_ip)/deltat)) + 1
    # to prevent problems with rounding errors:
    tmax_ip = tmin_ip + (nsamples_ip-1) * deltat

    if load_data:
        tsamples_ip = num.arange(nsamples_ip) * deltat + tmin_ip

    for i in range(nchannels):
        if load_data:
            arr = data_arrays[i]
            assert arr.size == nsamples
            ydata = num.round(
                num.interp(tsamples_ip, tsamples, arr)).astype(num.int32)
            tr_tmax = None
        else:
            ydata = None
            tr_tmax = tmax_ip

        tr = trace.Trace('', h['DEV_NO'], '', 'p%i' % i, deltat=deltat,
                         ydata=ydata, tmin=tmin_ip, tmax=tr_tmax, meta=h)

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
