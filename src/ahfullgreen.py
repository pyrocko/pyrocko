import math
import numpy as num
from pyrocko import trace
from pyrocko import ahfullgreen_ext as ext

class AhfullgreenError(Exception):
    pass


def add_seismogram(
        vp, vs, density, qp, qs, x, f, m6,
        out_quantity, out_delta, out_offset,
        out_x, out_y, out_z, stf, want_far=1, want_intermediate=1, want_near=1):

    ns = [out.size for out in (out_x, out_y, out_z) if out is not None]

    if not all(n == ns[0] for n in ns):
        raise AhfullgreenError('length of component arrays must be identical')

    n = ns[0]

    nout = out_x.size//2 + 1

    specs = []
    for out in (out_x, out_y, out_z):
        if out is not None:
            specs.append(num.zeros(nout, dtype=num.complex))
        else:
            specs.append(None)

    x = num.asarray(x, num.float)
    f = num.asarray(f, num.float)
    m6 = num.asarray(m6, num.float)

    oc_c = {
        'displacement': 1, # treated externally
        'velocity': 1,
        'acceleration': 2}[out_quantity]


    out_spec_delta = float(2.0 * math.pi / (n*out_delta))
    out_spec_offset = 0.0

    omega = out_spec_offset + out_spec_delta * num.arange(nout)

    coeffs_stf = stf(omega/(2.*math.pi))

    r = math.sqrt(num.sum(x**2))

    ext.add_seismogram(
        float(vp), float(vs), float(density), float(qp), float(qs),
        x, f, m6, oc_c, out_spec_delta, out_spec_offset,
        specs[0], specs[1], specs[2], want_far, want_intermediate, want_near)


    tp = r / vp
    ts = r / vs

    tpad = stf.t_cutoff()

    if tpad is not None:
        icut1 = max(0, int(num.floor((tp - tpad) / out_delta)))
        icut2 = min(n, int(num.ceil((ts + tpad) / out_delta)))
    else:
        icut1 = 0
        icut2 = n

    for i, out in enumerate((out_x, out_y, out_z)):
        if out is None:
            continue

        temp = num.fft.irfft(coeffs_stf * specs[i])

        temp[:icut1] = 0.0
        temp[icut2:] = 0.0

        if out_quantity == 'displacement':
            out[:] += num.cumsum(temp) * out_delta
        else:
            out[:] += temp


class Impulse(object):
    def __init__(self):
        pass

    def t_cutoff(self):
        return None

    def __call__(self, f):
        omega = num.ones(len(f))

        return omega

class Gauss(object):
    def __init__(self, tau):
        self._tau = tau

    def t_cutoff(self):
        return self._tau * 2.

    def __call__(self, f):
        omega = f * 2. * math.pi

        return num.exp(-(omega**2 * self._tau**2 / 8.))

if __name__ == '__main__':

    x = (1000., 0., 0.)
    f = (0., 0., 0.)
    m6 = (0., 0., 0., 1., 0., 0.)

    vp = 3600.
    vs = 2000.

    tlen = x[0] / vs * 2.

    deltat = 0.001

    n = int(num.round(tlen / deltat))


    out_x = num.zeros(n)
    out_y = num.zeros(n)
    out_z = num.zeros(n)

    import pylab as lab

    tau = 0.01
    t = num.arange(1000) * deltat
    lab.plot(t, num.exp(-t**2/tau**2))
    #lab.show()

    add_seismogram(
        vp, vs, 1.0, 1.0, 1.0, x, f, m6, 'displacement', deltat, 0.0,
        out_x, out_y, out_z, Gauss(tau))


    trs = []
    for out, comp in zip([out_x, out_y, out_z], 'NED'):
        tr = trace.Trace('', 'Naja!', '', comp, deltat=deltat, tmin=0.0, ydata=out)
        trs.append(tr)

    trace.snuffle(trs)


