from pyrocko.gf import STF
from pyrocko.guts import Float
import numpy as num


class LinearRampSTF(STF):
    '''Linearly decreasing ramp from maximum amplitude to zero.'''

    duration = Float.T(
        default=0.0,
        help='baseline of the ramp')

    anchor = Float.T(
        default=0.0,
        help='anchor point with respect to source-time: ('
             '-1.0: left -> source duration [0, T] ~ hypocenter time, '
             ' 0.0: center -> source duration [-T/2, T/2] ~ centroid time, '
             '+1.0: right -> source duration [-T, 0] ~ rupture end time)')

    def discretize_t(self, deltat, tref):
        # method returns discrete times and the respective amplitudes
        tmin_stf = tref - self.duration * (self.anchor + 1.) * 0.5
        tmax_stf = tref + self.duration * (1. - self.anchor) * 0.5
        tmin = round(tmin_stf / deltat) * deltat
        tmax = round(tmax_stf / deltat) * deltat
        nt = int(round((tmax - tmin) / deltat)) + 1
        times = num.linspace(tmin, tmax, nt)
        if nt > 1:
            amplitudes = num.linspace(1., 0., nt)
            amplitudes /= num.sum(amplitudes)  # normalise to keep moment
        else:
            amplitudes = num.ones(1)

        return times, amplitudes

    def base_key(self):
        # method returns STF name and the values
        return (self.__class__.__name__, self.duration, self.anchor)
