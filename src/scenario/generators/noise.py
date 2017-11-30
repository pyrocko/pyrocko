from builtins import range
import hashlib
import math

from pyrocko.guts import Float
from pyrocko import trace

from ..base import Generator

guts_prefix = 'pf.scenario'


class NoiseGenerator(Generator):

    def get_time_increment(self, deltat):
        return deltat * 1024

    def get_intersecting_snippets(self, deltat, codes, tmin, tmax):
        raise NotImplemented()

    def add_noise(self, tr):
        for ntr in self.get_intersecting_snippets(
                tr.deltat, tr.nslc_id, tr.tmin, tr.tmax):
            tr.add(ntr)


class WhiteNoiseGenerator(NoiseGenerator):

    scale = Float.T(default=1e-6)

    def get_seed_offset2(self, deltat, iw, codes):
        m = hashlib.sha1(('%e %i %s.%s.%s.%s' % ((deltat, iw) + codes))
                         .encode('utf8'))
        return int(m.hexdigest(), base=16) % 1000

    def get_intersecting_snippets(self, deltat, codes, tmin, tmax):
        tinc = self.get_time_increment(deltat)
        iwmin = int(math.floor(tmin / tinc))
        iwmax = int(math.floor(tmax / tinc))

        trs = []
        for iw in range(iwmin, iwmax+1):
            seed_offset = self.get_seed_offset2(deltat, iw, codes)
            rstate = self.get_rstate(seed_offset)

            n = int(round(tinc // deltat))

            trs.append(trace.Trace(
                codes[0], codes[1], codes[2], codes[3],
                deltat=deltat,
                tmin=iw*tinc,
                ydata=rstate.normal(loc=0, scale=self.scale, size=n)))

        return trs
