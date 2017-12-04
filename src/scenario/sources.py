import os.path as op

from pyrocko.guts import Timestamp, Float, Int, Bool
from pyrocko import moment_tensor, util, gf

from .base import LocationGenerator, ScenarioError

km = 1e3

guts_prefix = 'pf.scenario'


class SourceGenerator(LocationGenerator):

    avoid_water = Bool.T(
        default=False,
        help='Avoid sources offshore under the ocean / lakes.')


class DCSourceGenerator(SourceGenerator):
    nevents = Int.T(default=10)

    time_min = Timestamp.T(default=util.str_to_time('2017-01-01 00:00:00'))
    time_max = Timestamp.T(default=util.str_to_time('2017-01-03 00:00:00'))

    magnitude_min = Float.T(default=4.0)
    magnitude_max = Float.T(default=7.0)

    depth_min = Float.T(default=0.0)
    depth_max = Float.T(default=30*km)

    strike = Float.T(optional=True)
    dip = Float.T(optional=True)
    rake = Float.T(optional=True)
    perturbation_angle_std = Float.T(optional=True)

    def get_source(self, ievent):
        rstate = self.get_rstate(ievent)
        time = rstate.uniform(self.time_min, self.time_max)
        lat, lon = self.get_latlon(ievent)
        depth = rstate.uniform(self.depth_min, self.depth_max)
        magnitude = rstate.uniform(self.magnitude_min, self.magnitude_max)

        if self.strike is None and self.dip is None and self.rake is None:
            mt = moment_tensor.MomentTensor.random_dc(x=rstate.uniform(size=3))
        else:
            if None in (self.strike, self.dip, self.rake):
                raise ScenarioError(
                    'DCSourceGenerator: '
                    'strike, dip, and rake must be used in combination')

            mt = moment_tensor.MomentTensor(
                strike=self.strike, dip=self.dip, rake=self.rake)

            if self.perturbation_angle_std is not None:
                mt = mt.random_rotated(
                    self.perturbation_angle_std,
                    rstate=rstate)

        (s, d, r), (_, _, _) = mt.both_strike_dip_rake()

        source = gf.DCSource(
            time=float(time),
            lat=float(lat),
            lon=float(lon),
            depth=float(depth),
            magnitude=float(magnitude),
            strike=float(s),
            dip=float(d),
            rake=float(r))

        return source

    def get_sources(self):
        sources = []
        for ievent in range(self.nevents):
            sources.append(self.get_source(ievent))

        return sources

    def dump_data(self, path):
        fn_sources = op.join(path, 'sources.yml')
        with open(fn_sources, 'w') as f:
            for src in self.get_sources():
                f.write(src.dump())

        fn_events = op.join(path, 'events.txt')
        with open(fn_events, 'w') as f:
            for src in self.get_sources():
                f.write(src.pyrocko_event().dump())

        return [fn_events, fn_sources]
