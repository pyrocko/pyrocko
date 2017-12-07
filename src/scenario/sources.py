import numpy as num
import os.path as op

from pyrocko.guts import Timestamp, Float, Int, Bool
from pyrocko import moment_tensor, util, gf

from .base import LocationGenerator, ScenarioError

km = 1e3

guts_prefix = 'pf.scenario'


class SourceGenerator(LocationGenerator):

    nevents = Int.T(default=2)
    avoid_water = Bool.T(
        default=False,
        help='Avoid sources offshore under the ocean / lakes.')

    radius = Float.T(
        default=10*km)

    time_min = Timestamp.T(default=util.str_to_time('2017-01-01 00:00:00'))
    time_max = Timestamp.T(default=util.str_to_time('2017-01-03 00:00:00'))

    magnitude_min = Float.T(default=4.0)
    magnitude_max = Float.T(default=7.0)

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


class DCSourceGenerator(SourceGenerator):

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


class RectangularSourceGenerator(SourceGenerator):
    depth_min = Float.T(default=0.0)
    depth_max = Float.T(default=5*km)

    strike = Float.T(
        optional=True)
    dip = Float.T(
        optional=True)
    rake = Float.T(
        optional=True)
    depth = Float.T(
        optional=True)
    width = Float.T(
        optional=True)
    length = Float.T(
        optional=True)

    def get_source(self, ievent):
        rstate = self.get_rstate(ievent)
        time = rstate.uniform(self.time_min, self.time_max)
        lat, lon = self.get_latlon(ievent)
        depth = rstate.uniform(self.depth_min, self.depth_max)
        magnitude = rstate.uniform(self.magnitude_min, self.magnitude_max)

        moment = moment_tensor.magnitude_to_moment(magnitude)

        # After Mai and Beroza (2000)
        length = num.exp(-6.27 + 0.4*num.log(moment))
        width = num.exp(-4.24 + 0.32*num.log(moment))

        length = length if not self.length else self.length
        width = width if not self.width else self.width
        depth = depth if not self.depth else self.depth

        if self.strike is None and self.dip is None and self.rake is None:
            strike, rake = rstate.uniform(-180., 180., 2)
            dip = rstate.uniform(0., 90.)
        else:
            if None in (self.strike, self.dip, self.rake):
                raise ScenarioError(
                    'RectangularFaultGenerator: '
                    'strike, dip, rake'
                    ' must be used in combination')

            strike = self.strike
            dip = self.dip
            rake = self.rake

        source = gf.RectangularSource(
            time=float(time),
            lat=float(lat),
            lon=float(lon),
            magnitude=magnitude,

            depth=float(depth),
            length=float(length),
            width=float(width),
            strike=float(strike),
            dip=float(dip),
            rake=float(rake))

        return source
