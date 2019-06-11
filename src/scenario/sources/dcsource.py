import numpy as num

from pyrocko.guts import Float
from pyrocko import moment_tensor, gf

from .base import SourceGenerator
from ..error import ScenarioError

km = 1e3
guts_prefix = 'pf.scenario'


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
        magnitude = self.draw_magnitude(rstate)

        if self.strike is None and self.dip is None and self.rake is None:
            mt = moment_tensor.MomentTensor.random_dc(x=rstate.uniform(size=3))
        else:
            if None in (self.strike, self.dip, self.rake):
                raise ScenarioError(
                    'DCSourceGenerator: '
                    'strike, dip, and rake must be used in combination.')

            mt = moment_tensor.MomentTensor(
                strike=self.strike, dip=self.dip, rake=self.rake)

            if self.perturbation_angle_std is not None:
                mt = mt.random_rotated(
                    self.perturbation_angle_std,
                    rstate=rstate)

        (s, d, r), (_, _, _) = mt.both_strike_dip_rake()

        source = gf.DCSource(
            name='ev%04i' % ievent,
            time=float(time),
            lat=float(lat),
            lon=float(lon),
            depth=float(depth),
            magnitude=float(magnitude),
            strike=float(s),
            dip=float(d),
            rake=float(r))

        return source

    def add_map_artists(self, automap):
        from pyrocko import gmtpy

        for source in self.get_sources():
            event = source.pyrocko_event()
            mt = event.moment_tensor.m_up_south_east()

            xx = num.trace(mt) / 3.
            mc = num.matrix([[xx, 0., 0.], [0., xx, 0.], [0., 0., xx]])
            mc = mt - mc
            mc = mc / event.moment_tensor.scalar_moment() * \
                moment_tensor.magnitude_to_moment(5.0)
            m6 = tuple(moment_tensor.to6(mc))
            symbol_size = 20.
            automap.gmt.psmeca(
                S='%s%g' % ('d', symbol_size / gmtpy.cm),
                in_rows=[(source.lon, source.lat, 10) + m6 + (1, 0, 0)],
                M=True,
                *automap.jxyr)
