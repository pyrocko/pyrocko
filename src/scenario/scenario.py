import numpy as num
import logging

from pyrocko.guts import List
from pyrocko import moment_tensor, gmtpy, pile

from .base import LocationGenerator, ScenarioError
from .sources import SourceGenerator, DCSourceGenerator
from .targets import TargetGenerator

logger = logging.getLogger('pyrocko.scenario')
guts_prefix = 'pf.scenario'


class ScenarioGenerator(LocationGenerator):

    target_generators = List.T(
        TargetGenerator.T(),
        default=[],
        help='Targets to spawn in the scenario.')

    source_generator = SourceGenerator.T(
        default=DCSourceGenerator.D(),
        help='Sources to spawn in the scenario.')

    def __init__(self, **kwargs):
        LocationGenerator.__init__(self, **kwargs)

        for gen in self.target_generators:
            gen.update_hierarchy(self)

        for itry in range(self.ntries):

            try:
                self.get_stations()
                self.get_sources()
                return

            except ScenarioError:
                self.retry()

        raise ScenarioError(
            'could not generate scenario within %i tries' % self.ntries)

    def init_modelling(self, engine):
        self._engine = engine

    def get_engine(self):
        return self._engine

    def get_sources(self):
        return self.source_generator.get_sources()

    def collect(collector):
        if not callable(collector):
            raise AttributeError('This method should not be called directly.')

        def method(self, *args, **kwargs):
            result = []
            for gen in self.target_generators:
                result.extend(
                    collector(self, *args, **kwargs)(gen, *args, **kwargs))
            return result

        return method

    @collect
    def get_stations(self):
        return lambda gen: gen.get_stations()

    @collect
    def get_waveforms(self, tmin=None, tmax=None):
        return lambda gen, *args, **kwargs: gen.get_waveforms(
            self._engine, self.get_sources(), *args, **kwargs)

    @collect
    def get_insar_scenes(self, tmin=None, tmax=None):
        return lambda gen, *args, **kwargs: gen.get_insar_scenes(
            self._engine, self.get_sources(), *args, **kwargs)

    @collect
    def get_gnss_offsets(self, tmin=None, tmax=None):
        return lambda gen, *args, **kwargs: gen.get_gnss_offsets(
            self._engine, self.get_sources(), *args, **kwargs)

    @collect
    def dump_data(self, path, tmin=None, tmax=None, overwrite=False):
        self.source_generator.dump_data(path)
        return lambda gen, *args, **kwargs: gen.dump_data(
            self._engine, self.get_sources(), *args, **kwargs)

    @collect
    def _get_time_ranges(self):
        return lambda gen: [gen.get_time_range(self.get_sources())]

    def get_time_range(self):
        ranges = num.array(self._get_time_ranges())
        return ranges.min(), ranges.max()

    def get_pile(self, tmin=None, tmax=None):
        trs = self.get_waveforms(tmin, tmax)
        return pile(trs)


def draw_scenario_gmt(generator, fn):
    from pyrocko import automap

    lat, lon = generator.station_generator.get_center_latlon()
    radius = generator.station_generator.get_radius()

    m = automap.Map(
        width=30.,
        height=30.,
        lat=lat,
        lon=lon,
        radius=radius,
        show_topo=False,
        show_grid=True,
        show_rivers=False,
        color_wet=(216, 242, 254),
        color_dry=(238, 236, 230))

    stations = generator.get_stations()
    lats = [s.lat for s in stations]
    lons = [s.lon for s in stations]

    m.gmt.psxy(
        in_columns=(lons, lats),
        S='t8p',
        G='black',
        *m.jxyr)

    if len(stations) < 20:
        for station in stations:
            m.add_label(station.lat, station.lon, '.'.join(
                x for x in (station.network, station.station) if x))

    sources = generator.get_sources()

    for source in sources:

        event = source.pyrocko_event()

        mt = event.moment_tensor.m_up_south_east()
        xx = num.trace(mt) / 3.
        mc = num.matrix([[xx, 0., 0.], [0., xx, 0.], [0., 0., xx]])
        mc = mt - mc
        mc = mc / event.moment_tensor.scalar_moment() * \
            moment_tensor.magnitude_to_moment(5.0)
        m6 = tuple(moment_tensor.to6(mc))

        symbol_size = 20.
        m.gmt.psmeca(
            S='%s%g' % ('d', symbol_size / gmtpy.cm),
            in_rows=[(source.lon, source.lat, 10) + m6 + (1, 0, 0)],
            M=True,
            *m.jxyr)

    for patch in generator.get_insar_patches():
        symbol_size = 50.
        coords = num.array(patch.get_corner_coordinates())
        m.gmt.psxy(in_rows=num.fliplr(coords),
                   L=True,
                   *m.jxyr)

    m.save(fn)
