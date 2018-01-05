import numpy as num
import logging
import os.path as op

from pyrocko.guts import List
from pyrocko import pile, util, model

from .base import LocationGenerator, ScenarioError
from .sources import SourceGenerator, DCSourceGenerator
from .targets import TargetGenerator, AVAILABLE_TARGETS

logger = logging.getLogger('pyrocko.scenario')
guts_prefix = 'pf.scenario'


class CannotCreate(Exception):
    pass


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

    def get_events(self):
        return [s.pyrocko_event() for s in self.get_sources()]

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
        return lambda gen, *a, **kw: gen.get_waveforms(
            self._engine, self.get_sources(), *a, **kw)

    @collect
    def get_insar_scenes(self, tmin=None, tmax=None):
        return lambda gen, *a, **kw: gen.get_insar_scenes(
            self._engine, self.get_sources(), *a, **kw)

    @collect
    def get_gnss_campaigns(self, tmin=None, tmax=None):
        return lambda gen, *a, **kw: gen.get_gnss_campaign(
            self._engine, self.get_sources(), *a, **kw)

    @collect
    def dump_data(self, path, tmin=None, tmax=None, overwrite=False):
        self.source_generator.dump_data(path)

        meta_dir = op.join(path, 'meta')
        util.ensuredir(meta_dir)

        model.station.dump_stations(
            self.get_stations(), op.join(meta_dir, 'stations.txt'))
        model.station.dump_kml(
            self.get_stations(), op.join(meta_dir, 'stations.kml'))

        dump_readme(path)

        def dump_data(gen, *a, **kw):
            logger.info('Creating files from %s...' % gen.__class__.__name__)
            return gen.dump_data(self._engine, self.get_sources(), *a, **kw)

        return dump_data

    @collect
    def _get_time_ranges(self):
        return lambda gen: [gen.get_time_range(self.get_sources())]

    def get_time_range(self):
        ranges = num.array(self._get_time_ranges())
        return ranges.min(), ranges.max()

    def get_pile(self, tmin=None, tmax=None):
        p = pile.Pile()

        trf = pile.MemTracesFile(None, self.get_waveforms(tmin, tmax))
        p.add_file(trf)
        return p

    def make_map(self, filename):
        logger.info('Plotting scenarios\' map...')
        draw_scenario_gmt(self, filename)

    def draw_map(self, fn):
        from pyrocko import automap

        lat, lon = self.get_center_latlon()
        radius = self.get_radius()

        m = automap.Map(
            width=30.,
            height=30.,
            lat=lat,
            lon=lon,
            radius=radius,
            show_topo=True,
            show_grid=True,
            show_rivers=True,
            color_wet=(216, 242, 254),
            color_dry=(238, 236, 230)
            )

        self.source_generator.add_map_artists(m)

        sources = self.get_sources()
        for gen in self.target_generators:
            gen.add_map_artists(self.get_engine(), sources, m)

        # for patch in self.get_insar_patches():
        #     symbol_size = 50.
        #     coords = num.array(patch.get_corner_coordinates())
        #     m.gmt.psxy(in_rows=num.fliplr(coords),
        #                L=True,
        #                *m.jxyr)

        m.save(fn)

    @classmethod
    def initialize(
            cls, path,
            center_lat=None, center_lon=None, radius=None,
            targets=AVAILABLE_TARGETS,
            force=False):
        import os.path as op

        if op.exists(path) and not force:
            raise CannotCreate('Directory %s alread exists! May use force?'
                               % path)

        util.ensuredir(path)
        fn = op.join(path, 'scenario.yml')
        logger.debug('Writing new scenario to %s' % fn)

        scenario = cls()
        scenario.target_generators.extend([t() for t in targets])

        for gen in scenario.target_generators:
            gen.update_hierarchy(scenario)

        scenario.center_lat = center_lat
        scenario.center_lon = center_lon
        scenario.radius = radius

        scenario.dump(filename=fn)

        return scenario


def draw_scenario_gmt(generator, fn):
    return generator.draw_map(fn)


def dump_readme(path):
    readme = '''# Pyrocko Earthquake Scenario

The directory structure of a scenario is layed out as follows:

## Map of the scenario
A simple map is generated from `pyrocko.automap` in map.pdf

## Earthquake Sources

Can be found as events.txt and sources.yml hosts the pyrocko.gf sources.

## Folder `meta`

Contains stations.txt and StationXML files for waveforms as well as KML data.
The responses are flat with gain of 1.0 at 1.0 Hz.

## Folder `waveforms`

Waveforms as mini-seed are stored here, segregated into days.

## Folder `gnss`

The GNSS campaign.yml is living here.
Use `pyrocko.guts.load(filename='campaign.yml)` to load the campaign.

## Folder `insar`

Kite InSAR scenes for ascending and descending tracks are stored there.
Use `kite.Scene.load(<filename>)` to inspect the scenes.

'''
    fn = op.join(path, 'README.md')
    with open(fn, 'w') as f:
        f.write(readme)

    return [fn]
