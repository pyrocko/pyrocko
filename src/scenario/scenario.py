import numpy as num
import logging
import hashlib
from functools import reduce

from pyrocko.guts import StringChoice, Float
from pyrocko import gf, model, util, trace, moment_tensor, gmtpy

from .base import LocationGenerator, ScenarioError
from .generators import StationGenerator, RandomStationGenerator,\
    SourceGenerator, DCSourceGenerator,\
    InSARDisplacementGenerator,\
    NoiseGenerator, WhiteNoiseGenerator


logger = logging.getLogger('pyrocko.scenario')
guts_prefix = 'pf.scenario'


class ScenarioGenerator(LocationGenerator):
    station_generator = StationGenerator.T(
        default=RandomStationGenerator.D())

    insar_generator = InSARDisplacementGenerator.T(
        default=InSARDisplacementGenerator.D(),
        optional=True)

    source_generator = SourceGenerator.T(
        default=DCSourceGenerator.D())

    noise_generator = NoiseGenerator.T(
        default=WhiteNoiseGenerator.D())

    store_id = gf.StringID.T(
        optional=True)
    store_id_static = gf.StringID.T(
        optional=True)

    seismogram_quantity = StringChoice.T(
        choices=['displacement', 'velocity', 'acceleration', 'counts'],
        default='displacement')

    vmin_cut = Float.T(default=2000.)
    vmax_cut = Float.T(default=8000.)

    fmin = Float.T(default=0.01)

    def __init__(self, **kwargs):
        LocationGenerator.__init__(self, **kwargs)

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

    def get_stations(self):
        return self.station_generator.get_stations()

    def get_insar_patches(self):
        if self.insar_generator:
            return self.insar_generator.get_scene_patches()
        else:
            return None

    def get_store_id(self, source, station):
        if self.store_id is not None:
            return self.store_id
        else:
            return 'global_2s'

    def get_static_store_id(self):
        if self.store_id_static is not None:
            return self.store_id_static
        else:
            return 'static_local'

    def get_waveform_targets(self, source):
        targets = []
        for station in self.get_stations():
            channel_data = []
            channels = station.get_channels()
            if channels:
                for channel in channels:
                    channel_data.append([
                        channel.name, channel.azimuth, channel.dip])

            else:
                for c_name in ['BHZ', 'BHE', 'BHN']:
                    channel_data.append([
                        c_name,
                        model.guess_azimuth_from_name(c_name),
                        model.guess_dip_from_name(c_name)])

            for c_name, c_azi, c_dip in channel_data:

                target = gf.Target(
                    codes=(
                        station.network,
                        station.station,
                        station.location,
                        c_name),
                    quantity='displacement',
                    lat=station.lat,
                    lon=station.lon,
                    depth=station.depth,
                    store_id=self.get_store_id(source, station),
                    optimization='enable',
                    interpolation='nearest_neighbor',
                    azimuth=c_azi,
                    dip=c_dip)

                targets.append(target)

        return targets

    def get_insar_targets(self):
        targets = [s.get_target() for s in self.get_insar_patches()]

        for t in targets:
            t.store_id = self.get_static_store_id()

        return targets

    def get_targets(self, source):
        targets = self.get_waveform_targets(source)
        targets.extend(self.get_insar_targets())
        return targets

    def get_sources(self):
        return self.source_generator.get_sources()

    def get_station_distance_range(self):
        dists = []
        for source in self.get_sources():
            for station in self.get_stations():
                dists.append(
                    source.distance_to(station))

        return num.min(dists), num.max(dists)

    def get_time_range(self):
        dmin, dmax = self.get_station_distance_range()

        times = num.array(
            [source.time for source in self.get_sources()], dtype=num.float)

        tmin_events = num.min(times)
        tmax_events = num.max(times)

        tmin = tmin_events + dmin / self.vmax_cut - 10.0 / self.fmin
        tmax = tmax_events + dmax / self.vmin_cut + 10.0 / self.fmin

        return tmin, tmax

    def get_engine(self):
        return self._engine

    def get_codes_to_deltat(self):
        engine = self.get_engine()

        deltats = {}
        for source in self.get_sources():
            for target in self.get_waveform_targets(source):
                deltats[target.codes] = engine.get_store(
                    target.store_id).config.deltat

        return deltats

    def get_useful_time_increment(self):
        _, dmax = self.get_station_distance_range()
        tinc = dmax / self.vmin_cut + 2.0 / self.fmin

        deltats = set(self.get_codes_to_deltat().values())

        deltat = reduce(util.lcm, deltats)
        tinc = int(round(tinc / deltat)) * deltat
        return tinc

    def get_relevant_sources(self, tmin, tmax):
        dmin, dmax = self.get_station_distance_range()
        tmin_events = tmin - dmax / self.vmin_cut - 1.0 / self.fmin
        tmax_events = tmax - dmin / self.vmax_cut + 1.0 / self.fmin

        return [source for source in self.get_sources()
                if tmin_events <= source.time and source.time <= tmax_events]

    def get_waveforms(self, tmin, tmax):
        logger.info('Calculating waveforms...')
        engine = self.get_engine()

        trs = {}

        for nslc, deltat in self.get_codes_to_deltat().items():
            tr_tmin = int(round(tmin / deltat)) * deltat
            tr_tmax = (int(round(tmax / deltat))-1) * deltat
            n = int(round((tr_tmax - tr_tmin) / deltat)) + 1

            tr = trace.Trace(
                nslc[0], nslc[1], nslc[2], nslc[3],
                tmin=tr_tmin,
                ydata=num.zeros(n),
                deltat=deltat)

            self.noise_generator.add_noise(tr)

            trs[nslc] = tr

        for source in self.get_relevant_sources(tmin, tmax):
            targets = self.get_waveform_targets(source)
            resp = engine.process(source, targets)
            for _, target, tr in resp.iter_results():
                resp = self.get_transfer_function(target.codes)
                if resp:
                    tr = tr.transfer(transfer_function=resp)

                trs[target.codes].add(tr)

        return list(trs.values())

    def get_transfer_function(self, codes):
        if self.seismogram_quantity == 'displacement':
            return None
        elif self.seismogram_quantity == 'velocity':
            return trace.DifferentiationResponse(1)
        elif self.seismogram_quantity == 'acceleration':
            return trace.DifferentiationResponse(2)
        elif self.seismogram_quantity == 'counts':
            raise NotImplemented()

    def get_insar_scenes(self, tmin=None, tmax=None):
        engine = self.get_engine()
        logger.info('Calculating InSAR displacement...')

        scenario_tmin, scenario_tmax = self.get_time_range()
        if not tmin:
            tmin = scenario_tmin
        if not tmax:
            tmax = scenario_tmax

        targets = self.get_insar_targets()

        relevant_sources = self.get_relevant_sources(tmin, tmax)

        resp = engine.process(
            relevant_sources,
            targets,
            nthreads=0)

        scenes = [r.scene for r in resp.static_results()]
        for sc in scenes:
            hs = '%s%s%s' % (tmin, tmax, ''.join(
                r.dump() for r in relevant_sources))
            hs = hs.encode('utf8')

            sc.meta.time_master = float(tmin)
            sc.meta.time_slave = float(tmax)
            sc.meta.hash = hashlib.sha1(hs).hexdigest()

        return scenes


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
