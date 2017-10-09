import math
import os
import errno
import time
import os.path as op
import hashlib
import logging
import tarfile
from datetime import datetime

import numpy as num

from pyrocko.guts import (Object, Int, Bool, Float, Timestamp, Tuple,
                          StringChoice)
from pyrocko import orthodrome as od, trace, guts
from pyrocko import model, moment_tensor, gf, util, io, pile
from pyrocko.plot import gmtpy
from pyrocko.dataset import gshhg


logger = logging.getLogger('pyrocko.scenario.base')

guts_prefix = 'pf.scenario'

km = 1000.
d2r = num.pi/180.
N = 10000000

coastlines = None


def mtime(p):
    return os.stat(p).st_mtime


def get_gsshg():
    global coastlines
    if coastlines is None:
        logger.debug('Initialising GSHHG database.')
        coastlines = gshhg.GSHHG.intermediate()
    return coastlines


class ScenarioError(Exception):
    pass


def is_on_land(lat, lon, method='topo'):
    if method == 'topo':
        from pyrocko import topo
        return topo.elevation(lat, lon) > 0.

    elif method == 'coastlines':
        logger.debug('Testing %.4f %.4f' % (lat, lon))
        return get_gsshg().is_point_on_land(lat, lon)


def random_lat(rstate, lat_min=-90., lat_max=90.):
    lat_min_ = 0.5*(math.sin(lat_min * math.pi/180.)+1.)
    lat_max_ = 0.5*(math.sin(lat_max * math.pi/180.)+1.)
    return math.asin(rstate.uniform(lat_min_, lat_max_)*2.-1.)*180./math.pi


class Generator(Object):
    seed = Int.T(optional=True)

    def __init__(self, **kwargs):
        Object.__init__(self, **kwargs)
        self._seed = None
        self._parent = None
        self.update_hierarchy()
        self._retry_offset = 0

    def retry(self):
        self.clear()
        self._retry_offset += 1
        for val in self.T.ivals(self):
            if isinstance(val, Generator):
                val.retry()

    def clear(self):
        self._seed = None

    def hash(self):
        return hashlib.sha1(
            self.dump() + '\n\n%i' % self._retry_offset).hexdigest()

    def get_seed_offset(self):
        return int(self.hash(), base=16) % N

    def update_hierarchy(self, parent=None):
        self._parent = parent
        for val in self.T.ivals(self):
            if isinstance(val, Generator):
                val.update_hierarchy(parent=self)

    def get_seed(self):
        if self._seed is None:
            if self.seed is None:
                if self._parent is not None:
                    self._seed = self._parent.get_seed()
                else:
                    self._seed = num.random.randint(N)
            elif self.seed == 0:
                self._seed = num.random.randint(N)
            else:
                self._seed = self.seed

        return self._seed + self.get_seed_offset()

    def get_rstate(self, i):
        return num.random.RandomState(self.get_seed() + i)


class LocationGenerator(Generator):

    avoid_water = Bool.T(
        default=True,
        help='Set whether wet areas should be avoided.')
    center_lat = Float.T(
        optional=True,
        help='Center latitude for the random locations in [deg].')
    center_lon = Float.T(
        optional=True,
        help='Center longitude for the random locations in [deg].')
    radius = Float.T(
        optional=True,
        help='Radius for distribution of random locations [m].')
    ntries = Int.T(
        default=500,
        help='Maximum number of tries to find a location satisifying all '
             'given constraints')

    def __init__(self, **kwargs):
        Generator.__init__(self, **kwargs)
        self._center_latlon = None

    def clear(self):
        Generator.clear(self)
        self._center_latlon = None

    def get_center_latlon(self):
        assert (self.center_lat is None) == (self.center_lon is None)

        if self._center_latlon is None:

            if self.center_lat is not None and self.center_lon is not None:
                self._center_latlon = self.center_lat, self.center_lon

            else:
                if self._parent:
                    self._center_latlon = self._parent.get_center_latlon()
                else:
                    rstate = self.get_rstate(0)
                    lat = random_lat(rstate)
                    lon = rstate.uniform(-180., 180.)
                    self._center_latlon = lat, lon

        return self._center_latlon

    def get_radius(self):
        if self.radius is not None:
            return self.radius
        elif self._parent is not None:
            return self._parent.get_radius()
        else:
            return None

    def get_latlon(self, i):
        rstate = self.get_rstate(i)
        for itry in xrange(self.ntries):
            radius = self.get_radius()
            if radius is None:
                lat = random_lat(rstate)
                lon = rstate.uniform(-180., 180.)
            else:
                lat_center, lon_center = self.get_center_latlon()
                while True:
                    north = rstate.uniform(-radius, radius)
                    east = rstate.uniform(-radius, radius)
                    if math.sqrt(north**2 + east**2) <= radius:
                        break

                lat, lon = od.ne_to_latlon(lat_center, lon_center, north, east)

            if not self.avoid_water or is_on_land(lat, lon):
                return lat, lon

        if self.avoid_water:
            sadd = ' (avoiding water)'

        raise ScenarioError('could not generate location%s' % sadd)


class StationGenerator(LocationGenerator):
    pass


class RandomStationGenerator(StationGenerator):

    nstations = Int.T(
        default=10,
        help='Number of randomly distributed stations.')

    def __init__(self, **kwargs):
        LocationGenerator.__init__(self, **kwargs)
        self._stations = None

    def clear(self):
        StationGenerator.clear(self)
        self._stations = None

    def nsl(self, istation):
        return '', 'S%03i' % (istation + 1), '',

    def get_stations(self):
        if self._stations is None:
            stations = []
            for istation in xrange(self.nstations):
                lat, lon = self.get_latlon(istation)

                net, sta, loc = self.nsl(istation)
                station = model.Station(
                    net, sta, loc,
                    lat=lat,
                    lon=lon)

                stations.append(station)

            self._stations = stations

        return self._stations


class ScenePatch(Object):
    lat_center = Float.T(
        help='Center latitude anchor.')
    lon_center = Float.T(
        help='center longitude anchor.')
    time_master = Timestamp.T(
        help='Timestamp of the master.')
    time_slave = Timestamp.T(
        help='Timestamp of the slave.')
    inclination = Float.T(
        help='Inclination of the satellite orbit towards equatorial plane.')
    apogee = Float.T(
        help='Apogee of the satellite in [m].')
    swath_width = Float.T(
        default=250 * km,
        help='Swath width in [m].')
    track_length = Float.T(
        help='Track length in [m].')
    incident_angle = Float.T(
        help='Near range incident angle in [deg].')
    resolution = Tuple.T(
        help='Resolution of raster in east x north [px].')
    track_direction = StringChoice.T(
        ['ascending', 'descending'],
        help='Orbit direction.')
    mask_water = Bool.T(
        help='Mask water bodies.')

    class SatelliteGeneratorTarget(gf.SatelliteTarget):

        def __init__(self, scene_patch, *args, **kwargs):
            gf.SatelliteTarget.__init__(self, *args, **kwargs)

            self.scene_patch = scene_patch

        def post_process(self, *args, **kwargs):
            resp = gf.SatelliteTarget.post_process(self, *args, **kwargs)

            from kite import Scene
            from kite.scene_io import SceneIO

            gen = self.scene_patch

            grid, _ = gen.get_grid()

            displacement = num.empty_like(grid)
            displacement.fill(num.nan)
            displacement[gen.get_mask()] = resp.result['displacement.los']

            theta, phi = gen.get_incident_angles()

            llLat, llLon = gen.get_ll_anchor()
            urLat, urLon = od.ne_to_latlon(llLat, llLon,
                                           gen.track_length, gen.width)
            dLon = num.abs(llLon - urLon) / gen.resolution[0]
            dLat = num.abs(llLat - urLat) / gen.resolution[1]

            io = SceneIO()
            io.container = {
                'phi': theta,    # Look orientation
                                 # counter-clockwise angle from east
                'theta': phi,  # Look elevation angle
                               # (up from horizontal in degree) 90deg North
                'displacement': displacement,  # Displacement towards LOS
                'frame': {
                    'llLon': llLon,  # Lower left corner latitude
                    'llLat': llLat,  # Lower left corner londgitude
                    'dLat': dLat,   # Pixel delta latitude
                    'dLon': dLon,   # Pixel delta longitude
                },
                # Meta information
                'meta': {
                    'title': 'Pyrocko Scenario Generator ({})'
                             .format(datetime.now()),
                    'orbit_direction': 'Ascending',
                    'satellite_name': 'Sentinel-1',
                    'wavelength': None,
                    'time_master': None,
                    'time_slave': None
                },
                # All extra information
                'extra': {}
            }

            scene = Scene()
            scene = Scene._import_from_dict(scene, io.container)
            scene.save('/tmp/test-%s.npz' % self.scene_patch.track_direction)
            resp.scene = scene

            return resp

    @property
    def width(self):
        track_shift = num.abs(num.cos(self.inclination*d2r)
                              * self.track_length)
        return self.swath_width + track_shift

    def get_ll_anchor(self):
        return od.ne_to_latlon(self.lat_center, self.lon_center,
                               -self.track_length/2, -self.width/2)

    def get_grid(self):
        '''Return relative positions of scatterer.

        :param track: Acquisition track, from `'asc'` or `'dsc'`.
        :type track: string
        '''
        easts = num.linspace(0, self.width,
                             self.resolution[0])
        norths = num.linspace(0, self.track_length,
                              self.resolution[1])

        return num.meshgrid(easts, norths)

    def get_mask_track(self):
        east_shifts, north_shifts = self.get_grid()
        norths = north_shifts[:, 0]
        track = num.abs(num.cos(self.inclination*d2r)) * norths

        track_mask = num.logical_and(
            east_shifts > track[:, num.newaxis],
            east_shifts < (track + self.swath_width)[:, num.newaxis])

        if self.track_direction == 'ascending':
            track_mask = num.fliplr(track_mask)

        return track_mask

    def get_mask_water(self):
        east_shifts, north_shifts = self.get_grid()

        east_shifts -= east_shifts[0, -1]/2
        north_shifts -= north_shifts[-1, -1]/2

        latlon = od.ne_to_latlon(self.lat_center, self.lon_center,
                                 north_shifts.ravel(), east_shifts.ravel())
        points = num.array(latlon).T
        mask_water = get_gsshg().get_land_mask(points)\
            .reshape(*east_shifts.shape)

        return mask_water

    def get_mask(self):
        mask_track = self.get_mask_track()
        if self.mask_water:
            mask_water = self.get_mask_water()
            return num.logical_and(mask_track, mask_water)
        return mask_track

    def get_incident_angles(self):
        east_shifts, north_shifts = self.get_grid()

        phi = north_shifts.copy()
        phi.fill(self.incident_angle*d2r - num.pi/2)

        east_shifts += num.arctan(self.incident_angle) * self.apogee
        theta = num.tan(self.apogee/east_shifts)

        return theta, phi

    def get_target(self):
        gE, gN = self.get_grid()
        mask = self.get_mask()

        east_shifts = gE[mask].ravel()
        north_shifts = gN[mask].ravel()
        llLat, llLon = self.get_ll_anchor()

        ncoords = east_shifts.size

        theta, phi = self.get_incident_angles()

        theta = theta[mask].ravel()
        phi = phi[mask].ravel()

        return self.SatelliteGeneratorTarget(
            scene_patch=self,
            lats=num.full(ncoords, fill_value=llLat),
            lons=num.full(ncoords, fill_value=llLon),
            east_shifts=east_shifts,
            north_shifts=north_shifts,
            theta=theta,
            phi=phi)


class SatelliteSceneGenerator(LocationGenerator):
    # https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/acquisition-modes/interferometric-wide-swath
    inclination = Float.T(
        default=98.2,
        help='Inclination of the satellite orbit towards equatorial plane'
             ' in [deg]. Defaults to Sentinel-1 (98.1 deg).')
    apogee = Float.T(
        default=693. * km,
        help='Apogee of the satellite in [m]. '
             'Defaults to Sentinel-1 (693 km).')
    swath_width = Float.T(
        default=250 * km,
        help='Swath width in [m]. '
             'Defaults to Sentinel-1 Interfeometric Wide Swath Mode (IW).'
             ' (IW; 250 km).')
    track_length = Float.T(
        default=150 * km,
        help='Track length in [m]. Defaults to 200 km.')
    incident_angle = Float.T(
        default=29.1,
        help='Near range incident angle in [deg]. Defaults to 29.1 deg;'
             ' Sentinel IW mode.')
    resolution = Tuple.T(
        default=(250, 250),
        help='Resolution of raster in east x north [px].')
    mask_water = Bool.T(
        default=True,
        help='Mask out water bodies.')

    def get_scene_patches(self):
        lat_center, lon_center = self.get_center_latlon()

        scene_patches = []
        for direction in ('ascending', 'descending'):
            patch = ScenePatch(
                lat_center=lat_center,
                lon_center=lon_center,
                time_master='2017-01-01 00:00:00',
                time_slave='2017-01-12 00:00:00',
                inclination=self.inclination,
                apogee=self.apogee,
                swath_width=self.swath_width,
                track_length=self.track_length,
                incident_angle=self.incident_angle,
                resolution=self.resolution,
                track_direction=direction,
                mask_water=self.mask_water)
            scene_patches.append(patch)
        return scene_patches


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
        for ievent in xrange(self.nevents):
            sources.append(self.get_source(ievent))

        return sources


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
        m = hashlib.sha1('%e %i %s.%s.%s.%s' % ((deltat, iw) + codes))
        return int(m.hexdigest(), base=16) % 1000

    def get_intersecting_snippets(self, deltat, codes, tmin, tmax):
        tinc = self.get_time_increment(deltat)
        iwmin = int(math.floor(tmin / tinc))
        iwmax = int(math.floor(tmax / tinc))

        trs = []
        for iw in xrange(iwmin, iwmax+1):
            seed_offset = self.get_seed_offset2(deltat, iw, codes)
            rstate = self.get_rstate(seed_offset)

            n = int(round(tinc // deltat))

            trs.append(trace.Trace(
                codes[0], codes[1], codes[2], codes[3],
                deltat=deltat,
                tmin=iw*tinc,
                ydata=rstate.normal(loc=0, scale=self.scale, size=n)))

        return trs


class ScenarioGenerator(LocationGenerator):
    station_generator = StationGenerator.T(
        default=StationGenerator.D())

    satellite_generator = SatelliteSceneGenerator.T(
        default=SatelliteSceneGenerator.D(),
        optional=True)

    source_generator = SourceGenerator.T(
        default=DCSourceGenerator.D())

    noise_generator = NoiseGenerator.T(
        default=WhiteNoiseGenerator.D())

    store_id = gf.StringID.T(optional=True)
    static_store_id = gf.StringID.T(optional=True)

    seismogram_quantity = StringChoice.T(
        choices=['displacement', 'velocity', 'acceleration', 'counts'],
        default='displacement')

    vmin_cut = Float.T(default=2000.)
    vmax_cut = Float.T(default=8000.)

    fmin = Float.T(default=0.01)

    def __init__(self, **kwargs):
        LocationGenerator.__init__(self, **kwargs)

        for itry in xrange(self.ntries):

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

    def get_scene_patches(self):
        if self.satellite_generator:
            return self.satellite_generator.get_scene_patches()
        else:
            return None

    def get_store_id(self, source, station):
        if self.store_id is not None:
            return self.store_id
        else:
            return 'global_2s'

    def get_static_store_id(self):
        if self.static_store_id is not None:
            return self.static_store_id
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

    def get_satellite_targets(self, source):
        return [s.get_target() for s in self.get_scene_patches()]

    def get_targets(self, source):
        targets = self.get_waveform_targets(source)
        targets.extend(self.get_satellite_targets(source))
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

    def get_waveforms(self, tmin, tmax):
        engine = self.get_engine()

        dmin, dmax = self.get_station_distance_range()

        tmin_events = tmin - dmax / self.vmin_cut - 1.0 / self.fmin
        tmax_events = tmax - dmin / self.vmax_cut + 1.0 / self.fmin

        trs = {}

        for nslc, deltat in self.get_codes_to_deltat().iteritems():
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

        relevant_sources = [
            source for source in self.get_sources()
            if tmin_events <= source.time and source.time <= tmax_events]

        for source in relevant_sources:
            targets = self.get_waveform_targets(source)
            resp = engine.process(source, targets)
            for _, target, tr in resp.iter_results():
                resp = self.get_transfer_function(target.codes)
                if resp:
                    tr = tr.transfer(transfer_function=resp)

                trs[target.codes].add(tr)

        return trs.values()

    def get_transfer_function(self, codes):
        if self.seismogram_quantity == 'displacement':
            return None
        elif self.seismogram_quantity == 'velocity':
            return trace.DifferentiationResponse(1)
        elif self.seismogram_quantity == 'acceleration':
            return trace.DifferentiationResponse(2)
        elif self.seismogram_quantity == 'counts':
            raise NotImplemented()

    def get_displacement_scenes(self):
        engine = self.get_engine()

        scene_patches = self.satellite_generator.get_scene_patches()

        relevant_sources = [source for source in self.get_sources()]

        targets = [p.get_target() for p in scene_patches]

        for t in targets:
            t.store_id = self.get_static_store_id()

        resp = engine.process(relevant_sources, targets,
                              nprocs=0)

        return [r.scene for r in resp.static_results()]


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

    m.save(fn)


class ScenarioCollectionItem(Object):
    scenario_id = gf.StringID.T()
    time_created = Timestamp.T()

    def __init__(self, **kwargs):
        Object.__init__(self, **kwargs)
        self._path = None
        self._pile = None
        self._engine = None

    def set_base_path(self, path):
        self._path = path

    def init_modelling(self, engine):
        self._engine = engine

    def get_path(self, *entry):
        return op.join(*((self._path,) + entry))

    def get_generator(self):
        generator = guts.load(filename=self.get_path('generator.yaml'))
        generator.init_modelling(self._engine)
        return generator

    def have_waveforms(self, tmin, tmax):
        p = self.get_waveform_pile()
        trs_have = p.all(
            tmin=tmin, tmax=tmax, load_data=False, degap=False)

        return any(tr.data_len() > 0 for tr in trs_have)

    def get_waveform_pile(self):
        if self._pile is None:
            path_waveforms = self.get_path('waveforms')
            util.ensuredir(path_waveforms)
            fns = util.select_files(
                [path_waveforms], show_progress=False)
            self._pile = pile.Pile()
            if fns:
                self._pile.load_files(
                    fns, fileformat='mseed', show_progress=False)

        return self._pile

    def make_map(self, path_pdf):
        draw_scenario_gmt(self.get_generator(), path_pdf)

    def get_map(self, format='pdf'):
        path_pdf = self.get_path('map.pdf')

        if not op.exists(path_pdf):
            self.make_map(path_pdf)

        path = self.get_path('map.%s' % format)

        outdated = op.exists(path) and mtime(path) < mtime(path_pdf)
        if not op.exists(path) or outdated:
            gmtpy.convert_graph(path_pdf, path)

        return path

    def ensure_waveforms(self, tmin, tmax):
        path_waveforms = self.get_path('waveforms')
        path_traces = op.join(
            path_waveforms,
            '%(wmin_year)s',
            '%(wmin_month)s',
            '%(wmin_day)s',
            'waveform_%(network)s_%(station)s_'
            + '%(location)s_%(channel)s_%(tmin)s_%(tmax)s.mseed')

        generator = self.get_generator()
        tmin_all, tmax_all = generator.get_time_range()

        tmin = tmin if tmin is not None else tmin_all
        tmax = tmax if tmax is not None else tmax_all

        tinc = generator.get_useful_time_increment()

        tmin = math.floor(tmin / tinc) * tinc
        tmax = math.ceil(tmax / tinc) * tinc

        nwin = int(round((tmax - tmin) / tinc))

        p = self.get_waveform_pile()

        for iwin in xrange(nwin):
            tmin_win = max(tmin, tmin + iwin*tinc)
            tmax_win = min(tmax, tmin + (iwin+1)*tinc)
            if tmax_win <= tmin_win:
                continue

            if self.have_waveforms(tmin_win, tmax_win):
                continue

            trs = generator.get_waveforms(tmin_win, tmax_win)
            tts = util.time_to_str

            fns = io.save(
                trs, path_traces,
                additional=dict(
                    wmin_year=tts(tmin_win, format='%Y'),
                    wmin_month=tts(tmin_win, format='%m'),
                    wmin_day=tts(tmin_win, format='%d'),
                    wmin=tts(tmin_win, format='%Y-%m-%d_%H-%M-%S'),
                    wmax_year=tts(tmax_win, format='%Y'),
                    wmax_month=tts(tmax_win, format='%m'),
                    wmax_day=tts(tmax_win, format='%d'),
                    wmax=tts(tmax_win, format='%Y-%m-%d_%H-%M-%S')))

            if fns:
                p.load_files(fns, fileformat='mseed', show_progress=False)

        return p

    def get_time_range(self):
        return self.get_generator().get_time_range()

    def get_archive(self):
        path_tar = self.get_path('archive.tar')
        if not op.exists(path_tar):
            path_base = self.get_path()
            path_waveforms = self.get_path('waveforms')

            tmin, tmax = self.get_time_range()
            self.ensure_waveforms(tmin, tmax)
            fns = util.select_files(
                [path_waveforms], show_progress=False)

            f = tarfile.TarFile(path_tar, 'w')
            for fn in fns:
                fna = fn[len(path_base)+1:]
                f.add(fn, fna)

            f.close()

        return path_tar


class ScenarioCollection(object):

    def __init__(self, path, engine):
        self._scenario_suffix = 'scenario'
        self._path = path
        util.ensuredir(self._path)
        self._engine = engine
        self._load_scenarios()

    def _load_scenarios(self):
        scenarios = []
        base_path = self.get_path()
        for path_entry in os.listdir(base_path):
            scenario_id, suffix = op.splitext(path_entry)
            if suffix == '.' + self._scenario_suffix:
                path = op.join(base_path, path_entry, 'scenario.yaml')
                scenario = guts.load(filename=path)
                assert scenario.scenario_id == scenario_id
                scenario.set_base_path(op.join(base_path, path_entry))
                scenario.init_modelling(self._engine)
                scenarios.append(scenario)

        self._scenarios = scenarios
        self._scenarios.sort(key=lambda s: s.time_created)

    def get_path(self, scenario_id=None, *entry):
        if scenario_id is not None:
            return op.join(self._path, '%s.%s' % (
                scenario_id, self._scenario_suffix), *entry)
        else:
            return self._path

    def add_scenario(self, scenario_id, scenario_generator):

        if scenario_generator.seed is None:
            scenario_generator = guts.clone(scenario_generator)
            scenario_generator.seed = random.randint(1, 2**32-1)

        path = self.get_path(scenario_id)
        try:
            os.mkdir(path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                raise ScenarioError(
                    'scenario id is already in use: %s' % scenario_id)
            else:
                raise

        scenario = ScenarioCollectionItem(
            scenario_id=scenario_id,
            time_created=time.time())

        scenario_path = self.get_path(scenario_id, 'scenario.yaml')
        guts.dump(scenario, filename=scenario_path)

        generator_path = self.get_path(scenario_id, 'generator.yaml')
        guts.dump(scenario_generator, filename=generator_path)

        scenario.set_base_path(self.get_path(scenario_id))
        scenario.init_modelling(self._engine)

        self._scenarios.append(scenario)

    def list_scenarios(self, ilo=None, ihi=None):
        return self._scenarios[ilo:ihi]

    def get_scenario(self, scenario_id):
        for scenario in self._scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario

        raise KeyError(scenario_id)


__all__ = '''
    ScenarioError
    Generator
    LocationGenerator
    StationGenerator
    RandomStationGenerator
    SatelliteSceneGenerator
    SourceGenerator
    DCSourceGenerator
    NoiseGenerator
    WhiteNoiseGenerator
    ScenarioGenerator
    ScenarioCollectionItem
    ScenarioCollection
    draw_scenario_gmt
'''.split()
