from builtins import range
import hashlib
import math
import logging
import numpy as num

from os import path as op
from functools import reduce

from pyrocko.guts import StringChoice, Float
from pyrocko import gf, model, util, trace, io
from pyrocko.io_common import FileSaveError

from .station import StationGenerator, RandomStationGenerator
from .base import TargetGenerator, NoiseGenerator

DEFAULT_STORE_ID = 'global_2s'

logger = logging.getLogger('pyrocko.scenario.targets.waveform')
guts_prefix = 'pf.scenario'


class WaveformNoiseGenerator(NoiseGenerator):

    def get_time_increment(self, deltat):
        return deltat * 1024

    def get_intersecting_snippets(self, deltat, codes, tmin, tmax):
        raise NotImplemented()

    def add_noise(self, tr):
        for ntr in self.get_intersecting_snippets(
                tr.deltat, tr.nslc_id, tr.tmin, tr.tmax):
            tr.add(ntr)


class WhiteNoiseGenerator(WaveformNoiseGenerator):

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


class WaveformGenerator(TargetGenerator):

    station_generator = StationGenerator.T(
        default=RandomStationGenerator.D(),
        help='The StationGenerator for creating the stations.')

    noise_generator = WaveformNoiseGenerator.T(
        default=WhiteNoiseGenerator.D(),
        help='Add Synthetic noise on the waveforms.')

    store_id = gf.StringID.T(
        default=DEFAULT_STORE_ID,
        help='The GF store to use for forward-calculations.')

    seismogram_quantity = StringChoice.T(
        choices=['displacement', 'velocity', 'acceleration', 'counts'],
        default='displacement')

    vmin_cut = Float.T(
        default=2000.,
        help='Minimum velocity to seismic velicty to consider in the model.')
    vmax_cut = Float.T(
        default=8000.,
        help='Maximum velocity to seismic velicty to consider in the model.')

    fmin = Float.T(
        default=0.01,
        help='Minimum frequency/wavelength to resolve in the'
             ' synthetic waveforms.')

    def get_stations(self):
        return self.station_generator.get_stations()

    def get_targets(self):
        targets = []
        for station in self.get_stations():
            channel_data = []
            channels = station.get_channels()
            if channels:
                for channel in channels:
                    channel_data.append([
                        channel.name,
                        channel.azimuth,
                        channel.dip])

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
                    store_id=self.store_id,
                    optimization='enable',
                    interpolation='nearest_neighbor',
                    azimuth=c_azi,
                    dip=c_dip)

                targets.append(target)

        return targets

    def get_time_range(self, sources):
        dmin, dmax = self.station_generator.get_distance_range(sources)

        times = num.array([source.time for source in sources],
                          dtype=num.float)

        tmin_events = num.min(times)
        tmax_events = num.max(times)

        tmin = tmin_events + dmin / self.vmax_cut - 10.0 / self.fmin
        tmax = tmax_events + dmax / self.vmin_cut + 10.0 / self.fmin

        return tmin, tmax

    def get_codes_to_deltat(self, engine, sources):
        deltats = {}
        for source in sources:
            for target in self.get_targets():
                deltats[target.codes] = engine.get_store(
                    target.store_id).config.deltat

        return deltats

    def get_useful_time_increment(self, engine, sources):
        _, dmax = self.station_generator.get_distance_range(sources)
        tinc = dmax / self.vmin_cut + 2.0 / self.fmin

        deltats = set(self.get_codes_to_deltat(engine, sources).values())

        deltat = reduce(util.lcm, deltats)
        tinc = int(round(tinc / deltat)) * deltat
        return tinc

    def get_waveforms(self, engine, sources, tmin=None, tmax=None):
        trs = {}

        tmin_all, tmax_all = self.get_time_range(sources)
        tmin = tmin if tmin is not None else tmin_all
        tmax = tmax if tmax is not None else tmax_all
        tts = util.time_to_str

        for nslc, deltat in self.get_codes_to_deltat(engine, sources).items():
            tr_tmin = int(round(tmin / deltat)) * deltat
            tr_tmax = (int(round(tmax / deltat))-1) * deltat
            nsamples = int(round((tr_tmax - tr_tmin) / deltat)) + 1

            tr = trace.Trace(
                *nslc,
                tmin=tr_tmin,
                ydata=num.zeros(nsamples),
                deltat=deltat)

            self.noise_generator.add_noise(tr)

            trs[nslc] = tr

        logger.debug('Calculating waveforms between %s - %s...'
                     % (tts(tmin, format='%Y-%m-%d_%H-%M-%S'),
                        tts(tmax, format='%Y-%m-%d_%H-%M-%S')))

        for source in sources:
            targets = self.get_targets()
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

    def dump_data(self, engine, sources, path,
                  tmin=None, tmax=None, overwrite=False):
        fns = []
        fns.extend(
            self.dump_waveforms(engine, sources, path, tmin, tmax, overwrite))
        fns.extend(
            self.dump_responses(path))
        return fns

    def dump_waveforms(self, engine, sources, path,
                       tmin=None, tmax=None, overwrite=False):
        path_waveforms = op.join(path, 'waveforms')
        util.ensuredir(path_waveforms)

        path_traces = op.join(
            path_waveforms,
            '%(wmin_year)s',
            '%(wmin_month)s',
            '%(wmin_day)s',
            'waveform_%(network)s_%(station)s_'
            + '%(location)s_%(channel)s_%(tmin)s_%(tmax)s.mseed')

        tmin_all, tmax_all = self.get_time_range(sources)
        tmin = tmin if tmin is not None else tmin_all
        tmax = tmax if tmax is not None else tmax_all
        tts = util.time_to_str

        tinc = self.get_useful_time_increment(engine, sources)
        tmin = math.floor(tmin / tinc) * tinc
        tmax = math.ceil(tmax / tinc) * tinc

        nwin = int(round((tmax - tmin) / tinc))

        for iwin in range(nwin):
            tmin_win = max(tmin, tmin + iwin*tinc)
            tmax_win = min(tmax, tmin + (iwin+1)*tinc)

            if tmax_win <= tmin_win:
                continue

            trs = self.get_waveforms(engine, sources, tmin_win, tmax_win)

            try:
                io.save(
                    trs, path_traces,
                    additional=dict(
                        wmin_year=tts(tmin_win, format='%Y'),
                        wmin_month=tts(tmin_win, format='%m'),
                        wmin_day=tts(tmin_win, format='%d'),
                        wmin=tts(tmin_win, format='%Y-%m-%d_%H-%M-%S'),
                        wmax_year=tts(tmax_win, format='%Y'),
                        wmax_month=tts(tmax_win, format='%m'),
                        wmax_day=tts(tmax_win, format='%d'),
                        wmax=tts(tmax_win, format='%Y-%m-%d_%H-%M-%S')),
                    overwrite=overwrite)
            except FileSaveError as e:
                logger.debug('Waveform exists %s' % e)

        return [path_waveforms]

    def dump_responses(self, path):
        from pyrocko.io import stationxml

        logger.debug('Writing out StationXML...')

        path_responses = op.join(path, 'meta')
        util.ensuredir(path_responses)
        fn_stationxml = op.join(path_responses, 'stations.xml')

        stations = self.station_generator.get_stations()
        sxml = stationxml.FDSNStationXML.from_pyrocko_stations(stations)

        sunit = {
            'displacement': 'M',
            'velocity': 'M/S',
            'acceleration': 'M/S**2',
            'counts': 'COUNTS'}[self.seismogram_quantity]

        response = stationxml.Response(
            instrument_sensitivity=stationxml.Sensitivity(
                value=1.,
                frequency=1.,
                input_units=stationxml.Units(sunit),
                output_units=stationxml.Units('COUNTS')),
            stage_list=[])

        for net, station, channel in sxml.iter_network_station_channels():
            channel.response = response

        sxml.dump_xml(filename=fn_stationxml)

        return [path_responses]

    def add_map_artists(self, engine, sources, automap):
        automap.add_stations(self.get_stations())
