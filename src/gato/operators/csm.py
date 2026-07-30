# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

import logging

from pyrocko import squirrel, guts, util
from pyrocko.squirrel.model import get_selection_args
from pyrocko.gato.array import deduplicate_locations, SensorArray

from .base import GatoOperator

logger = logging.getLogger('gato.operators.csm')

guts_prefix = 'gato'


class ArrayProcessingSetup:
    __slots__ = [
        'array',
        'array_incarnation',
        'mapping',
        'generic_delay_table']

    def __init__(self, array, array_incarnation, mapping, generic_delay_table):
        self.array = array
        self.array_incarnation = array_incarnation
        self.mapping = mapping
        self.generic_delay_table = generic_delay_table

    def describe(self):
        return '''
array incarnation: %s
mapping: %s
gdt: %s''' % (
            self.array_incarnation.summary,
            self.mapping.describe(),
            self.generic_delay_table.describe())


class CSMOperator(GatoOperator):
    name = guts.String.T(default='csm')
    in_codes = guts.List.T(squirrel.CodesNSLCE.T())
    downsampling_deltat = guts.Duration.T(optional=True)
    whitening_bandwidth = guts.Float.T(optional=True)
    time_normalization_deltat = guts.Duration.T(optional=True)
    time_window = guts.Duration.T(optional=True)
    nsubwindows = guts.Int.T(default=10)
    sample_rate_min = guts.Float.T(optional=True)
    deduplicate_distance_cutoff = guts.Float.T(optional=True)
    sensor_arrays = guts.List.T(SensorArray.T())

    @property
    def kind_requires(self):
        return ('waveform', 'channel')

    def get_effective_time_window(self):
        return self.time_window or (20.0 / self.frequency_min)

    def get_sensor_arrays(self):
        if self.sensor_arrays:
            return self.sensor_arrays

        return (
            self.get_squirrel().get_sensor_arrays()
            or [SensorArray(name='array0', codes=['*.*.*.*.*'])])

    def get_outlets(self):
        outlets = []
        for array in self.get_sensor_arrays():
            outlets.extend(self.get_outlets_for_array(array))
        return outlets

    def _update_mappings_specific(self, added, removed):
        if not added and not removed:
            return False

        setups = []
        for array in self.get_sensor_arrays():
            incarnation = array.get_incarnation(
                self._input,
                codes=self.in_codes or None,
                deduplicate=False)

            in_codes = self.codes_projection.filter(incarnation.codes)
            for k, in_codes_group in util.group_by(
                    self.codes_projection.group_key, in_codes).items():

                out_codes = self.codes_projection.project(
                    self, in_codes_group, self.get_outlets_for_array(array))

                mapping = squirrel.operators.base.CodesMapping()

                mapping.group_key = k
                mapping.in_codes = tuple(in_codes_group)
                mapping.in_codes_set = set(in_codes)
                mapping.out_codes = out_codes

                setups.append(self.make_array_processing_setup(
                    array, incarnation, mapping))

        mappings = dict(
            ((setup.array.name, setup.mapping.group_key), setup.mapping)
            for setup in setups)

        self._setups = setups
        self._mappings = mappings

        return True

    def get_setups(self):
        self.update_mappings()
        return self._setups

    def iter_csms(self, mapping, tmin=None, tmax=None, codes=None):

        in_codes = list(mapping.in_codes)

        codes_ok = None
        for kind in ('channel', 'waveform'):
            coverages = self._input.get_coverage(
                kind, codes=in_codes, tmin=tmin, tmax=tmax)

            codes_ok_this = set()

            for coverage in coverages:
                count = coverage.contiguous(tmin, tmax)
                if count >= 1:
                    codes_ok_this.add(coverage.codes)

            if codes_ok is None:
                codes_ok = codes_ok_this
            else:
                codes_ok &= codes_ok_this

        codes_ok = list(codes_ok)
        if not codes_ok:
            logger.warning(
                '%s: No channels with complete waveform and channel coverage '
                'for time window %s - %s.',
                self.name,
                util.time_to_str(tmin),
                util.time_to_str(tmax))
            return []

        channels = self._input.get_channels(
            codes=codes_ok, tmin=tmin, tmax=tmax)

        if self.deduplicate_distance_cutoff is None:
            channels_use = channels
        else:
            channels_use = deduplicate_locations(
                channels, distance_cutoff=self.deduplicate_distance_cutoff)

        codes_use = sorted(set(channel.codes for channel in channels_use))

        time_window = self.get_effective_time_window()

        chopper = self._input.chopper_waveforms(
            tmin=tmin,
            tmax=tmax,
            tinc=time_window,
            sample_rate_min=self.sample_rate_min,
            codes=codes_use,
            want_incomplete=False)

        def gen():
            for batch in chopper:

                if not batch.traces:
                    logger.warning(
                        '%s: No traces for time window %s - %s',
                        self.name,
                        util.time_to_str(batch.tmin),
                        util.time_to_str(batch.tmax))

                    yield batch, None, None, None, None
                    continue

                if len(batch.traces) != len(codes_use):
                    logger.warning(
                        '%s: Preprocessing failed for %i of %i traces.',
                        self.name,
                        len(codes_use) - len(batch.traces),
                        len(codes_use))

                carpet = batch.as_carpet(deltat=self.downsampling_deltat)

                if self.whitening_bandwidth is not None:
                    carpet.whiten(deltaf=self.whitening_bandwidth)

                if self.time_normalization_deltat is not None:
                    carpet.normalize(deltat=self.time_normalization_deltat)

                cspectrum_sum = None
                nsum = 0
                for subwindow in carpet.chopper(
                        tinc=time_window/self.nsubwindows):

                    frequency_delta, ntrans, cspectrum = \
                        subwindow.get_cross_spectrum()

                    if cspectrum_sum is None:
                        cspectrum_sum = cspectrum
                    else:
                        cspectrum_sum += cspectrum

                    nsum += 1

                cspectrum_sum /= nsum

                yield batch, carpet, frequency_delta, cspectrum_sum, codes_use

        try:
            nwindows = len(chopper)
            return util.GeneratorWithLen(gen(), nwindows)

        except TypeError:
            return gen()

    def make_carpets(self, tmin=None, tmax=None, codes=None):
        raise NotImplementedError()

    def chopper_carpets(
            self, obj=None, tmin=None, tmax=None, time=None, codes=None,
            tinc=None, tpad=0., want_incomplete=True, snap_window=False):

        tmin, tmax, codes = get_selection_args(
            squirrel.WAVEFORM, obj, tmin, tmax, time, codes)

        tmin_content, tmax_content = self.get_time_span(['waveform'])

        source_gen = util.iter_windows(
                    tmin=tmin,
                    tmax=tmax,
                    tinc=tinc,
                    tpad=tpad,
                    snap_window=snap_window,
                    tmin_content=tmin_content,
                    tmax_content=tmax_content)

        def gen():
            for batch in source_gen:
                carpets = self.make_carpets(
                    tmin=batch.tmin-tpad, tmax=batch.tmax+tpad, codes=codes)
                batch.carpets = carpets
                yield batch

        return util.GeneratorWithLen(gen(), len(source_gen))


__all__ = [
    'CSMOperator',
]
