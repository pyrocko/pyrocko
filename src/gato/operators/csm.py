# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

import logging

from pyrocko import squirrel, guts, util
from pyrocko.squirrel.model import get_selection_args
from pyrocko.gato.array import deduplicate_locations

from .base import GatoOperator

logger = logging.getLogger('gato.operators.csm')

guts_prefix = 'gato'


class CSMOperator(GatoOperator):

    in_codes = guts.List.T(squirrel.CodesNSLCE.T())
    downsampling_deltat = guts.Duration.T()
    whitening_bandwidth = guts.Float.T(optional=True)
    time_normalization_deltat = guts.Duration.T(optional=True)
    time_window = guts.Duration.T()
    nsubwindows = guts.Int.T(default=10)
    sample_rate_min = guts.Float.T(optional=True)
    deduplicate_distance_cutoff = guts.Float.T(optional=True)

    @property
    def kind_requires(self):
        return ('waveform', 'channel')

    def set_arrays(self, arrays):
        self._arrays = arrays
        self.update_mappings()

    def get_arrays(self):
        return self._arrays

    def _update_mappings_specific(self, added, removed):
        if not added and not removed:
            return False

        self._mappings = {}
        self._array_infos = {}

        arrays = self.get_arrays()
        for array in arrays:
            info = array.get_info(
                self._input,
                codes=self.in_codes or None,
                deduplicate=False)

            in_codes = info.codes
            out_codes = [squirrel.CodesNSLCE(
                '', array.name, '', out_channel)
                    for out_channel in sum(
                        self.get_out_channels().values(), start=[])]

            mapping = squirrel.operators.base.CodesMapping()

            mapping.in_codes = tuple(in_codes)
            mapping.in_codes_set = set(in_codes)
            mapping.out_codes = out_codes

            self._mappings[array.name] = mapping
            self._array_infos[array.name] = info

        return True

    def iter_csms(self, mapping, tmin=None, tmax=None, codes=None):

        in_codes = list(mapping.in_codes)

        codes_ok = None
        for kind in ('channel', 'waveform'):
            coverages = self._input.get_coverage(
                kind, codes=in_codes, tmin=tmin, tmax=tmax)

            codes_ok_this = set()

            for coverage in coverages:
                count = coverage.contiguous(tmin, tmax)
                if count == 1:
                    codes_ok_this.add(coverage.codes)

            if codes_ok is None:
                codes_ok = codes_ok_this
            else:
                codes_ok &= codes_ok_this

        codes_ok = list(codes_ok)

        channels = self._input.get_channels(
            codes=codes_ok, tmin=tmin, tmax=tmax)

        channels_use = deduplicate_locations(
            channels, distance_cutoff=self.deduplicate_distance_cutoff)

        codes_use = sorted(set(channel.codes for channel in channels_use))

        chopper = self._input.chopper_waveforms(
            tmin=tmin,
            tmax=tmax,
            tinc=self.time_window,
            sample_rate_min=self.sample_rate_min,
            codes=codes_use,
            want_incomplete=False)

        def gen():
            for batch in chopper:

                if not batch.traces:
                    yield batch, None, None, None, None
                    continue

                if len(batch.traces) != len(codes_use):
                    logger.warning(
                        'Preprocessing failed for %i of %i traces.',
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
                        tinc=self.time_window/self.nsubwindows):

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
