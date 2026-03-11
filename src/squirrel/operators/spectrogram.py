# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

import logging
import math
import hashlib
from collections import OrderedDict

import numpy as num

import matplotlib.pyplot as plt

from pyrocko import util, guts, plot
from pyrocko.squirrel import model, error
from . import base
from ...carpet import Carpet

logger = logging.getLogger('psq.ops.spectrogram')


class LRU:
    def __init__(self, func, maxsize=128):
        self.cache = OrderedDict()
        self.func = func
        self.maxsize = maxsize

    def __call__(self, *args):
        if args in self.cache:
            self.cache.move_to_end(args)
            return self.cache[args]

        result = self.func(*args)
        self.cache[args] = result
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

        return result


def is_pow2(x):
    return round(math.log(x) / math.log(2.0), 7) % 1.0 == 0.0


def antialiasing_taper(n, corner=0.9):
    h = num.ones(n)
    f = num.arange(n) / (n-1)
    mask = f > corner
    h[mask] *= (0.5 + 0.5 * num.cos(
        (f[mask] - corner) / (1.0 - corner) * num.pi))**2
    # h[mask] = 0.0
    return h


def cos_taper(x):
    return 0.5 - 0.5 * num.cos(x)


class Pow2Windowing(guts.Object):
    nblock = guts.Int.T(default=1024)
    nlevels = guts.Int.T(default=4)
    weighting_exponent = guts.Int.T(default=2)

    def __init__(self, **kwargs):
        guts.Object.__init__(self, **kwargs)
        if not is_pow2(self.nblock):
            raise ValueError(
                'Pow2Windowing: `nblock` must be a power of 2.')

        self.nfrequencies = self.nblock // 2 + 1
        self.taper_td = num.hanning(self.nblock)
        self.taper_fd_2 = antialiasing_taper(self.nfrequencies // 2 + 1, 0.9)
        self.taper_td_2_inv_center = 1.0 \
            / num.hanning(self.nblock // 2)[
                self.nblock // 8: 3 * self.nblock // 8]

    def hash(self):
        return hashlib.sha256(b'%i,%i' % (self.nblock, self.nlevels))

    def time_increment(self, deltat, ilevel=0):
        if ilevel < 0:
            ilevel = self.nlevels + ilevel

        if not (0 <= ilevel < self.nlevels):
            raise ValueError(
                'Pow2Windowing: invalid `ilevel`: %i' % ilevel)

        return deltat * (self.nblock // 2) * 2**ilevel

    def iwindow_range(self, deltat, tmin, tmax, ilevel):
        itmin = int(round(tmin/deltat))
        itmax = int(round(tmax/deltat))
        tinc = self.time_increment(deltat, ilevel)
        itinc = int(round(tinc / deltat))
        imin = (itmin + (itinc-1)) // itinc
        imax = (itmax - 1) // itinc + 1
        iadd = 2 if ilevel > 0 else 0
        return num.arange(imin-iadd, imax+iadd)

    def frequencies_and_weights(self, deltat):
        eps1 = 1e-4
        eps2 = 1e-7
        nfrequencies = self.nfrequencies
        nfrequencies_all \
            = nfrequencies + nfrequencies // 2 * (self.nlevels - 1)

        frequencies_all = num.zeros(nfrequencies_all)
        ifmin = 0
        for ilevel in range(self.nlevels-1, -1, -1):
            if ilevel == self.nlevels-1:
                ifmax = ifmin + nfrequencies
                fmin_fill = 0.0
            else:
                ifmax = ifmin + nfrequencies // 2
                df = 1.0 / (deltat*self.nblock * 2**ilevel)
                fmin_fill = df * ((nfrequencies // 2) + 1)

            fmax = (0.5 / deltat) / 2**ilevel
            frequencies_all[ifmin:ifmax] = num.linspace(
                fmin_fill, fmax, ifmax - ifmin)

            ifmin = ifmax

        log_frequencies_all = num.zeros_like(frequencies_all)
        log_frequencies_all[1:] = num.log(frequencies_all[1:])

        norm_log_frequencies_all = \
            (log_frequencies_all[1:] - log_frequencies_all[1]) \
            / (log_frequencies_all[-1] - log_frequencies_all[1])

        weights = num.zeros((self.nlevels, frequencies_all.size))
        for ilevel in range(self.nlevels):
            fmin = 1.0 / (deltat*self.nblock * 2**ilevel)
            fmax = (0.5 / deltat) / 2**ilevel
            fmask = num.logical_and(
                fmin < frequencies_all, frequencies_all < fmax)

            # weights[ilevel, fmask] = cos_taper(
            #     num.sqrt((frequencies_all[fmask] - fmin) / (fmax - fmin))
            #     * 2.0 * num.pi)** self.weighting_exponent

            weights[ilevel, fmask] = cos_taper(
                (log_frequencies_all[fmask] - num.log(fmin))
                / (num.log(fmax) - num.log(fmin)) * 2.0 * num.pi) \
                ** self.weighting_exponent

            if ilevel == 0:
                weights[ilevel, 1:] += cos_taper(
                    norm_log_frequencies_all*num.pi)**2 * eps1

            if ilevel == self.nlevels - 1:
                weights[ilevel, 1:] += cos_taper(
                    (1.0 - norm_log_frequencies_all)*num.pi)**2 * eps1

        weights += eps2
        weights /= num.sum(weights, axis=0)[num.newaxis, :]

        return frequencies_all, weights


def make_block(ilevel, windowing, samples_td):
    if num.all(samples_td == 0.0):
        return (
            num.full(samples_td.size // 2 + 1, num.nan),
            num.zeros(samples_td.size // 4))

    taper_td = windowing.taper_td
    taper_fd_2 = windowing.taper_fd_2
    taper_td_2_inv_center = windowing.taper_td_2_inv_center
    nblock = windowing.nblock
    dmean = num.mean(samples_td*taper_td)
    samples_fd = num.fft.rfft((samples_td-dmean) * taper_td)
    nf = samples_fd.size
    samples_td_2 = num.fft.irfft(samples_fd[:nf//2 + 1] * taper_fd_2) / 2
    samples_td_2_center = samples_td_2[nblock//8:3*nblock//8]
    samples_td_2_center *= taper_td_2_inv_center
    samples_td_2_center += dmean
    samples_out = num.abs(samples_fd)
    samples_out[samples_out == 0.0] = num.nan
    # old_settings = num.seterr(divide='ignore')
    num.log(samples_out, out=samples_out)
    # num.seterr(**old_settings)
    samples_out *= 2.0  # power-spectrum
    samples_out += num.log(2**ilevel)

    return (samples_out, samples_td_2_center)


class SpectrogramGroup(guts.Object):
    codes = model.CodesNSLCE.T()
    tmin = guts.Timestamp.T()
    tmax = guts.Timestamp.T()
    deltat = guts.Float.T()
    windowing = Pow2Windowing.T()
    levels = guts.List.T(Carpet.T())

    def get_multi_spectrogram(self, interpolation='cos'):
        frequencies, weights = self.windowing.frequencies_and_weights(
            self.deltat)

        times = self.levels[0].times
        spectrogram = num.zeros((frequencies.size, times.size))
        nfcut = frequencies.size

        # thalf0 = 0.5 * self.levels[0].deltat
        thalf0 = 0.0

        for ilevel, level in enumerate(self.levels):
            index_f = num.floor(
                frequencies[:nfcut]
                / level.component_axes['frequency'][1]).astype(int)

            # thalf = 0.0625 * level.deltat
            thalf = 0.0

            if interpolation == 'nearest_neighbor' or ilevel == 0:
                index_t = num.round((
                    (times + thalf0) - (level.tmin + thalf))
                    / level.deltat).astype(int)

                spectrogram[:nfcut, :] \
                    += level.data[index_f][:, index_t] \
                    * weights[ilevel, :nfcut][:, num.newaxis]

            elif interpolation == 'cos':
                index_t = num.clip(
                    num.floor(
                        ((times + thalf0) - (level.tmin + thalf))
                        / level.deltat).astype(int),
                    0,
                    level.times.size-2)

                w1 = cos_taper(
                    num.pi * ((times+thalf0) - (level.times[index_t]+thalf))
                    / level.deltat)

                w0 = 1.0 - w1

                spectrogram[:nfcut, :] \
                    += w0[num.newaxis, :] \
                    * level.data[index_f][:, index_t] \
                    * weights[ilevel, :nfcut][:, num.newaxis]

                spectrogram[:nfcut, :] \
                    += w1[num.newaxis, :] \
                    * level.data[index_f][:, index_t+1] \
                    * weights[ilevel, :nfcut][:, num.newaxis]

            else:
                raise ValueError(
                    'invalid interpolation method: %s' % interpolation)

            nfcut -= self.windowing.nfrequencies // 2

        return Carpet(
            codes=self.codes,
            tmin=self.levels[0].tmin,
            deltat=self.levels[0].deltat,
            component_axes={'frequency': frequencies},
            data=spectrogram)

    def get_frequency_range(self):
        return (
            1.0 / (self.deltat * self.windowing.nblock * 2**(
                self.windowing.nlevels-1)),
            0.5 / self.deltat)

    def fill_template(self, path):
        return path.format(
            codes=self.codes.safe_str,
            tmin=util.time_to_str(self.tmin, '%Y-%m-%dT%H-%M-%S'),
            tmax=util.time_to_str(self.tmax, '%Y-%m-%dT%H-%M-%S'))

    def plot_construction(self, path=None, interpolation='cos'):

        fig = plt.figure(figsize=(60, 20))
        fig.suptitle(self.fill_template('{codes} ({tmin} - {tmax})'))
        axes_over = fig.add_subplot(3, 1, 2)
        axes_over.set_xscale('log')

        fslice = slice(2, None)
        levels = [
            level.crop(fslice=fslice)
            for level in self.levels]

        vmin = min(level.stats.p10 for level in levels)
        vmax = max(level.stats.max for level in levels)

        axes = None
        for ilevel, level in enumerate(levels):

            axes = fig.add_subplot(
                3, self.windowing.nlevels + 1, ilevel+1,
                sharex=axes)

            level.mpl_draw(
                axes,
                vmin=vmin,
                vmax=vmax,
                component_axis='frequency')

            axes.set_xlim(self.tmin, self.tmax)
            axes.set_ylim(*self.get_frequency_range())
            plot.mpl_time_axis(axes)
            axes.set_yscale('log')

            mean_amps = num.nanmean(level.data, axis=1)
            axes_over.plot(level.component_axes['frequency'], mean_amps)

        spectrogram = self.get_multi_spectrogram(interpolation=interpolation) \
            .crop(fslice=fslice)

        axes = fig.add_subplot(
            3, self.windowing.nlevels + 1, self.windowing.nlevels + 1,
            sharex=axes)
        spectrogram.mpl_draw(
            axes,
            vmin=vmin,
            vmax=vmax,
            component_axis='frequency')

        axes.set_xlim(self.tmin, self.tmax)
        axes.set_ylim(*self.get_frequency_range())
        plot.mpl_time_axis(axes)
        axes.set_yscale('log')

        mean_amps = num.nanmean(spectrogram.data, axis=1)
        axes_over.plot(
            spectrogram.component_axes['frequency'],
            mean_amps, color='black', zorder=10)

        frequencies, weights = self.windowing.frequencies_and_weights(
            self.deltat)

        axes_weights = fig.add_subplot(3, 1, 3)
        axes_weights.set_xscale('log')

        for ilevel in range(self.windowing.nlevels):
            axes_weights.plot(
                frequencies[fslice], weights[ilevel, fslice])

        axes_weights.plot(
            frequencies[fslice], num.sum(weights, axis=0)[fslice])

        if path is not None:
            fig.savefig(self.fill_template(path))
        else:
            plt.show()


class MultiSpectrogramOperator(base.Operator):

    translation = base.Translation.T(
        default=base.AddSuffixTranslation.D(suffix='MS'))
    windowing = Pow2Windowing.T(
        default=Pow2Windowing.D())
    quantity = model.QuantityType.T(default='velocity')
    tinc_max = guts.Float.T(optional=True)
    interpolation = guts.StringChoice.T(
        choices=['cos', 'nearest_neighbor'], default='cos')

    def __init__(self, *args, **kwargs):
        base.Operator.__init__(self, *args, **kwargs)
        self._caches = {}
        self._tinc_max_effective = self.tinc_max
        self._accessor_id = 'spectrogram_%s' % str(self._operator_id)

    @property
    def kind_requires(self):
        if self.quantity == 'counts':
            return ('waveform',)
        else:
            return ('waveform', 'response')

    @property
    def kind_provides(self):
        return ('channel', 'carpet')

    def make_block(self, deltat, kind_codes_id, ilevel, iwindow):
        if ilevel == 0:
            tmin_window = iwindow * self.windowing.time_increment(
                deltat, ilevel)
            tmax_window = tmin_window + self.windowing.nblock * deltat

            trs = self._input.get_waveforms(
                tmin=tmin_window,
                tmax=tmax_window,
                kind_codes_ids=[kind_codes_id],
                want_incomplete=False,
                accessor_id=self._accessor_id)

            ntraces = len(trs)

            assert ntraces <= 1
            if ntraces:
                tr = trs[0]
                nblock = self.windowing.nblock
                assert tr.ydata.size == nblock
                return make_block(ilevel, self.windowing, tr.ydata)

            else:
                return make_block(
                    ilevel, self.windowing, num.zeros(self.windowing.nblock))

        else:

            blocks = [
                self.get_block(deltat, kind_codes_id, ilevel - 1, iwin)
                for iwin in range(iwindow*2-1, iwindow*2+4)]

            nblock = self.windowing.nblock
            aggregation = (
                [blocks[0][1][nblock//8:]]
                + [block[1] for block in blocks[1:4]]
                + [blocks[-1][1][:-nblock//8]])

            have_zero_block = any(
                num.all(block[1] == 0.0) for block in blocks)
            samples = num.concatenate(aggregation)
            if have_zero_block:
                samples[:] = 0.0

            return make_block(ilevel, self.windowing, samples)

    def get_block(self, deltat, kind_codes_id, ilevel, iwindow):
        k1 = (deltat, kind_codes_id, ilevel)
        if k1 not in self._caches:
            def make_get_block_window(deltat, kind_codes_id, ilevel):
                def get_block_window(iwindow):
                    return self.make_block(
                        deltat, kind_codes_id, ilevel, iwindow)

                cachesize = int(
                    max(
                        4 * self._tinc_max_effective,
                        16 * self.windowing.time_increment(
                            deltat, self.windowing.nlevels-1))
                    / (deltat*self.windowing.nblock))

                logger.debug('Cache size: %i' % cachesize)
                return LRU(get_block_window, maxsize=cachesize)

            self._caches[k1] = make_get_block_window(
                deltat, kind_codes_id, ilevel)

        return self._caches[k1](iwindow)

    def get_spectrogram_groups(
            self,
            obj=None,
            tmin=None,
            tmax=None,
            codes=None,
            nsamples_limit=None,
            **kwargs):

        if self._tinc_max_effective is None:
            self._tinc_max_effective = tmax - tmin

        if tmax - tmin > self._tinc_max_effective:
            logger.warning(
                'MultiSpectrogramOperator: Cache size may be too small. '
                'Try setting or increasing `tinc_max` parameter.')

        tmin, tmax, codes = model.get_selection_args(
            model.CHANNEL, obj, tmin, tmax, None, codes)

        if None in (tmin, tmax):
            raise ValueError('No time span given.')

        info = self._input.get_codes_info('waveform', codes=codes)
        groups = []

        for pattern, kind_codes_id, codes, deltat in info:
            nsamples_required = int(round((tmax - tmin) / deltat))
            if nsamples_limit is not None \
                    and nsamples_required > nsamples_limit:
                continue

            resp = None
            if self.quantity != 'counts':
                try:
                    resp = self._input \
                        .get_response(codes=codes, tmin=tmin, tmax=tmax) \
                        .get_effective(self.quantity)

                except error.SquirrelError as e:
                    logger.error(e)
                    continue

            levels = []
            for ilevel in range(self.windowing.nlevels):
                iwindows = self.windowing.iwindow_range(
                    deltat, tmin, tmax, ilevel)

                times = iwindows \
                    * self.windowing.time_increment(deltat, ilevel)

                nwindows = iwindows.size
                nfrequencies = self.windowing.nfrequencies
                values = num.zeros(
                    (nfrequencies, nwindows))

                for i in range(nwindows):
                    values[:, i] = self.get_block(
                        deltat, kind_codes_id, ilevel, iwindows[i])[0]

                frequency_delta = 1.0 \
                    / (deltat * self.windowing.nblock * 2**ilevel)

                frequencies = num.arange(nfrequencies) * frequency_delta

                if resp:
                    values[1:, :] -= 2.0 * num.log(num.abs(resp.evaluate(
                        frequencies[1:])))[:, num.newaxis]

                deltat_carpet = self.windowing.time_increment(deltat, ilevel)
                levels.append(Carpet(
                    codes=codes,
                    tmin=times[0]+deltat_carpet,
                    deltat=deltat_carpet,
                    component_axes={'frequency': frequencies},
                    data=values))

            groups.append(SpectrogramGroup(
                codes=codes,
                tmin=tmin,
                tmax=tmax,
                deltat=deltat,
                windowing=self.windowing,
                levels=levels))

        self.advance_accessor(self._accessor_id, 'waveform')

        return groups

    def get_carpets(
            self,
            codes=None,
            tmin=None,
            tmax=None,
            show_construction=False,
            nsamples_limit=None):

        groups = self.get_spectrogram_groups(
            codes=codes,
            tmin=tmin,
            tmax=tmax,
            nsamples_limit=nsamples_limit)

        carpets = []
        for group in groups:
            if show_construction:
                group.plot_construction(interpolation=self.interpolation)

            fslice = slice(2,
                           None)
            carpet = group.get_multi_spectrogram(
                interpolation=self.interpolation).crop(fslice=fslice)

            carpet.data = carpet.data.astype(num.float32)
            carpet.codes = self.translation.translate(carpet.codes)
            carpets.append(carpet)

        return carpets


__all__ = [
    'Pow2Windowing',
    'SpectrogramGroup',
    'MultiSpectrogramOperator',
]
