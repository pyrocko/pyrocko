# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

import numpy as num

from pyrocko.guts import Float
from pyrocko.model.codes import match_codes_any
from pyrocko.carpet import Carpet
from pyrocko.gato.grid import (
    Grid, LocationGrid, UnstructuredLocationGrid, distances_3d)
from pyrocko.gato.delay.base import DelayMethod
from pyrocko.squirrel.model import unpack_rich, pack_rich
from .csm import CSMOperator

guts_prefix = 'gato'


class FKBeamOperator(CSMOperator):

    frequency_min = Float.T(default=0.1)
    frequency_max = Float.T(default=0.3)
    source_grid = Grid.T()
    delay_method = DelayMethod.T()

    @property
    def kind_provides(self):
        return ('carpet',)

    def get_out_channels(self):
        channels = []
        names = [name for name, _ in self.source_grid.native_coordinates()]
        if len(names) > 1:
            for name in names:
                channels.append(name + '_max')

        if len(names) > 2:
            for ia, name_a in enumerate(names):
                for name_b in names[ia+1:]:
                    channels.append(name_a + name_b + '_max')

        channels.append(''.join(names))
        channels.append('avail')
        return {'carpets': channels}

    def make_carpets(self, tmin=None, tmax=None, codes=None):
        mappings = self.get_mappings()
        from pyrocko.gato import GenericDelayTable

        arrays = self.get_arrays()
        gdts = []
        for array in arrays:
            array_info = array.get_info(
                self._input,
                codes=self.in_codes,
                deduplicate=False,
                tmin=tmin,
                tmax=tmax)

            # array_info = self._array_infos[array.name]
            receiver_grid = UnstructuredLocationGrid.from_locations(
                array_info.sensors)

            if isinstance(self.source_grid, LocationGrid):
                receiver_grid.origin = self.source_grid.origin

            gdts.append(GenericDelayTable(
                source_grid=self.source_grid,
                receiver_grid=receiver_grid,
                method=self.delay_method))

        carpets_out = []
        for array, mapping, gdt in zip(self.get_arrays(), mappings, gdts):
            carpet_out = None

            iterator = self.iter_csms(
                mapping, tmin=tmin, tmax=tmax, codes=codes)

            in_codes = list(mapping.in_codes)

            out_codes_avail = [
                c for c in mapping.out_codes if c.channel == 'avail'][0]
            if codes is None or match_codes_any(codes, out_codes_avail):
                availability = Carpet(
                    codes=out_codes_avail,
                    component_codes=in_codes,
                    deltat=self.time_window,
                    tmin=tmin + 0.5 * self.time_window,
                    data=num.zeros((len(in_codes), len(iterator)), dtype=int))

                carpets_out.append(availability)
            else:
                availability = None

            codes_to_icodes = dict((c, i) for (i, c) in enumerate(in_codes))

            coverages = self._input.get_rich_coverage(
                tmin=tmin, tmax=tmax, codes=in_codes)

            for (batch, carpet_in, frequency_delta, cspectrum_sum,
                    codes_use) in iterator:

                if carpet_in is None:
                    continue

                if availability:
                    codes_avail = set(tr.codes for tr in batch.traces)
                    for coverage in coverages:
                        icodes = codes_to_icodes[coverage.codes]
                        time = 0.5 * (batch.tmin + batch.tmax)
                        itime = int(round(
                            (time - availability.tmin) / availability.deltat))
                        values = unpack_rich(coverage.get(time))
                        values[5] = int(coverage.codes in codes_use)
                        values[6] = int(coverage.codes in codes_avail)
                        availability.data[icodes, itime] = pack_rich(values)

                # nrecords = cspectrum_sum.shape[0]
                nfrequencies = cspectrum_sum.shape[2]
                frequencies = num.arange(nfrequencies) * frequency_delta

                codes_to_i_info = dict(
                    (codes, i) for (i, codes) in enumerate(array_info.codes))

                icomps = num.array([
                    codes_to_i_info[codes]
                    for codes in carpet_in.component_codes])

                ifmin = int(round(self.frequency_min / frequency_delta))
                ifmax = int(round(self.frequency_max / frequency_delta))

                delay_spectra, delay_spectra_conj = \
                    gdt.get_delay_spectra(frequencies[ifmin:ifmax])

                isds = distances_3d(gdt.receiver_grid, gdt.receiver_grid)
                # print('isd', isds.shape)
                isds = isds[icomps, :][:, icomps]
                # print(isds)

                # cspectrum_sum *= (isds < 4000.)[:, :, num.newaxis]

                map = num.einsum(
                    'iqk,qmk,imk->i',
                    delay_spectra[:, icomps, :],
                    cspectrum_sum[:, :, ifmin:ifmax],
                    delay_spectra_conj[:, icomps, :])

                norm = num.sum(num.abs(cspectrum_sum[:, :, ifmin:ifmax])**2)
                # print((num.abs(map)).max())
                # print(norm.max())

                amap_flat = num.abs(map) / norm
                if carpet_out is None:
                    carpet_out = []
                    for out_codes in mapping.out_codes:
                        if out_codes.channel == 'avail':
                            continue

                        if codes is not None \
                                and not match_codes_any(codes, out_codes):
                            continue

                        projection = out_codes.channel.split('_')[0]
                        coords = gdt.source_grid \
                            .native_coordinates_projected(projection)

                        axes = dict(
                            (name, vals)
                            for (name, vals) in zip(projection, coords.T))

                        carpet_out.append(Carpet(
                            codes=out_codes,
                            tmin=tmin,
                            deltat=self.time_window,
                            component_axes=axes,
                            data=num.zeros((coords.shape[0], batch.n))))

                amap = amap_flat.reshape(gdt.source_grid.shape)

                coord_names = [
                    name for (name, _) in gdt.source_grid.native_coordinates()]

                for i, carpet in enumerate(carpet_out):
                    channel = carpet.codes.channel
                    if channel.endswith('_max'):
                        projection = channel[:-4]
                        axis = tuple(
                            k for (k, name) in enumerate(coord_names)
                            if name not in projection)

                        carpet.data[:, batch.i] = num.max(
                            amap, axis=axis).flatten()
                    else:
                        carpet.data[:, batch.i] = amap_flat

            if carpet_out:
                carpets_out.extend(carpet_out)

        return carpets_out


__all__ = [
    'FKBeamOperator',
]
