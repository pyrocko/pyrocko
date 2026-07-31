# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

import logging

import numpy as num

from pyrocko.guts import Float, String
from pyrocko.model.codes import match_codes_any
from pyrocko.carpet import Carpet
from pyrocko.squirrel.model import unpack_rich, pack_rich
from pyrocko.gato.grid import (
    Grid,
    LocationGrid,
    distances_3d,
    CartesianSlownessGrid)
from pyrocko.gato.delay import DelayMethod, PlaneWaveDM
from pyrocko.gato.error import GatoError
from .csm import CSMOperator, ArrayProcessingSetup
from pyrocko.squirrel.operators.base import basic_codes_projection_t, Outlet

guts_prefix = 'gato'

logger = logging.getLogger('gato.operators.csm_image')


class CSMImageOperator(CSMOperator):
    name = String.T(default='csmi')

    codes_projection = basic_codes_projection_t(
        '.{o.array}.{o.field}.{i.channel_component}.')

    frequency_min = Float.T(default=0.1)
    frequency_max = Float.T(default=0.3)
    source_grid = Grid.T(optional=True)
    delay_method = DelayMethod.T(optional=True)

    def post_init(self):
        CSMOperator.post_init(self)
        self._setups = None

    @property
    def kind_provides(self):
        return ('carpet',)

    def get_effective_source_grid(self):
        if self.source_grid is not None:
            return self.source_grid

        # could check array size and frequency range for automatic smax
        # for array in self.get_sensor_arrays():
        #     array_incarnation = array.get_incarnation(
        #         self._input,
        #         codes=self.in_codes or None,
        #         deduplicate=False)
        #
        #     print(array_incarnation)

        # for now just this:

        slowness_max = 1. / 2000.
        return CartesianSlownessGrid.from_smax_2d(
            slowness_max, slowness_max / 20.)

    def get_fields(self):
        source_grid = self.get_effective_source_grid()
        fields = []
        names = list(source_grid.native_coordinates().keys())
        if len(names) > 1:
            for name in names:
                fields.append(name + '_max')

        if len(names) > 2:
            for ia, name_a in enumerate(names):
                for name_b in names[ia+1:]:
                    fields.append(name_a + name_b + '_max')

        fields.append(''.join(names))
        fields.append('avail')
        return fields

    def get_outlets_for_array(self, array, fields=None):
        outlets = []
        if fields is None:
            fields = self.get_fields()

        for field in fields:
            outlets.append(
                Outlet(
                    kinds=['carpet'],
                    attributes={
                        'array': array.name,
                        'field': field}))

        return outlets

    def make_array_processing_setup(self, array, incarnation, mapping):
        from pyrocko.gato import GenericDelayTable

        receiver_grid = incarnation.get_location_grid()
        source_grid = self.get_effective_source_grid()

        if isinstance(source_grid, LocationGrid):
            receiver_grid.origin = source_grid.origin

        delay_method = self.delay_method or PlaneWaveDM()

        gdt = GenericDelayTable(
            source_grid=source_grid,
            receiver_grid=receiver_grid,
            method=delay_method)

        return ArrayProcessingSetup(array, incarnation, mapping, gdt)

    def get_out_codes_by_field(self, array, mapping, codes=None):
        field_to_out_codes = {}
        for outlet in self.get_outlets_for_array(array):
            field = outlet.attributes['field']
            out_codes_group = self.codes_projection.project(
                self, mapping.in_codes, [outlet])
            if len(out_codes_group) != 1:
                raise GatoError(
                    f'Codes projection must return exactly one entity '
                    f'for field "{field}".\n'
                    f'codes_projection:\n{self.codes_projection}\n')

            out_codes = out_codes_group[0]
            if codes is None or match_codes_any(codes, out_codes):
                field_to_out_codes[field] = out_codes_group[0]

        return field_to_out_codes

    def get_out_codes_by_agf(self):
        d_out_codes = {}
        for setup in self.get_setups():
            for field, codes in self.get_out_codes_by_field(
                    setup.array, setup.mapping).items():

                group_key = self.codes_projection.group_key(codes)
                d_out_codes[setup.array.name, group_key, field] = codes

        return d_out_codes

    def make_carpets(self, tmin=None, tmax=None, codes=None):

        time_window = self.get_effective_time_window()

        setups = self.get_setups()
        out_carpets = []
        for setup in setups:
            field_to_out_codes = self.get_out_codes_by_field(
                setup.array, setup.mapping, codes)

            if not field_to_out_codes:
                continue

            # skips codes where locations change within [tmin, tmax)
            codes_to_ilocation = setup.array_incarnation \
                .codes_to_ilocation_unique(tmin, tmax)

            # codes with unique=usable locations
            codes_usable = list(codes_to_ilocation.keys())

            iterator = self.iter_csms(
                setup.mapping, tmin=tmin, tmax=tmax, codes=codes_usable)

            in_codes = list(setup.mapping.in_codes)

            if 'avail' in field_to_out_codes:
                availability = Carpet(
                    codes=field_to_out_codes['avail'],
                    component_codes=in_codes,
                    deltat=time_window,
                    tmin=tmin + 0.5 * time_window,
                    data=num.zeros((len(in_codes), len(iterator)), dtype=int))
            else:
                availability = None

            codes_to_icodes = dict((c, i) for (i, c) in enumerate(in_codes))

            coverages = self._input.get_rich_coverage(
                tmin=tmin, tmax=tmax, codes=in_codes)

            gdt = setup.generic_delay_table

            field_to_out_carpet = None
            for (batch, in_carpet, frequency_delta, cspectrum_sum,
                    codes_use) in iterator:

                if in_carpet is None:
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

                nfrequencies = cspectrum_sum.shape[2]
                frequencies = num.arange(nfrequencies) * frequency_delta

                ilocations = num.array([
                    codes_to_ilocation[codes]
                    for codes in in_carpet.component_codes])

                ifmin = int(round(self.frequency_min / frequency_delta))
                ifmax = int(round(self.frequency_max / frequency_delta))

                delay_spectra, delay_spectra_conj = \
                    gdt.get_delay_spectra(frequencies[ifmin:ifmax])

                isds = distances_3d(gdt.receiver_grid, gdt.receiver_grid)
                isds = isds[ilocations, :][:, ilocations]

                logger.debug('Codes: %i (mapping), %i (usable), %i (avail)' % (
                    len(in_codes),
                    len(codes_usable),
                    len(in_carpet.component_codes)))

                logger.debug('Locations: %i (usable), %i (avail)' % (
                    len(setup.array_incarnation.locations),
                    len(set(ilocations))))

                image_flat = num.einsum(
                    'iqk,qmk,imk->i',
                    delay_spectra[:, ilocations, :],
                    cspectrum_sum[:, :, ifmin:ifmax],
                    delay_spectra_conj[:, ilocations, :])

                idiag = num.arange(cspectrum_sum.shape[0])
                norm = num.sum(num.abs(
                    cspectrum_sum[idiag, idiag, ifmin:ifmax])) \
                    * cspectrum_sum.shape[0]

                abs_image_flat = num.abs(image_flat) / norm

                if field_to_out_carpet is None:
                    field_to_out_carpet = {}
                    for field, out_codes in field_to_out_codes.items():
                        if field == 'avail':
                            continue

                        projection = field
                        if field.endswith('_max'):
                            projection = field[:-4]

                        coords = gdt.source_grid \
                            .native_coordinate_slice_grid(projection)

                        axes = dict(
                            (name, vals)
                            for (name, vals) in zip(projection, coords.T))

                        field_to_out_carpet[field] = Carpet(
                            codes=out_codes,
                            tmin=tmin,
                            deltat=time_window,
                            component_axes=axes,
                            data=num.zeros((coords.shape[0], batch.n)))

                abs_image = abs_image_flat.reshape(gdt.source_grid.shape)

                coord_names = gdt.source_grid.native_coordinates().keys()

                for field, out_carpet in field_to_out_carpet.items():
                    projection = field
                    if field.endswith('_max'):
                        projection = field[:-4]

                        axis = tuple(
                            k for (k, name) in enumerate(coord_names)
                            if name not in projection)

                        out_carpet.data[:, batch.i] = num.max(
                            abs_image, axis=axis).flatten()
                    else:
                        out_carpet.data[:, batch.i] = abs_image_flat

            if availability:
                out_carpets.append(availability)

            if field_to_out_carpet:
                out_carpets.extend(field_to_out_carpet.values())

        return out_carpets


__all__ = [
    'CSMImageOperator',
]
