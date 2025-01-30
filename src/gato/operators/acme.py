# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

import numpy as num

from pyrocko.guts import Float
from pyrocko.carpet import Carpet
from .csm import CSMOperator

guts_prefix = 'gato'


class ACMEOperator(CSMOperator):
    frequency_min = Float.T(default=0.1)
    frequency_max = Float.T(default=0.3)

    @property
    def kind_provides(self):
        return ('carpet')

    def get_out_channels(self):
        return {
            'carpet': ['ACME'],
        }

    def make_carpets(self, tmin=None, tmax=None, codes=None):
        mappings = self.get_mappings()
        carpets_out = []
        for mapping in mappings:
            carpet_out = None
            for (batch, carpet_in, frequency_delta, cspectrum_sum,
                 codes_use) in self.iter_csms(
                    mapping, tmin=tmin, tmax=tmax, codes=codes):

                nrecords = cspectrum_sum.shape[0]
                nfrequencies = cspectrum_sum.shape[2]

                ifmin = int(num.ceil(
                    self.frequency_min / frequency_delta))
                ifmax = int(num.floor(
                    self.frequency_max / frequency_delta)) + 1

                assert ifmin < ifmax
                assert ifmax <= nfrequencies

                if carpet_out is None:
                    carpet_out = Carpet(
                        codes=mapping.out_codes[0],
                        tmin=tmin,
                        deltat=self.time_window,
                        component_axes={
                            'frequency':
                            num.arange(ifmin, ifmax) * frequency_delta},
                        data=num.zeros((ifmax - ifmin, batch.n)))

                for ifrequency in range(ifmin, ifmax):
                    eigenvalues, _ = num.linalg.eigh(
                        cspectrum_sum[:, :, ifrequency])

                    eigenvalues = eigenvalues[::-1]
                    assert all(eigenvalues[:-1] >= eigenvalues[1:])
                    spectral_width = \
                        num.sum(num.arange(nrecords)*eigenvalues) \
                        / num.sum(eigenvalues)

                    carpet_out.data[ifrequency-ifmin, batch.i] = spectral_width

            if carpet_out:
                carpets_out.append(carpet_out)

        return carpets_out


__all__ = [
    'ACMEOperator',
]
