# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function, division

from pyrocko.gui.snuffler.snuffling import Snuffling, Param
from pyrocko.trace import Trace
import numpy as num


class RootMeanSquareSnuffling(Snuffling):

    '''
    Create traces with blockwise root mean square values.
    '''

    def setup(self):
        '''Customization of the snuffling.'''

        self.set_name('Block RMS')
        self.add_parameter(Param(
            'Block Length [s]', 'block_length', 100., 0.1, 3600.))
        self.set_live_update(False)

    def call(self):
        '''Main work routine of the snuffling.'''

        self.cleanup()

        tinc = self.block_length

        tmin, tmax = self.get_selected_time_range(fallback=True)
        n = int((tmax-tmin)/tinc)

        rms_by_nslc = {}
        for traces in self.chopper_selected_traces(
                tinc=tinc,
                want_incomplete=False,
                fallback=True):

            for tr in traces:

                # ignore the block if it extends past the region of interest
                if tr.tmax > tmax:
                    continue

                # don't work traces produced by this (and other) snuffling
                if tr.meta and 'tabu' in tr.meta and tr.meta['tabu']:
                    continue

                # create a trace of required length if none has been
                # initialized yet
                if tr.nslc_id not in rms_by_nslc:
                    rms_by_nslc[tr.nslc_id] = Trace(
                        network=tr.network,
                        station=tr.station,
                        location=tr.location,
                        channel=tr.channel+'-RMS',
                        tmin=tmin + 0.5*tinc,
                        deltat=tinc,
                        ydata=num.zeros(n, dtype=num.float),
                        meta={'tabu': True})

                # create and insert the current sample
                i = int(round((tr.tmin - tmin)/tinc))
                if 0 <= i and i < n:
                    tr.ydata = num.asarray(tr.ydata, num.float)
                    tr.ydata -= num.mean(tr.ydata)
                    value = num.sqrt(num.sum(tr.ydata**2)/tr.ydata.size)

                    rms_by_nslc[tr.nslc_id].ydata[i] = value

        # add the newly created traces to the viewer
        if rms_by_nslc.values():
            self.add_traces(list(rms_by_nslc.values()))


def __snufflings__():
    '''Returns a list of snufflings to be exported by this module.'''

    return [RootMeanSquareSnuffling()]
