# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Multi-component waveform data model.
'''


import numpy as num

from . import trace
from .guts import Object, Float, Timestamp, List
from .guts_array import Array
from .squirrel.model import CodesNSLCE


class MultiTrace(Object):
    '''
    Container for multi-component waveforms with common time span and sampling.

    Instances of this class can be used to efficiently represent
    multi-component waveforms of a single sensor or of a sensor array. The data
    samples are stored in a single 2D array where the first index runs over
    components and the second index over time. Metadata contains sampling rate,
    start-time and :py:class:`~pyrocko.squirrel.model.CodesNSLCE` identifiers
    for the contained traces.

    :param traces:
        If given, construct multi-trace from given single-component waveforms
        (see :py:func:`~pyrocko.trace.get_traces_data_as_array`) and ignore
        any other arguments.
    :type traces:
        :py:class:`list` of :py:class:`~pyrocko.trace.Trace`
    '''

    codes = List.T(
        CodesNSLCE.T(),
        help='List of codes identifying the components.')
    data = Array.T(
        shape=(None, None),
        help='Array containing the data samples indexed as '
             '``(icomponent, isample)``.')
    tmin = Timestamp.T(
        default=Timestamp.D('1970-01-01 00:00:00'),
        help='Start time.')
    deltat = Float.T(
        default=1.0,
        help='Sampling interval [s]')

    def __init__(
            self,
            traces=None,
            codes=None,
            data=None,
            tmin=None,
            deltat=None):

        if traces is not None:
            if len(traces) == 0:
                data = num.zeros((0, 0))
            else:
                data = trace.get_traces_data_as_array(traces)
                deltat = traces[0].deltat
                tmin = traces[0].tmin
                codes = [tr.codes for tr in traces]

        self.ntraces, self.nsamples = data.shape

        if codes is None:
            codes = [CodesNSLCE()] * self.ntraces

        if len(codes) != self.ntraces:
            raise ValueError(
                'MultiTrace construction: mismatch between number of traces '
                'and number of codes given.')

        if deltat is None:
            deltat = self.T.deltat.default()

        if tmin is None:
            tmin = self.T.tmin.default()

        Object.__init__(self, codes=codes, data=data, tmin=tmin, deltat=deltat)

    def __len__(self):
        '''
        Get number of components.
        '''
        return self.ntraces

    def __getitem__(self, i):
        '''
        Get single component waveform (shared data).

        :param i:
            Component index.
        :type i:
            int
        '''
        return self.get_trace(i)

    @property
    def tmax(self):
        '''
        End time (time of last sample, read-only).
        '''
        return self.tmin + (self.nsamples - 1) * self.deltat

    def get_trace(self, i):
        '''
        Get single component waveform (shared data).

        :param i:
            Component index.
        :type i:
            int
        '''

        network, station, location, channel, extra = self.codes[i]
        return trace.Trace(
            network=network,
            station=station,
            location=location,
            channel=channel,
            extra=extra,
            tmin=self.tmin,
            deltat=self.deltat,
            ydata=self.data[i, :])

    def snuffle(self):
        '''
        Show in Snuffler.
        '''
        trace.snuffle(list(self))
