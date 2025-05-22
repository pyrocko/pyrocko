# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Implementation of :app:`squirrel benchmark`.
'''

import time
from pyrocko.guts import Object, Float, Int, Timestamp, List
from pyrocko import util, guts
from pyrocko.squirrel.error import ToolError
from pyrocko.squirrel.base import Batch
from pyrocko.squirrel.io.backends import mseed as sq_mseed
from pyrocko.io import mseed
from ..common import ldq


headline = 'Perform benchmark tests.'


def make_subparser(subparsers):
    return subparsers.add_parser(
        'benchmark',
        help=headline,
        description=headline + '''

The following benchmarks are available:

chopper-waveforms

Test performance of waveform data reading in typical continuous waveform
processing schemes. Use ``--dataset`` or ``--add`` to select input data and
``--tinc`` to set a processing time-window duration. Query options ``--codes``,
``--tmin`` and ``--tmax`` can be used to restrict the reading to a specific
subset of the available data.

mseed-files-parse

Test speed of the mseed waveform decoding which is performed in the libmseed
functions without the overhead of Squirrel's database querying. This test
sequentially reads all files in the set-up data selection. Any query options
are ignored in this test.

mseed-files-disk-io

Test speed of raw disk io without the overhead of decoding the mseed data. This
test sequentially reads all files in the set-up data selection. Any query
options are ignored in this test. Note: the io speed is reported under traces,
even though no traces are decoded in this test.

''')


def setup(parser):
    benchmark_choices = [
        'chopper-waveforms',
        'mseed-files-parse',
        'mseed-files-disk-io']

    parser.add_argument(
        'benchmark',
        metavar='BENCHMARK',
        help='Benchmark to perform. Choices: %s.'
        % ldq(benchmark_choices))

    parser.add_squirrel_selection_arguments()
    parser.add_squirrel_query_arguments(without=['kinds', 'time'])

    parser.add_argument(
        '--tinc',
        dest='tinc',
        type=guts.parse_duration,
        metavar='DURATION',
        default=3600.,
        help='Set processing time interval for ```chopper``` benchmark [s].')


class BatchInfo(Object):
    tmin = Float.T()
    tmax = Float.T()
    i = Int.T()
    n = Int.T()

    def __str__(self):
        return ('[%' + str(len(str(self.n))) + 'i/%i %3.0f%% %s - %s]') % (
            self.i+1,
            self.n,
            (self.i+1) / self.n * 100.0,
            util.time_to_str(self.tmin, format='%Y-%m-%d %H:%M:%S'),
            util.time_to_str(self.tmax, format='%Y-%m-%d %H:%M:%S'))

    @classmethod
    def make(cls, batch):
        return cls(
            tmin=batch.tmin,
            tmax=batch.tmax,
            i=batch.i,
            n=batch.n)


class ThroughputHistory:

    def __init__(self):
        self._history = []
        self._nbytes = 0
        self._nsamples = 0

    def update(self, batch=None, nbytes=None):
        if nbytes is not None:
            self._nbytes += nbytes
        else:
            self._nbytes += sum(tr.ydata.nbytes for tr in batch.traces)
            self._nsamples += sum(tr.ydata.size for tr in batch.traces)

        self._history.append((
            time.time(),
            BatchInfo.make(batch),
            mseed.g_bytes_read,
            self._nbytes,
            self._nsamples))

    def get_stats(self):
        return ThroughputStats.make(self._history)


def total_and_rates(label, total, rates, format=util.human_bytesize):
    return '%s: %s (%s)' % (
        label,
        format(total),
        ', '.join('%s/s' % format(rate) for rate in rates))


class ThroughputStats(Object):
    time = Timestamp.T()
    batch = BatchInfo.T(optional=True)
    nbytes_mseed = Int.T()
    nbytes_traces = Int.T()
    nsamples = Int.T()
    time_averages = List.T(Float.T())
    nbytes_mseed_rates = List.T(Float.T())
    nbytes_traces_rates = List.T(Float.T())
    nsamples_rates = List.T(Float.T())

    def __str__(self):
        return '''%s
    %s
    %s
    %s''' % (
            str(self.batch) if self.batch else '',
            total_and_rates(
                'mseed',
                self.nbytes_mseed,
                self.nbytes_mseed_rates),
            total_and_rates(
                'traces',
                self.nbytes_traces,
                self.nbytes_traces_rates),
            total_and_rates(
                'samples',
                self.nsamples,
                self.nsamples_rates,
                format=util.human_intsize)
        )

    @classmethod
    def make(cls, history, time_averages=(1., 3., 10., None)):
        end = history[-1]
        t, batch, nbytes_mseed, nbytes_traces, nsamples = end
        stats = cls(
            time=t,
            batch=batch,
            nbytes_mseed=nbytes_mseed,
            nbytes_traces=nbytes_traces,
            nsamples=nsamples)

        begins = []
        for time_average in time_averages:
            if time_average is None and len(history) > 1:
                begins.append(history[0])
            else:
                for i in range(len(history)-2, 0, -1):
                    if history[i][0] < t - time_average:
                        begins.append(history[i])
                        break

        for begin in begins:
            time_delta = end[0] - begin[0]
            if time_delta > 0:
                nbytes_mseed_rate, nbytes_traces_rate, nsamples_rate = [
                    (end[i] - begin[i]) / time_delta for i in range(2, 5)]

                stats.nbytes_mseed_rates.append(nbytes_mseed_rate)
                stats.nbytes_traces_rates.append(nbytes_traces_rate)
                stats.nsamples_rates.append(nsamples_rate)
                stats.time_averages.append(time_delta)

        return stats


def run(parser, args):
    sq = args.make_squirrel()

    history = ThroughputHistory()

    with util.SignalQuitable() as quitable:
        tlast = time.time()

        if args.benchmark == 'chopper-waveforms':

            for batch in sq.chopper_waveforms(
                    tinc=args.tinc,
                    **args.squirrel_query):

                history.update(batch)
                tnow = time.time()
                if tnow > tlast + 1.0:
                    print(history.get_stats())
                    tlast = tnow

                if quitable.quit_requested:
                    break

        elif args.benchmark == 'mseed-files-parse':
            if args.tinc != 3600. or any(
                    x is not None for x in args.squirrel_query.values()):

                raise ToolError(
                    'Invalid options given for benchmark "%s".'
                    % args.benchmark)

            paths = sq.get_paths(format='mseed')
            for ipath, path in enumerate(paths):
                nuts = list(sq_mseed.iload('mseed', path, 0, ('waveform',)))

                if not nuts:
                    continue

                batch = Batch(
                    i=ipath,
                    n=len(paths),
                    igroup=0,
                    ngroups=0,
                    tmin=min(nut.content.tmin for nut in nuts),
                    tmax=max(nut.content.tmax for nut in nuts),
                    traces=[nut.content for nut in nuts])

                history.update(batch)
                tnow = time.time()
                if tnow > tlast + 1.0:
                    print(history.get_stats())
                    tlast = tnow

                if quitable.quit_requested:
                    break

        elif args.benchmark == 'mseed-files-disk-io':
            if args.tinc != 3600. or any(
                    x is not None for x in args.squirrel_query.values()):

                raise ToolError(
                    'Invalid options given for benchmark "%s".'
                    % args.benchmark)

            paths = sq.get_paths(format='mseed')
            for ipath, path in enumerate(paths):

                with open(path, 'rb') as f:
                    data = f.read()

                batch = Batch(
                    i=ipath,
                    n=len(paths),
                    igroup=0,
                    ngroups=0,
                    tmin=0.0,
                    tmax=0.0,
                    traces=[])

                history.update(batch, nbytes=len(data))
                tnow = time.time()
                if tnow > tlast + 1.0:
                    print(history.get_stats())
                    tlast = tnow

                if quitable.quit_requested:
                    break

    print(history.get_stats())
