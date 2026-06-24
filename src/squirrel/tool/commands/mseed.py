# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Implementation of :app:`squirrel mseed`.
'''

import logging
import numpy as num
from matplotlib import pyplot as plt
from pyrocko import plot, progress, util
from pyrocko.plot import smartplot
from pyrocko.io import mseed
from pyrocko.guts import Object, Int, Float, Timestamp, List
from pyrocko.guts_array import Array
from ..common import SquirrelCommand
from pyrocko.model.codes import CodesNSLCE

logger = logging.getLogger('psq.cli.mseed')

headline = 'Mini-SEED specific utilities'

description = '''%s''' % headline


def make_task(*args):
    return progress.task(*args, logger=logger)


def bin_edges(ibins):
    parts = num.concatenate((
        [0], num.where(num.diff(ibins) != 0)[0] + 1, [ibins.size]))

    for ilow, ihigh in zip(parts[:-1], parts[1:]):
        yield ilow, ihigh


def uniq(xs):
    p = None
    for x in xs:
        if p is None or x != p:
            yield x

        p = x


def connects(a, b, eps=0.1):
    return a.deltat == b.deltat \
        and abs((a.tmax + a.deltat) - b.tmin) < eps * a.deltat


def fit_control_points(deltat, interval, offsets, times, eps=1e-4):
    ibins = (times / interval).astype(int)
    cxs = []
    cys = []
    for ilow, ihigh in bin_edges(ibins):
        if ihigh - ilow < 2:
            continue

        x = offsets[ilow:ihigh]
        y = times[ilow:ihigh]

        m, b = num.polyfit(x, y, 1)
        if abs(deltat - m) > eps*deltat:
            continue

        if len(cxs) == 0:
            cxs.append(offsets[0])
            cys.append(offsets[0] * m + b)

        cx = 0.5 * (offsets[ilow] + offsets[ihigh-1])
        cxs.append(cx)
        cys.append(cx * m + b)

    if len(cxs) == 0:
        return times

    cxs.append(offsets[-1]+1)
    cys.append((offsets[-1]+1) * m + b)

    cxs = num.array(cxs)
    cys = num.array(cys)
    times_ip = num.interp(offsets, cxs, cys)

    return times_ip


def plot_results(results):

    p = smartplot.Plot(
        x_dims=['time'],
        y_dims=['d', 't'] * len(results) + ['t2'])

    p.set_label('d', 'Sampling offset [s]')
    p.set_label('t', 'Rate error $\\Delta t_{est} / \\Delta t_{nom}$')
    p.set_label('t2', 'Rate error $\\Delta t_{est} / \\Delta t_{nom}$')
    p.set_label('time', '')

    for i in range(2 * len(results)):
        plot.mpl_time_axis(p(0, i))

    plot.mpl_time_axis(p(0, len(results)*2))
    have_label = set()

    for iresult, result in enumerate(results):
        axes1 = p(0, iresult*2)
        axes2 = p(0, iresult*2 + 1)
        axes3 = p(0, len(results)*2)

        for chunk in result.chunks:
            deltat = chunk.deltat
            twrap = chunk.twrap
            tmins = chunk.tmins
            tmin = chunk.tmin
            tmax = chunk.tmax
            axes1.plot(
                tmins, ((tmins + twrap) % deltat) - twrap, 'o', ms=2.0)
            axes1.axvline(tmin, color='black', alpha=0.3)
            axes1.axvline(tmax, color='black', alpha=0.3)
            axes2.axvline(tmin, color='black', alpha=0.3)
            axes2.axvline(tmax, color='black', alpha=0.3)
            axes3.axvline(tmin, color='black', alpha=0.3)
            axes3.axvline(tmax, color='black', alpha=0.3)
            axes1.plot(
                [tmin, tmax], [deltat, deltat],
                color='black', alpha=0.3)
            axes1.plot(
                [tmin, tmax], [-twrap, -twrap],
                color='black', alpha=0.3)

            if chunk.deltat_est.size > 1:
                axes2.plot(
                    [chunk.tmin, chunk.tmax],
                    [chunk.mean_deviation, chunk.mean_deviation],
                    color=plot.mpl_color('scarletred1'))

                axes2.plot(
                    chunk.times,
                    chunk.deviation_ip,
                    color=plot.mpl_color('skyblue2'))
                axes3.plot(
                    chunk.times, chunk.deviation_ip,
                    color=plot.mpl_graph_color(iresult),
                    label=(
                        result.codes.channel
                        if result.codes not in have_label
                        else None))

                have_label.add(result.codes)

    axes1.axhline(0.0, color='black', alpha=0.3)
    axes3.axhline(0.0, color='black', alpha=0.3)
    axes3.legend()

    plt.show()


class Record(Object):
    offset = Int.T()
    nsamples = Int.T()
    tmin = Timestamp.T()


class Chunk(Object):
    deltat = Float.T()
    records = List.T(Record.T())
    tmin = Timestamp.T(optional=True)
    tmax = Timestamp.T(optional=True)
    twrap = Float.T(optional=True)
    times = Array.T(
        optional=True, shape=(None,), serialize_as='base64+meta')
    deltat_est = Array.T(
        optional=True, shape=(None,), serialize_as='base64+meta')
    deviation = Array.T(
        optional=True, shape=(None,), serialize_as='base64+meta')
    tmins_ip = Array.T(
        optional=True, shape=(None,), serialize_as='base64+meta')
    deltat_est_ip = Array.T(
        optional=True, shape=(None,), serialize_as='base64+meta')
    deviation_ip = Array.T(
        optional=True, shape=(None,), serialize_as='base64+meta')

    def analyse(self, eps_wrap=0.01, control_point_interval=3600 * 24):
        self.offsets = num.array([r.offset for r in self.records], dtype=int)
        self.tmins = num.array(
            [r.nsamples for r in self.records],
            dtype=util.get_time_float())

        self.tmin = self.tmins[0]
        self.tmax = self.tmins[0] + (self.offsets[-1] - 1) * self.deltat
        self.twrap = eps_wrap * self.deltat
        self.times = 0.5 * (self.tmins[:-1] + self.tmins[1:])
        self.deltat_est = num.diff(self.tmins) / num.diff(self.offsets)
        self.deviation = self.deltat_est / self.deltat - 1.0

        if self.deltat_est.size > 1:
            self.mean_deltat_est = num.mean(self.deltat_est)
            self.mean_deviation = self.mean_deltat_est / self.deltat - 1.0

            self.tmins_ip = fit_control_points(
                self.deltat,
                control_point_interval,
                self.offsets,
                self.tmins-self.tmin)

            self.deltat_est_ip = num.diff(self.tmins_ip) \
                / num.diff(self.offsets)
            self.deviation_ip = self.deltat_est_ip / self.deltat - 1.0


class ChannelResult(Object):
    codes = CodesNSLCE.T()
    chunks = List.T(Chunk.T())


class Clockdrift(SquirrelCommand):

    def make_subparser(self, subparsers):
        headline = \
            'Analyse clock drift and possibly resample recordings.'

        return subparsers.add_parser(
            'clockdrift',
            help=headline,
            description=headline)

    def setup(self, parser):
        parser.add_squirrel_selection_arguments()
        parser.add_squirrel_query_arguments()

    def run(self, parser, args):
        with progress.view():
            self.run_main(parser, args)

    def run_main(self, parser, args):
        eps_connected = 0.1

        sq = args.make_squirrel()

        codes_all = [codes for (_, _, codes, _) in sq.get_codes_info(
            'waveform', codes=args.squirrel_query['codes'])]

        by_sensor = util.group_by(lambda c: c[:3] + (c[3][:2],), codes_all)

        task_sensors = make_task('Processing sensors')
        for (scodes, codes_sensor) in task_sensors(by_sensor.items()):
            task_channels = make_task('Processing channels')
            results = []
            for icodes, codes in task_channels(list(enumerate(codes_sensor))):
                nuts = sq.iter_nuts(codes=codes)
                nuts = sorted(nuts, key=lambda nut: nut.tmin)
                paths = [nut.file_path for nut in nuts]

                chunks = []
                tr_previous = None
                offset = 0
                task_paths = make_task('Scanning mseed records')
                for path in task_paths(list(uniq(paths))):
                    for tr in mseed.iload(
                            path, load_data=False, segment_size=1):

                        if tr_previous is None or not connects(
                                tr_previous, tr, eps_connected):
                            offset = 0
                            chunks.append(Chunk(deltat=tr.deltat))

                        chunks[-1].records.append(Record(
                            offset=offset,
                            nsamples=tr.data_len(),
                            tmin=tr.tmin))

                        offset += tr.data_len()
                        tr_previous = tr

                for chunk in chunks:
                    chunk.analyse()

                results.append(ChannelResult(
                    codes=codes,
                    chunks=chunks))

            plot_results(results)


def make_subparser(subparsers):
    return subparsers.add_parser(
        'mseed',
        help=headline,
        subcommands=[Clockdrift()],
        description=description)


def setup(parser):
    pass


def run(parser, args):
    parser.print_help()
