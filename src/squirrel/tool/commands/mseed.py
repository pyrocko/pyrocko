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
from pyrocko import plot, progress, util, signal_ext, trace
from pyrocko.plot import smartplot
from pyrocko.io import mseed, FileSaveError
from pyrocko.guts import Object, Int, Float, Timestamp, List, String
from pyrocko.guts_array import Array
from pyrocko.squirrel.error import ToolError
from ..common import SquirrelCommand
from pyrocko.model.codes import CodesNSLCE
from pyrocko.squirrel.tool.common import \
    squirrel_effective_storage_scheme_from_arguments


logger = logging.getLogger('psq.cli.mseed')

headline = 'Mini-SEED specific utilities'

description = '''%s''' % headline


def make_task(*args):
    return progress.task(*args, logger=logger)


g_filenames_all = set()


def check_append_hook(fn):
    return fn in g_filenames_all


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
    path = String.T()
    file_offset = Int.T()
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

    def analyse(self, eps_wrap=0.01, control_point_interval=3600.):
        self.offsets = num.array([r.offset for r in self.records], dtype=int)
        self.tmins = num.array(
            [r.tmin for r in self.records],
            dtype=util.get_time_float())

        self.tmin = self.tmins[0]
        self.tmax = self.tmins[0] + (self.offsets[-1] - 1) * self.deltat
        self.twrap = eps_wrap * self.deltat
        self.times = 0.5 * (self.tmins[:-1] + self.tmins[1:])
        self.deltat_est = num.diff(self.tmins) / num.diff(self.offsets)
        self.deviation = self.deltat_est / self.deltat - 1.0

        if self.deltat_est.size > 0:
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

        else:
            self.mean_deltat_est = self.deltat
            self.mean_deviation = 0.0
            self.tmins_ip = self.tmins-self.tmin
            self.deltat_est_ip = num.array([], dtype=float)
            self.deviation_ip = num.array([], dtype=float)

    def resample(
            self,
            storage_scheme,
            force=False,
            append=False,
            merge=False):

        blocksize = 100
        nsamples_polluted = 26

        nblocks = (len(self.records) - 1) // blocksize + 1
        tmin_cut = None
        task = make_task('Resampling blocks')
        for iblock in task(list(range(nblocks))):
            irecord_min = iblock * blocksize
            irecord_max = min((iblock + 1) * blocksize, len(self.records))

            if irecord_min != 0:
                irecord_min -= 1

            if irecord_max != len(self.records):
                irecord_max += 1

            records = self.records[irecord_min:irecord_max]
            tmins_ip = self.tmins_ip[irecord_min:irecord_max]
            offsets = self.offsets[irecord_min:irecord_max] \
                - self.offsets[irecord_min]

            tr_complete = None

            for record in records:
                (tr,) = list(mseed.iload(
                        record.path,
                        segment_size=1,
                        nsegments=1,
                        offset=record.file_offset))

                if tr_complete is None:
                    tr_complete = tr.copy()
                else:
                    tr_complete.append(tr.ydata)

            tmax_ip = tmins_ip[-1] \
                + self.deltat * (records[-1].nsamples - 1)

            tmin_new = num.ceil(
                (self.tmin + tmins_ip[0]) / self.deltat) * self.deltat
            tmax_new = num.floor(
                (self.tmin + tmax_ip) / self.deltat) * self.deltat

            n_new = int(round((tmax_new - tmin_new) / self.deltat))

            i_control = num.concatenate((
                offsets,
                [tr_complete.ydata.size-1]), dtype=int)

            t_control = num.concatenate((
                tmins_ip,
                [tmax_ip]),
                dtype=float) + self.tmin

            ydata_new = num.empty(n_new, dtype=float)
            signal_ext.antidrift(i_control, t_control,
                                 tr_complete.ydata.astype(float),
                                 tmin_new, self.deltat, ydata_new)

            tr_new = trace.Trace(
                network=tr_complete.network,
                station=tr_complete.station,
                location=tr_complete.location,
                channel=tr_complete.channel,
                extra=tr_complete.extra,
                deltat=self.deltat,
                tmin=tmin_new,
                ydata=ydata_new.astype(tr_complete.ydata.dtype))

            if irecord_max == len(self.records):
                tmax_cut = tr_new.tmax
            else:
                tmax_cut = tr_new.tmax - tr_new.deltat * nsamples_polluted

            if tmin_cut is None:
                tmin_cut = tr_new.tmin

            try:
                tr_new.chop(tmin_cut, tmax_cut)
                try:
                    g_filenames_all.update(storage_scheme.save(
                        [tr_new],
                        overwrite=force,
                        check_append_hook=check_append_hook if not (append or merge) else None,  # noqa
                        check_append_merge=merge))

                except FileSaveError as e:
                    raise ToolError(str(e))

                tmin_cut = tmax_cut

            except trace.NoData:
                pass


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
        parser.add_squirrel_storage_scheme_arguments()

        parser.add_argument(
            '--force',
            dest='force',
            action='store_true',
            default=False,
            help='Force overwriting of existing files.')

        parser.add_argument(
            '--append',
            dest='append',
            action='store_true',
            default=False,
            help='Append to existing files. This only works for mseed files. '
                 'Checks are preformed to ensure that appended traces have no '
                 'overlap with already existing traces.')

        parser.add_argument(
            '--merge',
            dest='merge',
            action='store_true',
            default=False,
            help='Merge with existing data in files. This only works for '
                 'mseed files.')

    def run(self, parser, args):
        with progress.view():
            self.run_main(parser, args)

    def run_main(self, parser, args):
        eps_connected = 0.3

        sq = args.make_squirrel()

        codes_all = [codes for (_, _, codes, _) in sq.get_codes_info(
            'waveform', codes=args.squirrel_query.get('codes'))]

        by_sensor = util.group_by(lambda c: c[:3] + (c[3][:2],), codes_all)

        tmin = args.squirrel_query.get('tmin')
        tmax = args.squirrel_query.get('tmax')
        storage_scheme = squirrel_effective_storage_scheme_from_arguments(args)

        task_sensors = make_task('Processing sensors')
        for (scodes, codes_sensor) in task_sensors(by_sensor.items()):
            task_channels = make_task('Processing channels')
            results = []
            for icodes, codes in task_channels(list(enumerate(codes_sensor))):
                nuts = sq.iter_nuts(codes=codes, tmin=tmin, tmax=tmax)
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
                            path=path,
                            file_offset=tr.meta['offset_start'],
                            offset=offset,
                            nsamples=tr.data_len(),
                            tmin=tr.tmin))

                        offset += tr.data_len()
                        tr_previous = tr

                task_analyse = make_task('Analysing drift')
                for chunk in task_analyse(chunks):
                    chunk.analyse()

                if storage_scheme:
                    task_resample = make_task('Resampling chunks')
                    for chunk in task_resample(chunks):
                        chunk.resample(
                            storage_scheme=storage_scheme,
                            force=args.force,
                            append=args.append,
                            merge=args.merge)

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
