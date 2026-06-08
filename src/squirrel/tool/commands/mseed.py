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
from ..common import SquirrelCommand

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


class Clockdrift(SquirrelCommand):

    def make_subparser(self, subparsers):
        headline = \
            'Analyse clock drift and possibly resample recordings.'

        return subparsers.add_parser(
            'clockdrift',
            help=headline,
            description=headline)

    def setup(self, parser):
        pass

    def run(self, parser, args):
        eps_connected = 0.1
        eps_wrap = 0.01
        control_point_interval = 3600 * 24

        sq = args.make_squirrel()

        codes_all = sq.get_codes()
        by_sensor = util.group_by(lambda c: c[:3] + (c[3][:2],), codes_all)

        for (scodes, codes) in by_sensor.items():

            p = smartplot.Plot(
                x_dims=['time'],
                y_dims=['d', 't'] * len(codes) + ['t2'])

            p.set_label('d', 'Sampling offset [s]')
            p.set_label('t', 'Rate error $\\Delta t_{est} / \\Delta t_{nom}$')
            p.set_label('t2', 'Rate error $\\Delta t_{est} / \\Delta t_{nom}$')
            p.set_label('time', '')

            for i in range(2 * len(codes)):
                plot.mpl_time_axis(p(0, i))

            plot.mpl_time_axis(p(0, len(codes)*2))

            task = make_task('Processing channels')
            for icodes, codes in task(list(enumerate(codes))):
                nuts = sq.iter_nuts(codes=codes)
                nuts = sorted(nuts, key=lambda nut: nut.tmin)
                paths = [nut.file_path for nut in nuts]

                chunks = []
                tr_previous = None
                offsets = 0
                task_paths = make_task('Scanning files')
                for path in task_paths(list(uniq(paths))):
                    for tr in mseed.iload(
                            path, load_data=False, segment_size=1):

                        if tr_previous is None or not connects(
                                tr_previous, tr, eps_connected):
                            offsets = 0
                            chunks.append((tr.deltat, []))

                        chunks[-1][1].append((offsets, tr.data_len(), tr.tmin))
                        offsets += tr.data_len()
                        tr_previous = tr

                axes1 = p(0, icodes*2)
                axes2 = p(0, icodes*2 + 1)
                axes3 = p(0, len(codes)*2)

                have_label = set()
                for deltat, data in chunks:
                    offsets, nsamples, tmins = num.array(data).T
                    tmin = tmins[0]
                    tmax = tmins[0] + (offsets[-1] - 1) * deltat
                    twrap = eps_wrap * deltat
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

                    times = 0.5 * (tmins[:-1] + tmins[1:])
                    deltat_est = num.diff(tmins) / num.diff(offsets)
                    deviation = deltat_est / deltat - 1.0
                    axes2.plot(
                        times, deviation, 'o',
                        color='black', ms=2.0, alpha=0.1)

                    if deltat_est.size > 1:
                        mean_deltat_est = num.mean(deltat_est)
                        mean_deviation = mean_deltat_est / deltat - 1.0
                        axes2.plot(
                            [tmin, tmax], [mean_deviation, mean_deviation],
                            color=plot.mpl_color('scarletred1'))

                        t0 = tmins[0]
                        tmins_ip = fit_control_points(
                            deltat,
                            control_point_interval,
                            offsets,
                            tmins-t0)

                        deltat_est_ip = num.diff(tmins_ip) / num.diff(offsets)
                        deviation_ip = deltat_est_ip / deltat - 1.0
                        axes2.plot(
                            times,
                            deviation_ip,
                            color=plot.mpl_color('skyblue2'))
                        axes3.plot(
                            times, deviation_ip,
                            color=plot.mpl_graph_color(icodes),
                            label=codes.channel if codes not in have_label
                            else None)

                        have_label.add(codes)

                axes1.axhline(0.0, color='black', alpha=0.3)
                axes3.axhline(0.0, color='black', alpha=0.3)
                axes3.legend()

            plt.show()


def make_subparser(subparsers):
    return subparsers.add_parser(
        'mseed',
        help=headline,
        subcommands=[Clockdrift],
        description=description)


def setup(parser):
    pass


def run(parser, args):
    parser.print_help()
