#!/usr/bin/env python3
import numpy as np
from pyrocko.util import time_to_str as tts
from pyrocko import squirrel


def rms(data):
    return np.sqrt(np.sum(data**2))


class ReportRMSTool(squirrel.SquirrelCommand):

    def make_subparser(self, subparsers):
        return subparsers.add_parser(
            'rms', help='Report hourly RMS values.')

    def setup(self, parser):
        parser.add_squirrel_selection_arguments()

        # Add '--codes', '--tmin' and '--tmax', but not '--time'.
        parser.add_squirrel_query_arguments(without=['time'])

        parser.add_argument(
            '--fmin',
            dest='fmin',
            metavar='FLOAT',
            type=float,
            help='Corner of highpass [Hz].')

        parser.add_argument(
            '--fmax',
            dest='fmax',
            metavar='FLOAT',
            type=float,
            help='Corner of lowpass [Hz].')

    def run(self, parser, args):
        sq = args.make_squirrel()

        fmin = args.fmin
        fmax = args.fmax

        query = args.squirrel_query
        sq.update(**query)
        sq.update_waveform_promises(**query)

        for batch in sq.chopper_waveforms(
                tinc=3600.,
                tpad=1.0/fmin if fmin is not None else 0.0,
                want_incomplete=False,
                snap_window=True,
                **query):

            for tr in batch.traces:

                if fmin is not None:
                    tr.highpass(4, fmin)

                if fmax is not None:
                    tr.lowpass(4, fmax)

                tr.chop(batch.tmin, batch.tmax)
                print(tr.str_codes, tts(batch.tmin), rms(tr.ydata))


class PlotRMSTool(squirrel.SquirrelCommand):

    def make_subparser(self, subparsers):
        return subparsers.add_parser(
            'plot', help='Plot RMS traces.')

    def run(self, parser, args):
        self.fail('Not implemented yet!')


squirrel.run(
    subcommands=[ReportRMSTool(), PlotRMSTool()],
    description='My favourite RMS tools.')
