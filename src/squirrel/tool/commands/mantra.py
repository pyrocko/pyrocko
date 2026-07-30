# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Implementation of :app:`squirrel mantra`.
'''

import logging

from pyrocko import progress, squirrel, util
from pyrocko.carpet import CarpetOverlapError
from pyrocko.squirrel.error import ToolError
from ..common import SquirrelCommand, add_mantra_arguments, \
    get_mantras_from_arguments

headline = 'Processing pipelines and operator management.'

description = '''%s''' % headline

logger = logging.getLogger('main')


class Show(SquirrelCommand):

    def make_subparser(self, subparsers):
        headline = \
            'Inspect available operator mappings.'

        return subparsers.add_parser(
            'show',
            help=headline,
            description=headline)

    def setup(self, parser):
        add_mantra_arguments(parser)
        parser.add_sensor_array_arguments()
        parser.add_squirrel_selection_arguments()
        parser.add_squirrel_query_arguments()

    def run(self, parser, args):
        with progress.view():
            self.run_main(parser, args)

    def run_main(self, parser, args):
        mantras = get_mantras_from_arguments(args)
        if not mantras:
            raise ToolError('No mantra given. Use --mantra option.')

        sq = args.make_squirrel(check_have_arrays=False)
        for mantra in mantras:
            mantra.setup(sq)
            print(mantra.describe())


class Process(SquirrelCommand):

    def make_subparser(self, subparsers):
        headline = \
            'Apply operator to data.'

        return subparsers.add_parser(
            'process',
            help=headline,
            description=headline)

    def setup(self, parser):
        add_mantra_arguments(parser)
        parser.add_sensor_array_arguments()

        parser.add_argument(
            '--tinc',
            dest='tinc',
            type=util.parse_duration,
            metavar='DURATION',
            default=3600.,
            help='Set processing batch size [s].')

        parser.add_argument(
            '--out',
            dest='out_storage_path',
            metavar='PATH',
            help='Store output in directory PATH.')

        parser.add_squirrel_selection_arguments()
        parser.add_squirrel_query_arguments()

    def run(self, parser, args):
        with progress.view():
            self.run_main(parser, args)

    def run_main(self, parser, args):
        mantras = get_mantras_from_arguments(args)
        if not mantras:
            raise ToolError('No mantra given. Use --mantra option.')

        storage = squirrel.get_storage_scheme('rug-store-100')
        if not args.out_storage_path:
            raise squirrel.ToolError(
                'Specify output storage directory with --out')

        storage.set_base_path(args.out_storage_path)

        sq = args.make_squirrel()
        for mantra in mantras:
            mantra.setup(sq)

            sq_tmin, sq_tmax = mantra.outlet.get_time_span('carpet')

            tmin = args.squirrel_query.get('tmin', sq_tmin)
            tmax = args.squirrel_query.get('tmax', sq_tmax)
            tinc = args.tinc

            task = progress.task('Processing time window', logger=logger)
            for batch in task(mantra.outlet.chopper_carpets(
                    tmin=tmin,
                    tmax=tmax,
                    tinc=tinc,
                    codes=args.squirrel_query.get('codes'),
                    snap_window=True)):

                carpets = []
                for carpet in batch.carpets:
                    carpet.codes = carpet.codes.replace(extra=mantra.name)
                    carpets.append(carpet)

                if args.out_storage_path:
                    try:
                        storage.save_carpets(carpets)
                    except CarpetOverlapError as e:
                        raise squirrel.ToolError(str(e))


def make_subparser(subparsers):
    return subparsers.add_parser(
        'mantra',
        help=headline,
        subcommands=[Show(), Process()],
        description=description)


def setup(parser):
    pass


def run(parser, args):
    parser.print_help()
