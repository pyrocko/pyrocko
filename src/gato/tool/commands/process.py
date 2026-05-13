# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Implementation of :app:`gato acme`.
'''

import sys  # noqa
import logging

from pyrocko import squirrel, guts, util, gato
from pyrocko.carpet import OverlappingCarpets
from pyrocko import progress

guts_prefix = 'gato'

days = 24. * 60. * 60.
km = 1000.


logger = logging.getLogger('main')

headline = 'Run array processing.'

description = '''
TODO description
'''


def make_subparser(subparsers):
    return subparsers.add_parser(
        'process',
        help=headline,
        description=headline + '\n\n' + description)


def setup(parser):

    parser.add_argument(
        '--mantra',
        dest='mantra_path',
        metavar='PATH',
        help='Configuration for processing setup. Run '
             '`gato mantra` to obtain default configuration files.')

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

    parser.add_argument(
        '--debug-operators',
        dest='debug_operators',
        action='store_true',
        help='Show operator mappings and exit.')

    gato.add_array_selection_arguments(parser)
    parser.add_squirrel_selection_arguments()
    parser.add_squirrel_query_arguments(without=['time', 'kinds'])


def run(parser, args):

    with progress.view():

        if args.mantra_path:
            mantras = guts.load_all(filename=args.mantra_path)
            for mantra in mantras:
                if not isinstance(mantra, squirrel.Mantra):
                    raise squirrel.ToolError(
                        'Configuration file must contain Mantra objects.')

        else:
            raise squirrel.ToolError('No --mantra defined.')

        arrays = gato.get_matching_arrays(
            args.array_names, args.array_paths, args.use_builtin_arrays)

        if not arrays:
            logger.info(
                'No array configuration specified. Creating ad-hoc array '
                'with all data: array0')

            arrays = {
                'array0': gato.SensorArray(
                    name='array0',
                    codes=['*.*.*.*.*']),
            }

        storage = squirrel.get_storage_scheme('rug-store-100')
        if not args.out_storage_path:
            raise squirrel.ToolError(
                'Specify output storage directory with --out')

        storage.set_base_path(args.out_storage_path)

        sq = args.make_squirrel()
        print(sq)

        arrays = [arrays[name] for name in sorted(arrays.keys())]

        for mantra in mantras:
            mantra.setup(sq, arrays)

            if args.debug_operators:
                mantra.print_operator_mappings()
                continue

            sq_tmin, sq_tmax = sq.get_time_span('waveform')

            tmin = args.squirrel_query.get('tmin', sq_tmin)
            tmax = args.squirrel_query.get('tmax', sq_tmax)
            tinc = args.tinc

            task = progress.task('Processing time window', logger=logger)
            for batch in task(mantra.outlet.chopper_carpets(
                    tmin=tmin,
                    tmax=tmax,
                    tinc=tinc,
                    snap_window=True)):

                carpets = []
                for carpet in batch.carpets:
                    carpet.codes = carpet.codes.replace(extra=mantra.name)
                    carpets.append(carpet)

                if args.out_storage_path:
                    try:
                        storage.save_carpets(carpets)
                    except OverlappingCarpets as e:
                        raise squirrel.ToolError(str(e))
