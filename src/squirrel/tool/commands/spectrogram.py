# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Implementation of :app:`squirrel spectrogram`.
'''

import logging

from pyrocko import progress, util
from pyrocko.squirrel.error import ToolError
from pyrocko.squirrel.storage import get_storage_scheme
from pyrocko.squirrel.model import QuantityType
from pyrocko.carpet import OverlappingCarpets
from pyrocko.squirrel.tool.common import ldq

logger = logging.getLogger('psq.cli.spectrogram')

headline = 'Calculate multi-resolution spectrograms.'


def make_subparser(subparsers):
    return subparsers.add_parser(
        'spectrogram',
        help=headline,
        description=headline)


def setup(parser):
    parser.add_squirrel_selection_arguments()
    parser.add_squirrel_query_arguments()

    parser.add_argument(
        '--block-size-exponent',
        dest='nblock_exponent',
        type=int,
        metavar='INT',
        default=10,
        help='Set block size used to calculate FFTs as a power-of-two '
             'exponent. Default: 10 (2**10 = 1024).')

    parser.add_argument(
        '--levels',
        dest='nlevels',
        type=int,
        metavar='INT',
        default=9,
        help='Set number of levels for the multi-resolution spectrograms.')

    parser.add_argument(
        '--weighting-exponent',
        dest='weighting_exponent',
        type=int,
        metavar='INT',
        default=4,
        help='Set exponent for the the weighting function which is used to '
             'merge the multi-resolution spectrogram components.')

    interpolation_choices = ['cos', 'nearest_neighbor']
    parser.add_argument(
        '--interpolation',
        dest='interpolation',
        choices=interpolation_choices,
        default='cos',
        help='Set interpolation method for multi-resolution spectrogram '
             'stitching. Choices: %s, Default: ``cos``'
             % ldq(interpolation_choices))

    parser.add_argument(
        '--tinc',
        dest='tinc',
        type=util.parse_duration,
        metavar='DURATION',
        default=3600.,
        help='Set time length of output files [s].')

    parser.add_argument(
        '--no-snap-window',
        dest='snap_window',
        default=True,
        action='store_false',
        help='Do not snap windows to multiples of the processing interval')

    parser.add_argument(
        '--show-construction',
        dest='show_construction',
        action='store_true',
        default=False,
        help='Show multi-resolutiion spectrogram construction plots (for '
             'debugging purposes).')

    parser.add_argument(
        '--quantity',
        dest='quantity',
        default='velocity',
        choices=QuantityType.choices,
        metavar='QUANTITY',
        help='Apply instrument correction for given ``QUANTITY``. Compatible '
             'instrument response information must be avaible. Set to '
             '``counts`` to disable. Choices: %s. Default: ``velocity``.'
             % ldq(QuantityType.choices))

    parser.add_argument(
        '--out-storage-path',
        dest='out_storage_path',
        metavar='PATH',
        help='Store output in directory PATH.')


def run(parser, args):
    from pyrocko.squirrel import MultiSpectrogramOperator, Pow2Windowing

    d = args.squirrel_query
    squirrel = args.make_squirrel()

    musop = MultiSpectrogramOperator(
        quantity=args.quantity,
        interpolation=args.interpolation,
        windowing=Pow2Windowing(
            nblock=2**args.nblock_exponent,
            nlevels=args.nlevels,
            weighting_exponent=args.weighting_exponent))

    musop.set_input(squirrel)

    sq_tmin, sq_tmax = squirrel.get_time_span(['waveform'], dummy_limits=False)
    if None in (sq_tmin, sq_tmax):
        raise ToolError(
            'No data. Add data with --add, --dataset or --persistent.')

    storage = get_storage_scheme('rug-store-100')
    if not args.out_storage_path and not args.show_construction:
        raise ToolError(
            'Specify output storage directory with --out-storage-path or use '
            '--show-construction')

    storage.set_base_path(args.out_storage_path)

    def log_startup(d):
        logger.info(
            'Processing time span %s - %s with %i windows of %s.',
            util.time_to_str(d['tmin']),
            util.time_to_str(d['tmax']),
            d['nwin'],
            util.str_duration(d['tinc']))

    with progress.view():
        task = progress.task('Calculating spectrograms', logger=logger)
        for batch in task(util.iter_windows(
                tmin=d.get('tmin', None),
                tmax=d.get('tmax', None),
                tinc=args.tinc,
                tpad=0.0,
                snap_window=args.snap_window,
                tmin_content=sq_tmin,
                tmax_content=sq_tmax,
                hook_startup=log_startup)):

            carpets = musop.get_carpets(
                d.get('codes'), tmin=batch.tmin, tmax=batch.tmax,
                show_construction=args.show_construction)

            for carpet in carpets:
                if args.out_storage_path:
                    try:
                        storage.save_carpets(carpet)
                    except OverlappingCarpets as e:
                        raise ToolError(str(e))
