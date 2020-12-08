# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

from pyrocko import io
from pyrocko.apps.jackseis import process, tfmt

from .. import common


def setup(subparser):
    p = common.add_parser(
        subparser, 'jackseis',
        help='squirrel\'s adaption of jackseis')

    p.add_argument(
        '--pattern',
        dest='regex',
        metavar='REGEX',
        help='only include files whose paths match REGEX')

    p.add_argument(
        '--quiet',
        dest='quiet',
        action='store_true',
        default=False,
        help='disable output of progress information')

    p.add_argument(
        '--debug',
        dest='debug',
        action='store_true',
        default=False,
        help='print debugging information to stderr')

    p.add_argument(
        '--tmin',
        dest='tmin',
        help='start time as "%s"' % tfmt)

    p.add_argument(
        '--tmax',
        dest='tmax',
        help='end time as "%s"' % tfmt)

    p.add_argument(
        '--tinc',
        dest='tinc',
        help='set time length of output files [s] or "auto" to automatically '
             'choose an appropriate time increment or "huge" to allow to use '
             'a lot of system memory to merge input traces into huge output '
             'files.')

    p.add_argument(
        '--downsample',
        dest='downsample',
        metavar='RATE',
        help='downsample to RATE [Hz]')

    p.add_argument(
        '--output',
        dest='output_path',
        metavar='TEMPLATE',
        help='set output path to TEMPLATE. Available placeholders '
             'are %%n: network, %%s: station, %%l: location, %%c: channel, '
             '%%b: begin time, %%e: end time, %%j: julian day of year. The '
             'following additional placeholders use the window begin and end '
             'times rather than trace begin and end times (to suppress '
             'producing many small files for gappy traces), %%(wmin_year)s, '
             '%%(wmin_month)s, %%(wmin_day)s, %%(wmin)s, %%(wmax_year)s, '
             '%%(wmax_month)s, %%(wmax_day)s, %%(wmax)s. Example: '
             '--output=\'data/%%s/trace-%%s-%%c.mseed\'')

    p.add_argument(
        '--output-dir',
        metavar='TEMPLATE',
        dest='output_dir',
        help='set output directory to TEMPLATE (see --output for details) '
             'and automatically choose filenames. '
             'Produces files like TEMPLATE/NET-STA-LOC-CHA_BEGINTIME.FORMAT')

    p.add_argument(
        '--output-format',
        dest='output_format',
        default='mseed',
        choices=io.allowed_formats('save'),
        help='set output file format. Choices: %s' %
             io.allowed_formats('save', 'cli_help', 'mseed'))

    p.add_argument(
        '--force',
        dest='force',
        action='store_true',
        default=False,
        help='force overwriting of existing files')

    p.add_argument(
        '--no-snap',
        dest='snap',
        action='store_false',
        default=True,
        help='do not start output files at even multiples of file length')

    p.add_argument(
        '--traversal',
        dest='traversal',
        choices=('station-by-station', 'channel-by-channel', 'chronological'),
        default='station-by-station',
        help='set traversal order for traces processing. '
             'Choices are \'station-by-station\' [default], '
             '\'channel-by-channel\', and \'chronological\'. Chronological '
             'traversal uses more processing memory but makes it possible to '
             'join multiple stations into single output files')

    p.add_argument(
        '--rename-network',
        action='append',
        default=[],
        dest='rename_network',
        metavar='/PATTERN/REPLACEMENT/',
        help='update network code, can be given more than once')

    p.add_argument(
        '--rename-station',
        action='append',
        default=[],
        dest='rename_station',
        metavar='/PATTERN/REPLACEMENT/',
        help='update station code, can be given more than once')

    p.add_argument(
        '--rename-location',
        action='append',
        default=[],
        dest='rename_location',
        metavar='/PATTERN/REPLACEMENT/',
        help='update location code, can be given more than once')

    p.add_argument(
        '--rename-channel',
        action='append',
        default=[],
        dest='rename_channel',
        metavar='/PATTERN/REPLACEMENT/',
        help='update channel code, can be given more than once')

    p.add_argument(
        '--output-data-type',
        dest='output_data_type',
        choices=('same', 'int32', 'int64', 'float32', 'float64'),
        default='same',
        metavar='DTYPE',
        help='set data type. Choices: same [default], int32, '
             'int64, float32, float64. The output file format must support '
             'the given type.')

    common.add_selection_arguments(p)
    return p


def call(parser, args):

    def get_pile():
        squirrel = common.squirrel_from_selection_arguments(args)
        return squirrel.pile

    args.station_fns = []
    args.event_fns = []

    return process(get_pile, args)
