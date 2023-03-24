# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from pyrocko import io
from pyrocko.apps.jackseis import process, tfmt

headline = "Squirrel's adaption of classic Jackseis."


def make_subparser(subparsers):
    return subparsers.add_parser(
        'jackseis-classic',
        help=headline,
        description=headline)


def setup(parser):
    parser.add_argument(
        '--pattern',
        dest='regex',
        metavar='REGEX',
        help='only include files whose paths match REGEX')

    parser.add_argument(
        '--quiet',
        dest='quiet',
        action='store_true',
        default=False,
        help='disable output of progress information')

    parser.add_argument(
        '--debug',
        dest='debug',
        action='store_true',
        default=False,
        help='print debugging information to stderr')

    parser.add_argument(
        '--tmin',
        dest='tmin',
        help='start time as "%s"' % tfmt)

    parser.add_argument(
        '--tmax',
        dest='tmax',
        help='end time as "%s"' % tfmt)

    parser.add_argument(
        '--tinc',
        dest='tinc',
        help='set time length of output files [s] or "auto" to automatically '
             'choose an appropriate time increment or "huge" to allow to use '
             'a lot of system memory to merge input traces into huge output '
             'files.')

    sample_snap_choices = ('shift', 'interpolate')
    parser.add_argument(
        '--sample-snap',
        dest='sample_snap',
        choices=sample_snap_choices,
        help='shift/interpolate traces so that samples are at even multiples '
        'of sampling rate. Choices: %s' % ', '.join(sample_snap_choices))

    parser.add_argument(
        '--downsample',
        dest='downsample',
        metavar='RATE',
        help='downsample to RATE [Hz]')

    parser.add_argument(
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
             "--output='data/%%s/trace-%%s-%%c.mseed'")

    parser.add_argument(
        '--output-dir',
        metavar='TEMPLATE',
        dest='output_dir',
        help='set output directory to TEMPLATE (see --output for details) '
             'and automatically choose filenames. '
             'Produces files like TEMPLATE/NET-STA-LOC-CHA_BEGINTIME.FORMAT')

    parser.add_argument(
        '--output-format',
        dest='output_format',
        default='mseed',
        choices=io.allowed_formats('save'),
        help='set output file format. Choices: %s' %
             io.allowed_formats('save', 'cli_help', 'mseed'))

    parser.add_argument(
        '--force',
        dest='force',
        action='store_true',
        default=False,
        help='force overwriting of existing files')

    parser.add_argument(
        '--no-snap',
        dest='snap',
        action='store_false',
        default=True,
        help='do not start output files at even multiples of file length')

    parser.add_argument(
        '--traversal',
        dest='traversal',
        choices=('station-by-station', 'channel-by-channel', 'chronological'),
        default='station-by-station',
        help='set traversal order for traces processing. '
             "Choices are 'station-by-station' [default], "
             "'channel-by-channel', and 'chronological'. Chronological "
             'traversal uses more processing memory but makes it possible to '
             'join multiple stations into single output files')

    parser.add_argument(
        '--rename-network',
        action='append',
        default=[],
        dest='rename_network',
        metavar='/PATTERN/REPLACEMENT/',
        help='update network code, can be given more than once')

    parser.add_argument(
        '--rename-station',
        action='append',
        default=[],
        dest='rename_station',
        metavar='/PATTERN/REPLACEMENT/',
        help='update station code, can be given more than once')

    parser.add_argument(
        '--rename-location',
        action='append',
        default=[],
        dest='rename_location',
        metavar='/PATTERN/REPLACEMENT/',
        help='update location code, can be given more than once')

    parser.add_argument(
        '--rename-channel',
        action='append',
        default=[],
        dest='rename_channel',
        metavar='/PATTERN/REPLACEMENT/',
        help='update channel code, can be given more than once')

    parser.add_argument(
        '--output-data-type',
        dest='output_data_type',
        choices=('same', 'int32', 'int64', 'float32', 'float64'),
        default='same',
        metavar='DTYPE',
        help='set data type. Choices: same [default], int32, '
             'int64, float32, float64. The output file format must support '
             'the given type.')

    parser.add_argument(
        '--output-record-length',
        dest='record_length',
        default=4096,
        choices=[b for b in io.mseed.VALID_RECORD_LENGTHS],
        type=int,
        metavar='RECORD_LENGTH',
        help='set the mseed record length in bytes. Choices: %s. '
             'Default is 4096 bytes, which is commonly used for archiving.'
             % ', '.join(str(b) for b in io.mseed.VALID_RECORD_LENGTHS))

    parser.add_argument(
        '--output-steim',
        dest='steim',
        choices=[1, 2],
        default=2,
        type=int,
        metavar='STEIM_COMPRESSION',
        help='set the mseed STEIM compression. Choices: 1 or 2. '
             'Default is STEIM-2, which can compress full range int32. '
             'NOTE: STEIM-2 is limited to 30 bit dynamic range.')

    quantity_choices = ('acceleration', 'velocity', 'displacement')
    parser.add_argument(
        '--output-quantity',
        dest='output_quantity',
        choices=quantity_choices,
        help='deconvolve instrument transfer function. Choices: %s'
        % ', '.join(quantity_choices))

    parser.add_argument(
        '--restitution-frequency-band',
        default='0.001,100.0',
        dest='str_fmin_fmax',
        metavar='FMIN,FMAX',
        help='frequency band for instrument deconvolution as FMIN,FMAX in Hz. '
             'Default: "%(default)s"')

    parser.add_argument(
        '--nthreads',
        metavar='NTHREADS',
        default=1,
        help='number of threads for processing, '
             'this can speed-up CPU bound tasks (Python 3.5+ only)')

    parser.add_squirrel_selection_arguments()


def run(parser, args):

    def get_pile():
        squirrel = args.make_squirrel()
        return squirrel.pile

    args.station_fns = []
    args.event_fns = []
    args.station_xml_fns = []
    args.record_length = int(args.record_length)

    return process(get_pile, args)
