#!/usr/bin/env python
# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
from __future__ import absolute_import, division

import sys
import re
import os
import logging
import signal
import math
from copy import copy
from optparse import OptionParser, Option, OptionValueError

import numpy as num

from .. import util, config, pile, model, io, trace

pjoin = os.path.join

logger = logging.getLogger('pyrocko.apps.jackseis')

program_name = 'jackseis'

usage = program_name + ' <inputs> ... [options]'

description = '''A simple tool to manipulate waveform archive data.'''

tfmt = 'YYYY-mm-dd HH:MM:SS[.xxx]'
tts = util.time_to_str
stt = util.str_to_time

valid_reclengths = tuple(2**n for n in range(8, 16))


def die(message):
    sys.exit('%s: error: %s' % (program_name, message))


name_to_dtype = {
    'int32': num.int32,
    'int64': num.int64,
    'float32': num.float32,
    'float64': num.float64}


def str_to_seconds(s):
    if s.endswith('s'):
        return float(s[:-1])
    elif s.endswith('m'):
        return float(s[:-1])*60.
    elif s.endswith('h'):
        return float(s[:-1])*3600.
    elif s.endswith('d'):
        return float(s[:-1])*3600.*24.
    else:
        return float(s)


def nice_seconds_floor(s):
    nice = [1., 10., 60., 600., 3600., 3.*3600., 12*3600., 24*3600., 48*3600.]
    p = s
    for x in nice:
        if s < x:
            return p

        p = x

    return s


def check_record_length(option, opt, value):
    reclen = int(value)
    if reclen in valid_reclengths:
        return reclen
    raise OptionValueError(
        'invalid record length %d. (choose from %s)'
        % (reclen, ', '.join(str(b) for b in valid_reclengths)))


class JackseisOptions(Option):
    TYPES = Option.TYPES + ('record_length',)
    TYPE_CHECKER = copy(Option.TYPE_CHECKER)
    TYPE_CHECKER['record_length'] = check_record_length


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = OptionParser(
        usage=usage,
        description=description,
        option_class=JackseisOptions,
        formatter=util.BetterHelpFormatter())

    parser.add_option(
        '--format',
        dest='format',
        default='detect',
        choices=io.allowed_formats('load'),
        help='assume input files are of given FORMAT. Choices: %s' %
             io.allowed_formats('load', 'cli_help', 'detect'))

    parser.add_option(
        '--pattern',
        dest='regex',
        metavar='REGEX',
        help='only include files whose paths match REGEX')

    parser.add_option(
        '--stations',
        dest='station_fns',
        action='append',
        default=[],
        metavar='STATIONS',
        help='read station information from file STATIONS')

    parser.add_option(
        '--event', '--events',
        dest='event_fns',
        action='append',
        default=[],
        metavar='EVENT',
        help='read event information from file EVENT')

    parser.add_option(
        '--cache',
        dest='cache_dir',
        default=config.config().cache_dir,
        metavar='DIR',
        help='use directory DIR to cache trace metadata '
             '(default=\'%default\')')

    parser.add_option(
        '--quiet',
        dest='quiet',
        action='store_true',
        default=False,
        help='disable output of progress information')

    parser.add_option(
        '--debug',
        dest='debug',
        action='store_true',
        default=False,
        help='print debugging information to stderr')

    parser.add_option(
        '--tmin',
        dest='tmin',
        help='start time as "%s"' % tfmt)

    parser.add_option(
        '--tmax',
        dest='tmax',
        help='end time as "%s"' % tfmt)

    parser.add_option(
        '--tinc',
        dest='tinc',
        help='set time length of output files [s] or "auto" to automatically '
             'choose an appropriate time increment or "huge" to allow to use '
             'a lot of system memory to merge input traces into huge output '
             'files.')

    parser.add_option(
        '--downsample',
        dest='downsample',
        metavar='RATE',
        help='downsample to RATE [Hz]')

    parser.add_option(
        '--output',
        dest='output_path',
        metavar='TEMPLATE',
        help='set output path to TEMPLATE. Available placeholders '
             'are %n: network, %s: station, %l: location, %c: channel, '
             '%b: begin time, %e: end time, %j: julian day of year. The '
             'following additional placeholders use the window begin and end '
             'times rather than trace begin and end times (to suppress '
             'producing many small files for gappy traces), %(wmin_year)s, '
             '%(wmin_month)s, %(wmin_day)s, %(wmin)s, %(wmax_year)s, '
             '%(wmax_month)s, %(wmax_day)s, %(wmax)s. Example: '
             '--output=\'data/%s/trace-%s-%c.mseed\'')

    parser.add_option(
        '--output-dir',
        metavar='TEMPLATE',
        dest='output_dir',
        help='set output directory to TEMPLATE (see --output for details) '
             'and automatically choose filenames. '
             'Produces files like TEMPLATE/NET-STA-LOC-CHA_BEGINTIME.FORMAT')

    parser.add_option(
        '--output-format',
        dest='output_format',
        default='mseed',
        choices=io.allowed_formats('save'),
        help='set output file format. Choices: %s' %
             io.allowed_formats('save', 'cli_help', 'mseed'))

    parser.add_option(
        '--force',
        dest='force',
        action='store_true',
        default=False,
        help='force overwriting of existing files')

    parser.add_option(
        '--no-snap',
        dest='snap',
        action='store_false',
        default=True,
        help='do not start output files at even multiples of file length')

    parser.add_option(
        '--traversal',
        dest='traversal',
        choices=('station-by-station', 'channel-by-channel', 'chronological'),
        default='station-by-station',
        help='set traversal order for traces processing. '
             'Choices are \'station-by-station\' [default], '
             '\'channel-by-channel\', and \'chronological\'. Chronological '
             'traversal uses more processing memory but makes it possible to '
             'join multiple stations into single output files')

    parser.add_option(
        '--rename-network',
        action='append',
        default=[],
        dest='rename_network',
        metavar='/PATTERN/REPLACEMENT/',
        help='update network code, can be given more than once')

    parser.add_option(
        '--rename-station',
        action='append',
        default=[],
        dest='rename_station',
        metavar='/PATTERN/REPLACEMENT/',
        help='update station code, can be given more than once')

    parser.add_option(
        '--rename-location',
        action='append',
        default=[],
        dest='rename_location',
        metavar='/PATTERN/REPLACEMENT/',
        help='update location code, can be given more than once')

    parser.add_option(
        '--rename-channel',
        action='append',
        default=[],
        dest='rename_channel',
        metavar='/PATTERN/REPLACEMENT/',
        help='update channel code, can be given more than once')

    parser.add_option(
        '--output-data-type',
        dest='output_data_type',
        choices=('same', 'int32', 'int64', 'float32', 'float64'),
        default='same',
        metavar='DTYPE',
        help='set data type. Choices: same [default], int32, '
             'int64, float32, float64. The output file format must support '
             'the given type.')

    parser.add_option(
        '--record-length',
        dest='record_length',
        type='record_length',
        default=4096,
        metavar='RECORD_LENGTH',
        help='set the mseed record length in bytes. Choices: %s. '
             'Default is 4096 bytes, which is commonly used for archiving.'
             % ', '.join(str(2**n) for n in range(8, 16)))

    (options, args) = parser.parse_args(args)

    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    if options.debug:
        util.setup_logging(program_name, 'debug')
    elif options.quiet:
        util.setup_logging(program_name, 'warning')
    else:
        util.setup_logging(program_name, 'info')

    tinc = None
    if options.tinc is not None:
        if tinc in ('huge', 'auto'):
            tinc = options.tinc
        else:
            try:
                tinc = str_to_seconds(options.tinc)
            except Exception:
                die('invalid argument to --tinc')

    tmin = None
    if options.tmin is not None:
        try:
            tmin = stt(options.tmin)
        except Exception:
            die('invalid argument to --tmin. '
                'Expected format is ""')

    tmax = None
    if options.tmax is not None:
        try:
            tmax = stt(options.tmax)
        except Exception:
            die('invalid argument to --tmax. '
                'Expected format is "%s"' % tfmt)

    target_deltat = None
    if options.downsample is not None:
        try:
            target_deltat = 1.0 / float(options.downsample)
        except Exception:
            die('invalid argument to --downsample')

    replacements = []
    for k, rename_k, in [
            ('network', options.rename_network),
            ('station', options.rename_station),
            ('location', options.rename_location),
            ('channel', options.rename_channel)]:

        for patrep in rename_k:
            m = re.match(r'/([^/]+)/([^/]*)/', patrep)
            if not m:
                die('invalid argument to --rename-%s. '
                    'Expected format is /PATTERN/REPLACEMENT/' % k)

            replacements.append((k, m.group(1), m.group(2)))

    stations = []
    for stations_fn in options.station_fns:
        stations.extend(model.load_stations(stations_fn))

    events = []
    for event_fn in options.event_fns:
        events.extend(model.load_events(event_fn))

    p = pile.make_pile(
        paths=args,
        selector=None,
        regex=options.regex,
        fileformat=options.format,
        cachedirname=options.cache_dir,
        show_progress=not options.quiet)

    if p.tmin is None:
        die('data selection is empty')

    if tinc == 'auto':
        tinc = nice_seconds_floor(p.get_deltatmin() * 1000000.)

    if options.snap:
        if tmin is None:
            tmin = p.tmin

        if tinc is not None:
            tmin = int(math.floor(tmin / tinc)) * tinc

    output_path = options.output_path
    output_dir = options.output_dir

    if output_path and not output_dir and os.path.isdir(output_path):
        output_dir = output_path  # compat. with old behaviour

    if output_dir and not output_path:
        output_path = 'trace_%(network)s-%(station)s-' \
                      '%(location)s-%(channel)s_%(wmin)s.' + \
                      options.output_format

    if output_dir and output_path:
        output_path = pjoin(output_dir, output_path)

    if not output_path:
        die('--output not given')

    tpad = 0.
    if target_deltat is not None:
        tpad = target_deltat * 10.

    if tinc is None:
        if ((tmax or p.tmax) - (tmin or p.tmin)) \
                / p.get_deltatmin() > 100000000.:
            die('use --tinc=huge to really produce such large output files '
                'or use --tinc=INC to split into smaller files.')

    kwargs = dict(tmin=tmin, tmax=tmax, tinc=tinc, tpad=tpad)

    if options.traversal == 'channel-by-channel':
        it = p.chopper_grouped(gather=lambda tr: tr.nslc_id, **kwargs)

    elif options.traversal == 'station-by-station':
        it = p.chopper_grouped(gather=lambda tr: tr.nslc_id[:2], **kwargs)

    else:
        it = p.chopper(**kwargs)

    abort = []

    def got_sigint(signum, frame):
        abort.append(True)

    old = signal.signal(signal.SIGINT, got_sigint)

    save_kwargs = {}
    if options.output_format == 'mseed':
        save_kwargs['record_length'] = options.record_length

    for traces in it:
        if traces:
            twmin = min(tr.wmin for tr in traces)
            twmax = max(tr.wmax for tr in traces)
            logger.info('processing %s - %s, %i traces' %
                        (tts(twmin), tts(twmax), len(traces)))

            if target_deltat is not None:
                out_traces = []
                for tr in traces:
                    try:
                        tr.downsample_to(
                            target_deltat, snap=True, demean=False)

                        if options.output_data_type == 'same':
                            tr.ydata = tr.ydata.astype(tr.ydata.dtype)

                        tr.chop(tr.wmin, tr.wmax)
                        out_traces.append(tr)

                    except (trace.TraceTooShort, trace.NoData):
                        pass

                traces = out_traces

            if options.output_data_type != 'same':
                for tr in traces:
                    tr.ydata = tr.ydata.astype(
                        name_to_dtype[options.output_data_type])

            if replacements:
                for tr in traces:
                    r = {}
                    for k, pat, repl in replacements:
                        oldval = getattr(tr, k)
                        newval, n = re.subn(pat, repl, oldval)
                        if n:
                            r[k] = newval

                    tr.set_codes(**r)

            if output_path:
                try:
                    io.save(traces, output_path, format=options.output_format,
                            overwrite=options.force,
                            additional=dict(
                                wmin_year=tts(twmin, format='%Y'),
                                wmin_month=tts(twmin, format='%m'),
                                wmin_day=tts(twmin, format='%d'),
                                wmin=tts(twmin, format='%Y-%m-%d_%H-%M-%S'),
                                wmax_year=tts(twmax, format='%Y'),
                                wmax_month=tts(twmax, format='%m'),
                                wmax_day=tts(twmax, format='%d'),
                                wmax=tts(twmax, format='%Y-%m-%d_%H-%M-%S')),
                            **save_kwargs)
                except io.FileSaveError as e:
                    die(str(e))

        if abort:
            break

    signal.signal(signal.SIGINT, old)

    if abort:
        die('interrupted.')


if __name__ == '__main__':
    main(sys.argv[1:])
