# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Implementation of :app:`squirrel merge`.
'''

import logging
from pyrocko import progress, util, trace, guts
from pyrocko.squirrel import error, model

logger = logging.getLogger('psq.cli.merge')

headline = 'Merge two waveform archives into a single one.'


def make_task(*args):
    return progress.task(*args, logger=logger)


def summarize_traces(trs):
    return ''.join('- %s\n' % tr.summary for tr in trs)


def same_sampling_rates(trs):
    return len(set(tr.deltat for tr in trs)) == 1


def snap_and_chop(tmin, tmax, traces, align=True):
    out = []

    for tr in traces:
        if align:
            dtype = tr.ydata.dtype
            tr = tr.snap(interpolate=True, inplace=False)
            tr.set_ydata(tr.get_ydata().astype(dtype))

        try:
            tr.chop(tmin, tmax)
            out.append(tr)
        except trace.NoData:
            pass

    return out


def merge_traces(traces_a, traces_b, align=True):
    # deoverlap: give precedence according to the order in which they appear
    # in the list, from high to low.
    traces = trace.deoverlap(
        traces_a + traces_b,
        precedence='first',
        snap_times_globally=align)

    # degapping: fill small gaps by linear interpolation
    return trace.degapper(traces, maxgap=5, maxlap=0)


def make_subparser(subparsers):
    return subparsers.add_parser(
        'merge',
        help=headline,
        description=headline + '''

Merge two datasets into a single one. Data gaps in the first are filled with
data from the second.
''')


def setup(parser):

    parser.add_argument(
        'dataset_a',
        metavar='DATASET_A',
        help='''
Primary dataset (higher priority): Path to YAML file with a Squirrel dataset
configuration. This file can be created using ```squirrel dataset create``` or
by modifying one of the examples from ```squirrel template```.'''.strip())

    parser.add_argument(
        'dataset_b',
        metavar='DATASET_B',
        help='''
Secondary dataset (lower priority).'''.strip())

    parser.add_argument(
        '--tinc',
        dest='tinc',
        type=float,
        metavar='SECONDS',
        help='Set time length of processing batches [s].')

    parser.add_argument(
        '--align',
        dest='align',
        action='store_true',
        help='Resample data so that samples are at exact multiples of the '
             'sampling interval with respect to UTC system time with zero on '
             '1970-01-01 00:00:00.')

    parser.add_argument(
        '--analyse-delay',
        dest='analyse_delay_path',
        metavar='PATH',
        help='Analyse time delays between A and B in regions of overlap '
             'and save results to PATH.')

    parser.add_squirrel_query_arguments(without=['time', 'kind'])

    parser.add_squirrel_storage_scheme_arguments()


def run(parser, args):
    from pyrocko.squirrel import base

    tinc = 3600.
    shift_analysis_min_samples = 500

    a = base.Squirrel(
        n_threads=getattr(args, 'n_threads', 1))

    b = base.Squirrel(
        n_threads=getattr(args, 'n_threads', 1))

    with progress.view():
        with progress.task('add datasets', n=2, logger=logger) as task:
            task.update(0, condition='Scanning dataset A')
            a.add_dataset(args.dataset_a)
            task.update(1, condition='Scanning dataset B')
            b.add_dataset(args.dataset_b)

    a_tmin, a_tmax = a.get_time_span(
        kinds=['waveform'],
        dummy_limits=False)

    if None in (a_tmin, a_tmax):
        raise error.ToolError('Dataset A: empty time span. No data?')

    b_tmin, b_tmax = b.get_time_span(
        kinds=['waveform'],
        dummy_limits=False)

    if None in (b_tmin, b_tmax):
        raise error.ToolError('Dataset B: empty time span. No data?')

    tts = util.time_to_str
    logger.info('Timespan A: %s - %s' % (tts(a_tmin), tts(a_tmax)))
    logger.info('Timespan B: %s - %s' % (tts(b_tmin), tts(b_tmax)))

    a_codes = set(a.get_codes(kind='waveform'))
    b_codes = set(b.get_codes(kind='waveform'))

    a_codes_only = a_codes - b_codes
    b_codes_only = b_codes - a_codes
    codes_both = a_codes & b_codes

    def sc(codes):
        if not codes:
            return '-'
        else:
            return ', '.join(c.safe_str for c in sorted(codes))

    logger.info('Only in A: %s' % sc(a_codes_only))
    logger.info('Only in B: %s' % sc(b_codes_only))
    logger.info('In both:   %s' % sc(codes_both))

    query = args.squirrel_query
    tmin = query.get('tmin', min(a_tmin, b_tmin))
    tmax = query.get('tmax', max(a_tmax, b_tmax))

    logger.info('Timespan: %s - %s' % (tts(tmin), tts(tmax)))

    codes_list = sorted(a_codes | b_codes)
    patterns = query.get('codes')
    if patterns is not None:
        codes_list = [
            codes for codes in codes_list
            if model.match_codes_any(patterns, codes)]

    if args.analyse_delay_path:
        analyse_delay_file = open(args.analyse_delay_path, 'w')
    else:
        analyse_delay_file = None

    storage_scheme = args.squirrel_effective_storage_scheme

    if storage_scheme is None and analyse_delay_file is None:
        raise error.ToolError(
            'Neither --analyse-delay nor --out-* options given. '
            'Nothing to do.')

    if storage_scheme is None:
        logger.info('No --out-* option selected. Only analysing delays.')

    with progress.view():
        task_outer = make_task('Processing channels')

        for codes in task_outer(codes_list):
            deltats = set()
            deltats.update(a.get_deltats(kind='waveform', codes=codes))
            deltats.update(b.get_deltats(kind='waveform', codes=codes))
            deltats = sorted(deltats)
            if len(deltats) == 0:
                continue

            if len(deltats) != 1:
                raise error.ToolError(
                    'Cannot handle multiple sampling rates (%s: %s).' % (
                        codes.safe_str,
                        ', '.join('%g' % deltat for deltat in deltats)))

            deltat = deltats[0]
            tpad = 100 * deltat

            logger.info('Processing: %s' % codes.safe_str)
            task = None
            try:
                for batch in util.iter_windows(
                        tmin=tmin,
                        tmax=tmax,
                        tinc=tinc,
                        snap_window=True):

                    if task is None:
                        task = make_task(
                            'Processing %s' % codes.safe_str, batch.n)

                    task.update(batch.i)

                    traces_a = a.get_waveforms(
                        codes=codes,
                        tmin=batch.tmin-tpad,
                        tmax=batch.tmax+tpad)
                    traces_b = b.get_waveforms(
                        codes=codes,
                        tmin=batch.tmin-tpad,
                        tmax=batch.tmax+tpad)

                    traces = traces_a + traces_b
                    if traces:
                        if not same_sampling_rates(traces):
                            raise error.ToolError(
                                'Cannot merge traces with differing sampling '
                                'rate.\n%s' % summarize_traces(traces))

                        if analyse_delay_file:
                            for tr_a in traces_a:
                                for tr_b in traces_b:
                                    try:
                                        result = trace.analyse_delay(
                                            tr_a, tr_b,
                                            shift_analysis_min_samples)

                                        guts.dump(
                                            result, stream=analyse_delay_file)

                                    except trace.TraceTooShort:
                                        pass

                        if storage_scheme:

                            traces_a = snap_and_chop(
                                batch.tmin, batch.tmax, traces_a,
                                align=args.align)

                            traces_b = snap_and_chop(
                                batch.tmin, batch.tmax, traces_b,
                                align=args.align)

                            traces_out = merge_traces(
                                traces_a, traces_b,
                                align=args.align)

                            storage_scheme.save(
                                traces_out,
                                tmin=batch.tmin,
                                tmax=batch.tmax)

                    a.advance_accessor('default', 'waveform')
                    b.advance_accessor('default', 'waveform')

            finally:
                if task:
                    task.done()

    if analyse_delay_file:
        analyse_delay_file.close()
