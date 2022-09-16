# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function
from pyrocko import util

headline = 'Check dataset consistency.'


def get_matching(coverages, coverage):
    matching = []
    for candidate in coverages:
        if candidate.codes == coverage.codes:
            matching.append(candidate)

    matching.sort(
        key=lambda c: (coverage.deltat == c.deltat, not c.deltat))

    matching.reverse()

    return matching


def make_subparser(subparsers):
    return subparsers.add_parser(
        'check',
        help=headline,
        description=headline + '''

A report listing potential data/metadata problems for a given data collection
is printed to standard output. The following problems are detected:

  [E1] Overlaps in channel/response epochs, waveform duplicates.
  [E2] No waveforms available for a channel/response listed in metadata.
  [E3] Channel/response information missing for an available waveform.
  [E4] Multiple channel/response entries matching an available waveform.
  [E5] Sampling rate of waveform does not match rate listed in metadata.
  [E6] Waveform is not incompletely covered by channel/response epochs.

''')


def setup(parser):
    parser.add_squirrel_selection_arguments()


def run(parser, args):
    squirrel = args.make_squirrel()

    codes_set = set()
    for kind in ['waveform', 'channel', 'response']:
        codes_set.update(squirrel.get_codes(kind=kind))

    nsl = None
    problems = False
    lines = []
    for codes in list(sorted(codes_set)):
        nsl_this = codes.codes_nsl
        if nsl is None or nsl != nsl_this:
            if lines:
                if problems:
                    print('\n'.join(lines) + '\n')
                else:
                    print(lines[0] + ' ok' + '\n')

            lines = []
            problems = False
            lines.append('%s:' % str(nsl_this))

        nsl = nsl_this

        coverage = {}
        for kind in ['waveform', 'channel', 'response']:
            coverage[kind] = squirrel.get_coverage(kind, codes=[codes])

        available = [
            kind for kind in ['waveform', 'channel', 'response']
            if coverage[kind]]

        lines.append(
            '  %s: %s' % (
                codes.channel
                + ('.%s' % codes.extra if codes.extra != '' else ''),
                ', '.join(available)))

        for kind in ['waveform', 'channel', 'response']:
            for cov in coverage[kind]:
                if any(count > 1 for (_, count) in cov.changes):
                    problems = True
                    lines.append('    - %s: %s [E1]' % (
                        kind,
                        'duplicates'
                        if kind == 'waveform' else
                        'overlapping epochs'))

        if 'waveform' not in available:
            problems = True
            lines.append('    - no waveforms [E2]')

        for cw in coverage['waveform']:
            for kind in ['channel', 'response']:
                ccs = get_matching(coverage[kind], cw)
                if not ccs:
                    problems = True
                    lines.append('    - no %s information [E3]' % kind)

                elif len(ccs) > 1:
                    problems = True
                    lines.append(
                        '    - multiple %s matches (waveform: %g Hz, %s: %s) '
                        '[E4]' % (kind, 1.0 / cw.deltat, kind, ', '.join(
                            '%g Hz' % (1.0 / cc.deltat)
                            if cc.deltat else '? Hz' for cc in ccs)))

                if ccs:
                    cc = ccs[0]
                    if cc.deltat and cc.deltat != cw.deltat:
                        lines.append(
                            '    - sampling rate mismatch '
                            '(waveform %g Hz, %s: %g Hz) [E5]' % (
                                1.0 / cw.deltat, kind, 1.0 / cc.deltat))

                    uncovered_spans = list(cw.iter_uncovered_by_combined(cc))
                    if uncovered_spans:
                        problems = True
                        lines.append(
                            '    - incompletely covered by %s [E6]:' % kind)

                        for span in uncovered_spans:
                            lines.append(
                                '      - %s - %s' % (
                                    util.time_to_str(span[0]),
                                    util.time_to_str(span[1])))

    if problems:
        print('\n'.join(lines) + '\n')
