# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

import math
import logging

from pyrocko.squirrel.error import SquirrelError
from pyrocko.progress import progress

logger = logging.getLogger('psq.cli.summon')

headline = 'Fill local cache.'


def make_subparser(subparsers):
    return subparsers.add_parser(
        'summon',
        help=headline,
        description=headline)


def setup(parser):
    parser.add_squirrel_selection_arguments()
    parser.add_squirrel_query_arguments()
    parser.add_argument(
        '--channel-priorities',
        dest='channel_priorities',
        metavar='CHA',
        help='TODO')


def run(parser, args):
    d = args.squirrel_query
    squirrel = args.make_squirrel()

    if 'tmin' not in d or 'tmax' not in d:
        raise SquirrelError('Time span required.')

    tinc = 3600.

    channel_priorities = None
    if args.channel_priorities:
        channel_priorities = [
            cha.strip() for cha in args.channel_priorities.split(',')]

    with progress.view():
        nwindows = int(math.ceil((d['tmax'] - d['tmin']) / tinc))
        task = progress.task('Summoning', nwindows)
        iwindow = 0
        for trs in squirrel.chopper_waveforms(
                tinc=tinc,
                load_data=False,
                channel_priorities=channel_priorities,
                **d):

            iwindow += 1
            task.update(iwindow)

        task.done()

    stats = str(squirrel)
    stats = '\n'.join('  ' + s for s in stats.splitlines())

    logger.info('Squirrel stats:\n%s' % stats)
