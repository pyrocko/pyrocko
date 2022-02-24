# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

import math
import logging

from pyrocko.squirrel.error import SquirrelError
from pyrocko.progress import progress

logger = logging.getLogger('psq.cli.summon')


def make_subparser(subparsers):
    return subparsers.add_parser(
        'summon',
        help='Fill local cache.')


def setup(parser):
    parser.add_squirrel_selection_arguments()
    parser.add_squirrel_query_arguments()


def run(parser, args):
    d = args.squirrel_query
    squirrel = args.make_squirrel()

    if 'tmin' not in d or 'tmax' not in d:
        raise SquirrelError('Time span required.')

    tinc = 3600.

    with progress.view():
        nwindows = int(math.ceil((d['tmax'] - d['tmin']) / tinc))
        task = progress.task('Summoning', nwindows)
        iwindow = 0
        for trs in squirrel.chopper_waveforms(
                tinc=tinc, load_data=False, **d):

            iwindow += 1
            task.update(iwindow)

        task.done()

    stats = str(squirrel)
    stats = '\n'.join('  ' + s for s in stats.splitlines())

    logger.info('Squirrel stats:\n%s' % stats)
