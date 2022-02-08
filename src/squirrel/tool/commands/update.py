# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

import logging

from .. import common
# from pyrocko.squirrel.error import SquirrelError
from pyrocko.squirrel import model

logger = logging.getLogger('psq.cli.update')


def setup_subcommand(subparsers):
    return common.add_parser(
        subparsers, 'update',
        help='Update remote sources inventories.')


def setup(parser):
    common.add_selection_arguments(parser)
    common.add_query_arguments(parser)

    parser.add_argument(
        '--promises',
        action='store_true',
        dest='promises',
        default=False,
        help='Update waveform promises.')

    parser.add_argument(
        '--responses',
        action='store_true',
        dest='responses',
        default=False,
        help='Update responses.')


def call(parser, args):
    d = common.squirrel_query_from_arguments(args)
    squirrel = common.squirrel_from_selection_arguments(args)

    tmin = d.get('tmin', model.g_tmin)
    tmax = d.get('tmax', model.g_tmax)
    codes = d.get('codes', None)

    squirrel.update(tmin=tmin, tmax=tmax)
    if args.promises:
        squirrel.update_waveform_promises(tmin=tmin, tmax=tmax, codes=codes)

    if args.responses:
        squirrel.update_responses()

    stats = str(squirrel)
    stats = '\n'.join('  ' + s for s in stats.splitlines())

    logger.info('Squirrel stats:\n%s' % stats)
