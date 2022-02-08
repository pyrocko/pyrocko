# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

from .. import common


def setup_subcommand(subparsers):
    return common.add_parser(
        subparsers, 'nuts',
        help='Search indexed contents.')


def setup(parser):
    common.add_selection_arguments(parser)
    common.add_query_arguments(parser)

    parser.add_argument(
        '--contents',
        action='store_true',
        dest='print_contents',
        default=False,
        help='Print contents.')


def call(parser, args):
    d = common.squirrel_query_from_arguments(args)
    squirrel = common.squirrel_from_selection_arguments(args)
    for nut in squirrel.iter_nuts(**d):
        if args.print_contents:
            print('# %s' % nut.summary)
            print(squirrel.get_content(nut).dump())
        else:
            print(nut.summary)
