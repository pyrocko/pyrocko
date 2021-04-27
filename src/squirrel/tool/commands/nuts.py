# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

from .. import common


def setup(subparsers):
    p = common.add_parser(
        subparsers, 'nuts',
        help='Search indexed contents.')

    common.add_selection_arguments(p)
    common.add_query_arguments(p)

    p.add_argument(
        '--contents',
        action='store_true',
        dest='print_contents',
        default=False,
        help='Print contents.')

    return p


def call(parser, args):
    d = common.squirrel_query_from_arguments(args)
    squirrel = common.squirrel_from_selection_arguments(args)
    for nut in squirrel.iter_nuts(**d):
        if args.print_contents:
            print('# %s' % nut.summary)
            print(squirrel.get_content(nut).dump())
        else:
            print(nut.summary)
