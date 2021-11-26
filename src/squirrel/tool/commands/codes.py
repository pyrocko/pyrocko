# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

from .. import common


def setup(subparsers):
    p = common.add_parser(
        subparsers, 'codes',
        help='Get summary of available data codes.')

    common.add_selection_arguments(p)

    return p


def call(parser, args):
    squirrel = common.squirrel_from_selection_arguments(args)
    for (kind, codes, deltat), count in squirrel.iter_counts():
        print(kind, '.'.join(codes), deltat, count)
