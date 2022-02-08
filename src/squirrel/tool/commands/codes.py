# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

from .. import common


def setup_subcommand(subparsers):
    return common.add_parser(
        subparsers, 'codes',
        help='Get summary of available data codes.')


def setup(parser):
    common.add_selection_arguments(parser)


def call(parser, args):
    squirrel = common.squirrel_from_selection_arguments(args)
    for (kind, codes, deltat), count in squirrel.iter_counts():
        print(kind, '.'.join(codes), deltat, count)
