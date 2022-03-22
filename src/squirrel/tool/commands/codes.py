# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

from pyrocko.squirrel.model import to_kind


headline = 'Get summary of available data codes.'


def make_subparser(subparsers):
    return subparsers.add_parser(
        'codes',
        help=headline,
        description=headline)


def setup(parser):
    parser.add_squirrel_selection_arguments()


def run(parser, args):
    squirrel = args.make_squirrel()
    for kind_id, codes, deltat, _, count in sorted(
            squirrel._iter_codes_info()):
        print(to_kind(kind_id), codes, deltat, count)
