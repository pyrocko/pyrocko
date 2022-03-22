# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function


headline = 'Experimental Squirrel-powered Snuffler.'


def make_subparser(subparsers):
    return subparsers.add_parser(
        'snuffler',
        help=headline,
        description=headline)


def setup(parser):
    parser.add_squirrel_selection_arguments()


def run(parser, args):
    squirrel = args.make_squirrel()
    squirrel.pile.snuffle()
