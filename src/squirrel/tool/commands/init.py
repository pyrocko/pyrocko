# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

from pyrocko import squirrel as sq


headline = 'Create local environment.'


def make_subparser(subparsers):
    return subparsers.add_parser(
        'init',
        help=headline,
        description=headline)


def setup(parser):
    pass


def run(parser, args):
    sq.init_environment()
