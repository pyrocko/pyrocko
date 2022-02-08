# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

from pyrocko import squirrel as sq

from .. import common


def setup_subcommand(subparsers):
    return common.add_parser(
        subparsers, 'init',
        help='Create local environment.')


def setup(parser):
    pass


def call(parser, args):
    sq.init_environment()
