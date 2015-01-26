# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

import sys

from pyrocko import util

from . import common
from .commands import command_modules
from pyrocko import squirrel as sq


g_program_name = 'squirrel'


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = common.PyrockoArgumentParser(
        prog=g_program_name,
        add_help=False,
        description='''
Pyrocko Squirrel - Prompt seismological data access with a fluffy tail.

This is `squirrel`, a command-line front-end to the Squirrel data access
infrastructure. The Squirrel infrastructure offers meta-data caching, blazingly
fast data lookup for large datasets and transparent online data download to
applications building on it.

In most cases, the Squirrel does its business discretely under the hood and
does not require human interaction or awareness. However, using this tool, some
aspects can be configured for the benefit of greater performance or
convenience, including (1) using a separate (isolated, local) environment for a
specific project, (2) using named selections to speed up access to very large
datasets, (3), pre-scanning/indexing of file collections. It can also be used
to inspect various aspects of a data collection.

This tool's functionality is available through several subcommands. Run
`squirrel [subcommand] --help` to get further help.''')

    parser.add_argument(
        '--help', '-h',
        action='store_true',
        help='Show this help message and exit.')

    parser.add_argument(
        '--loglevel', '-l',
        choices=['critical', 'error', 'warning', 'info', 'debug'],
        default='info',
        help='Set logger level. Default: %(default)s')

    subparsers = parser.add_subparsers(
        title='Subcommands')

    for mod in command_modules:
        subparser = mod.setup(subparsers)
        subparser.set_defaults(target=mod.call, subparser=subparser)

    args = parser.parse_args(args)
    subparser = args.__dict__.pop('subparser', None)
    if args.help:
        (subparser or parser).print_help()
        sys.exit(0)

    loglevel = args.__dict__.pop('loglevel')
    util.setup_logging(g_program_name, loglevel)

    target = args.__dict__.pop('target', None)
    if target:
        try:
            target(parser, args)
        except sq.SquirrelError as e:
            sys.exit(str(e))


__all__ = [
    'main',
]
