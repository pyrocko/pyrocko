# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

import sys
import logging

from . import common
from .commands import command_modules
from pyrocko import squirrel as sq


logger = logging.getLogger('psq.cli')


g_program_name = 'squirrel'


def main(args=None):
    from_command(
        args=args,
        program_name=g_program_name,
        subcommands=command_modules,
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
`squirrel SUBCOMMAND --help` to get further help.''')


def from_command(
        command=None,
        args=None,
        program_name=None,
        description='''
Pyrocko Squirrel based script.

Run with --help to get further help.''',
        subcommands=[],
        setup=None):

    if program_name is None:
        program_name = sys.argv[0]

    if args is None:
        args = sys.argv[1:]

    parser = common.PyrockoArgumentParser(
        prog=program_name,
        add_help=False,
        description=description)

    common.add_standard_arguments(parser)

    if subcommands:
        subparsers = parser.add_subparsers(
            metavar='SUBCOMMAND',
            title='Subcommands')

        for mod in subcommands:
            subparser = mod.setup_subcommand(subparsers)
            assert subparser is not None
            mod.setup(subparser)
            subparser.set_defaults(target=mod.call, subparser=subparser)

    elif command:
        command.setup(parser)

    else:
        common.add_selection_arguments(parser)

    args = parser.parse_args(args)
    subparser = args.__dict__.pop('subparser', None)

    common.process_standard_arguments(parser, args)

    target = args.__dict__.pop('target', None)

    if target:
        try:
            target(parser, args)
        except (sq.SquirrelError, sq.ToolError) as e:
            logger.fatal(str(e))
            sys.exit(1)

    elif command:
        command.call(parser, args)

    elif not subcommands:
        return common.squirrel_from_selection_arguments(args)

    else:
        parser.print_help()
        sys.exit(0)


__all__ = [
    'main',
    'from_command',
]
