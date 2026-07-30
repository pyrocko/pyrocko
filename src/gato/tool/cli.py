# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Gato command line tool main program.
'''

import logging
from pyrocko.squirrel import run
from .commands import command_modules
from ..error import GatoError


logger = logging.getLogger('gato.cli')

g_program_name = 'gato'


def main(args=None):
    run(
        args=args,
        prog=g_program_name,
        subcommands=command_modules,
        nice_fatal_errors=[GatoError],
        description='''
Pyrocko Gato - generalized array toolkit.

This is ```gato```, a command-line front-end to the functionality of the
generalized array toolkit.

This tool's functionality is available through several subcommands. Run
```gato SUBCOMMAND --help``` to get further help.''')


__all__ = [
    'main',
]
