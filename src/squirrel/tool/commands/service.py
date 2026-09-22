# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Implementation of :app:`squirrel service`.
'''

import logging
from pyrocko.squirrel.service import server
from ..common import add_mantra_arguments, get_mantras_from_arguments


logger = logging.getLogger('psq.cli.service')

headline = 'Fire up web UI.'


def make_subparser(subparsers):
    return subparsers.add_parser(
        'service',
        help=headline,
        description=headline + '''

Data is served through "gates". The ``default`` gate provides the data from
the selected inputs as it is. Use ``--mantra`` to add gates with processed
data: each of the selected mantras gets a gate named like the mantra. Mantra
names must be unique and may only contain lowercase letters, digits and
underscores. Data from all gates is shown together, so the codes of the
provided data should be distinct between gates.
''')


def setup(parser):
    parser.add_squirrel_selection_arguments()
    parser.add_squirrel_query_arguments(without='kinds')
    add_mantra_arguments(parser)
    server.add_cli_arguments(parser)


def run(parser, args):
    squirrel = args.make_squirrel()
    gates = {
        server.GATE_NAME_DEFAULT: server.Gate.from_query_arguments(
            **args.squirrel_query),
    }

    gates = server.gates_from_mantras(get_mantras_from_arguments(args), gates)

    server.run(
        squirrel=squirrel,
        gates=gates,
        host=args.host,
        port=args.port,
        open=args.open,
        debug=args.debug,
        cookie_secret_path=args.cookie_secret_path,
        page_path=args.page_path)
