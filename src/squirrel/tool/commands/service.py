# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Implementation of :app:`squirrel service`.
'''

import logging
from pyrocko.squirrel.service import server


logger = logging.getLogger('psq.cli.service')

headline = 'Fire up web UI.'


def make_subparser(subparsers):
    return subparsers.add_parser(
        'service',
        help=headline,
        description=headline + '''


''')


def setup(parser):
    parser.add_squirrel_selection_arguments()
    parser.add_squirrel_query_arguments(without='kinds')
    server.add_cli_arguments(parser)


def run(parser, args):
    squirrel = args.make_squirrel()
    gates = {
        'default': server.Gate.from_query_arguments(**args.squirrel_query),
    }

    server.run(
        squirrel=squirrel,
        gates=gates,
        host=args.host,
        port=args.port,
        open=args.open,
        debug=args.debug,
        page_path=args.page_path)
