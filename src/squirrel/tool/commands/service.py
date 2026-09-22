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

Starts a local web server providing a browser-based UI to interactively explore
the data available through a Squirrel selection: waveforms, carpets (e.g.
spectrograms) and their data coverage, channel and response metadata, and
events are shown on a scrollable timeline, station positions on a map.

Downsampled waveform overviews are computed (up to a given threshold) on the
fly while panning and zooming. Downsampled carpet overviews (cascades) can be
pre-computed using ```squirrel cascade``` for seamless presentation of
multi-year long datasets.

Data is served through "gates". The ``default`` gate provides the data from the
selected inputs as it is. Use ``--mantra`` to add gates with processed data:
each of the selected mantras gets a gate named like the mantra, and its outputs
(e.g. restituted waveforms or spectrograms) appear alongside the raw data in
the timeline. Mantra names must be unique and may only contain lowercase
letters, digits and underscores. Since data from all gates is shown together,
the codes of the provided data should be distinct between gates; this can for
example be achieved by including the mantra name in the operators' output
codes (see ```squirrel mantra show```).

By default, the service only listens on ``localhost`` and is not reachable
from other machines. Use ``--host`` to bind to a different address, and see
``--help-port-forwarding`` for ways to access a service running on a remote
host. Use ``--open`` to open the UI in a web browser automatically.

Examples: ```squirrel service --dataset campaign.dataset.yaml``` -- serve the
data of a preconfigured dataset. ```squirrel service --add data/ --mantra
mantras.yaml:restitution``` -- serve the raw data together with the additional
data produced by the ``restitution`` mantra defined in ```mantras.yaml```.
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
