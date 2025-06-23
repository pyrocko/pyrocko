# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Implementation of :app:`squirrel catalog`.
'''

import os
import logging
import time

from pyrocko import util
from pyrocko.model.event import EventFilter
from pyrocko.squirrel.error import ToolError

from ..common import ldq

km = 1000.

logger = logging.getLogger('psq.cli.catalog')

headline = 'Query and list catalog events.'


def make_subparser(subparsers):
    return subparsers.add_parser(
        'catalog',
        help=headline,
        description=headline + '''

Update earthquake catalog information and output summary information about
matching events to the terminal. It reads event information from catalog files
(see ``--add``) or from configured data sources (see ``--dataset``). If the
data sources include online catalogs, the local information is updated for the
given time span. If no time span is specified, the latest events available are
queried and shown. The available information can be filtered by time, magnitude
and event depth.

Examples: ```squirrel catalog -d :events-gcmt-m6``` -- show latest events in
the Global-CMT catalog. ```squirrel catalog -d :``` -- shortcut to see list of
preconfigured datasets. ```squirrel catalog --add events.yaml --tmin 2024
--tmax 2024-02``` -- show all events defined in the file ```events.yaml```
which occurred in January 2024.
''')


def setup(parser):

    parser.add_squirrel_selection_arguments()
    parser.add_squirrel_query_arguments(without=['time', 'codes', 'kind'])
    EventFilter.setup_argparse(parser)

    style_choices = ['table', 'summary', 'yaml']

    parser.add_argument(
        '--style',
        dest='style',
        choices=style_choices,
        default='table',
        help='Set style of presentation. Choices: %s' % ldq(style_choices))

    parser.add_argument(
        '--out',
        dest='output_path',
        metavar='PATH',
        help='Add matching events to ELK database at PATH. A new database '
             'is created if needed.')


def run(parser, args):
    sq = args.make_squirrel()
    event_filter = EventFilter.from_argparse(args)

    if 'tmin' in args.squirrel_query or 'tmax' in args.squirrel_query:
        tmin = args.squirrel_query.get('tmin')
        tmax = args.squirrel_query.get('tmax')

        if tmin is None or tmax is None:
            raise ToolError(
                'In catalog search, --tmin and --tmax must be provided '
                'together.')

        sq.update(tmin=tmin, tmax=tmax, inventory='event')
        events = sq.get_events(tmin=tmin, tmax=tmax, filter=event_filter)

    else:
        napprox = 10
        logger.info(
            'No time range specified, trying to get at least %i latest '
            'events.' % napprox)

        tmax = util.hour_start(time.time())
        tduration = 3600.*24.

        while True:
            tmin = tmax - tduration
            sq.update(tmin=tmin, tmax=tmax, inventory='event')
            events = sq.get_events(
                tmin=tmin,
                tmax=tmax,
                filter=event_filter)

            if len(events) > napprox or tmin < util.str_to_time(
                    '1900-01-01 00:00:00'):
                break

            tduration *= 2

    if not events:
        logger.info('No events found in the time range %s - %s.' % (
            util.time_to_str(tmin),
            util.time_to_str(tmax)))

        return

    if args.output_path:
        logger.info('Storing %i event%s for the time range %s - %s.' % (
            len(events),
            util.plural_s(events),
            util.time_to_str(tmin),
            util.time_to_str(tmax)))

        from pyrocko.squirrel import elk
        if not os.path.exists(args.output_path):
            util.ensuredirs(args.output_path)
            initialize = True
        else:
            initialize = False
        conn = elk.connect(args.output_path)
        if initialize:
            elk.init_database(conn, 'main')

        elk.add_events(conn, 'main', events)

    else:
        logger.info('Showing %i event%s for the time range %s - %s.' % (
            len(events),
            util.plural_s(events),
            util.time_to_str(tmin),
            util.time_to_str(tmax)))

        for ev in events:
            if args.style == 'yaml':
                print('# %s' % ev.summary)
                print(ev.dump())
            elif args.style == 'summary':
                print(ev.summary)
            elif args.style == 'table':
                print('%-20s %-30s %3.1f %3.0f %s' % (
                    ev.name,
                    util.time_to_str(ev.time),
                    ev.magnitude,
                    ev.depth / km,
                    ev.region))
            else:
                assert False
