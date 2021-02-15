# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

import argparse
import logging

from pyrocko import util
from pyrocko.squirrel import error


logger = logging.getLogger('pyrocko.squirrel.tool.common')


class PyrockoHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def __init__(self, *args, **kwargs):
        kwargs['width'] = 79
        argparse.RawDescriptionHelpFormatter.__init__(self, *args, **kwargs)


class PyrockoArgumentParser(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):

        kwargs['formatter_class'] = PyrockoHelpFormatter

        argparse.ArgumentParser.__init__(self, *args, **kwargs)

        if hasattr(self, '_action_groups'):
            for group in self._action_groups:
                if group.title == 'positional arguments':
                    group.title = 'Positional arguments'

                elif group.title == 'optional arguments':
                    group.title = 'Optional arguments'


def csvtype(choices):
    def splitarg(arg):
        values = arg.split(',')
        for value in values:
            if value not in choices:
                raise argparse.ArgumentTypeError(
                    'Invalid choice: {!r} (choose from {})'
                    .format(value, ', '.join(map(repr, choices))))
        return values
    return splitarg


def add_parser(subparsers, *args, **kwargs):
    kwargs['add_help'] = False
    p = subparsers.add_parser(*args, **kwargs)
    p.add_argument(
        '--help', '-h',
        action='help',
        help='Show this help message and exit.')
    return p


def add_selection_arguments(p):
    from pyrocko import squirrel as sq

    p.add_argument(
        'paths',
        nargs='*',
        help='Files and directories with waveforms, metadata and events.')

    p.add_argument(
        '--optimistic', '-o',
        action='store_false',
        dest='check',
        default=True,
        help='Disable checking file modification times.')

    p.add_argument(
        '--format', '-f',
        dest='format',
        metavar='FORMAT',
        default='detect',
        choices=sq.supported_formats(),
        help='Assume input files are of given FORMAT. Choices: %(choices)s. '
             'Default: %(default)s.')

    p.add_argument(
        '--kind', '-k',
        type=csvtype(sq.supported_content_kinds()),
        dest='kinds',
        help='Restrict meta-data scanning to given content kinds. '
             'KINDS is a comma-separated list of content kinds, choices: %s. '
             'By default, all content kinds are indexed.'
             % ', '.join(sq.supported_content_kinds()))

    p.add_argument(
        '--persistent', '-p',
        dest='persistent',
        metavar='NAME',
        help='Create/use persistent selection with given NAME. Persistent '
             'selections can be used to speed up startup of Squirrel-based '
             'applications.')

    p.add_argument(
        '--update', '-u',
        dest='update',
        action='store_true',
        default=False,
        help='Allow adding paths and datasets to existing persistent '
             'selection.')

    p.add_argument(
        '--dataset', '-d',
        dest='datasets',
        default=[],
        action='append',
        metavar='FILE',
        help='Add directories/files/sources from dataset description file. '
             'This option can be repeated to add multiple datasets.')


def squirrel_from_selection_arguments(args):
    from pyrocko.squirrel import base, environment, database
    persistent = args.persistent

    env = environment.get_environment()
    if persistent:
        db = database.get_database(env.database_path)
        persistent_exists = persistent in db.get_persistent_names()

    squirrel = base.Squirrel(env, persistent=persistent)

    if persistent and persistent_exists:
        if not args.update:
            logger.info(
                'Using existing persistent selection: %s' % persistent)
            if args.paths or args.datasets:
                logger.info(
                    'Ignoring path and dataset arguments. Use --update to add '
                    'to existing persistent selection.')

            return squirrel

        else:
            logger.info(
                'Updating existing persistent selection: %s' % persistent)

    kinds = args.kinds or None
    if args.paths:
        squirrel.add(
            args.paths, check=args.check, format=args.format, kinds=kinds)

    for dataset_path in args.datasets:
        squirrel.add_dataset(dataset_path, check=args.check)

    return squirrel


def add_query_arguments(p):
    p.add_argument(
        '--codes',
        dest='codes',
        metavar='CODES',
        help='Code pattern to query (STA, NET.STA, NET.STA.LOC, '
             'NET.STA.LOC.CHA, NET.STA.LOC.CHA.EXTRA, '
             'AGENCY.NET.STA.LOC.CHA.EXTRA).')

    p.add_argument(
        '--tmin',
        dest='tmin',
        metavar='TIME',
        help='Begin of time interval to query.')

    p.add_argument(
        '--tmax',
        dest='tmax',
        metavar='TIME',
        help='End of time interval to query.')

    p.add_argument(
        '--time',
        dest='time',
        metavar='TIME',
        help='Time instant to query.')


def squirrel_query_from_arguments(args):
    d = {}
    if (args.tmin and args.time) or (args.tmax and args.time):
        raise error.SquirrelError(
            'Options --tmin/--tmax and --time are mutually exclusive.')

    if args.kinds:
        d['kind'] = args.kinds
    if args.tmin:
        d['tmin'] = util.str_to_time(args.tmin)
    if args.tmax:
        d['tmax'] = util.str_to_time(args.tmax)
    if args.time:
        d['tmin'] = d['tmax'] = util.str_to_time(args.time)
    if args.codes:
        d['codes'] = tuple(args.codes.split('.'))

    return d
