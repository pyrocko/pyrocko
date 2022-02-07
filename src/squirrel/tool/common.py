# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

import argparse
import logging

from pyrocko import util, progress
from pyrocko.squirrel import error


logger = logging.getLogger('psq.tool.common')


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
    add_standard_arguments(p)
    return p


def add_standard_arguments(p):
    p.add_argument(
        '--help', '-h',
        action='help',
        help='Show this help message and exit.')

    p.add_argument(
        '--loglevel',
        choices=['critical', 'error', 'warning', 'info', 'debug'],
        default='info',
        metavar='LEVEL',
        help='Set logger level. Choices: %(choices)s. Default: %(default)s.')

    p.add_argument(
        '--progress',
        choices=['terminal', 'log', 'off'],
        default='terminal',
        metavar='DEST',
        help='Set how progress status is reported. Choices: %(choices)s. '
             'Default: %(default)s.')


def process_standard_arguments(parser, args):
    loglevel = args.__dict__.pop('loglevel')
    util.setup_logging(parser.prog, loglevel)

    pmode = args.__dict__.pop('progress')
    progress.set_default_viewer(pmode)


def add_selection_arguments(p):
    from pyrocko import squirrel as sq

    p.add_argument(
        'paths',
        nargs='*',
        help='Files and directories with waveforms, metadata and events.')

    p.add_argument(
        '--include',
        dest='include',
        metavar='REGEX',
        help='Only include files whose paths match REGEX. Examples: '
             '--include=\'\\.MSEED$\' would only match files ending with '
             '".MSEED". --include=\'\\.BH[EN]\\.\' would match paths '
             'containing ".BHE." or ".BHN.". --include=\'/2011/\' would match '
             'paths with a subdirectory "2011" in their path hierarchy.')

    p.add_argument(
        '--exclude',
        dest='exclude',
        metavar='REGEX',
        help='If given, files are only included if their paths do not match '
             'the regular expression pattern REGEX.')

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
    from pyrocko.squirrel import base, dataset

    datasets = [
        dataset.read_dataset(dataset_path) for dataset_path in args.datasets]

    persistents = [ds.persistent or '' for ds in datasets if ds.persistent]
    if args.persistent:
        persistent = args.persistent
    elif persistents:
        persistent = persistents[0]
        if not all(p == persistents for p in persistents[1:]):
            raise error.SquirrelError(
                'Given datasets specify different `persistent` settings.')

        if persistent:
            logger.info(
                'Persistent selection requested by dataset: %s' % persistent)
        else:
            persistent = None

    else:
        persistent = None

    squirrel = base.Squirrel(persistent=persistent)

    if persistent and not squirrel.is_new():
        if not args.update:
            logger.info(
                'Using existing persistent selection: %s' % persistent)
            if args.paths or datasets:
                logger.info(
                    'Avoiding dataset rescan. Use --update/-u to '
                    'rescan or add items to existing persistent selection.')

            return squirrel

        else:
            logger.info(
                'Updating existing persistent selection: %s' % persistent)

    kinds = args.kinds or None
    if args.paths:
        squirrel.add(
            args.paths,
            check=args.check,
            format=args.format,
            kinds=kinds,
            include=args.include,
            exclude=args.exclude)

    for ds in datasets:
        squirrel.add_dataset(ds, check=args.check)

    return squirrel


def add_query_arguments(p, without=[]):
    if 'codes' not in without:
        p.add_argument(
            '--codes',
            dest='codes',
            metavar='CODES',
            help='Code pattern to query (STA, NET.STA, NET.STA.LOC, '
                 'NET.STA.LOC.CHA, NET.STA.LOC.CHA.EXTRA, '
                 'AGENCY.NET.STA.LOC.CHA.EXTRA).')

    if 'tmin' not in without:
        p.add_argument(
            '--tmin',
            dest='tmin',
            metavar='TIME',
            help='Begin of time interval to query.')

    if 'tmax' not in without:
        p.add_argument(
            '--tmax',
            dest='tmax',
            metavar='TIME',
            help='End of time interval to query.')

    if 'time' not in without:
        p.add_argument(
            '--time',
            dest='time',
            metavar='TIME',
            help='Time instant to query.')


def squirrel_query_from_arguments(args):
    d = {}

    if 'kinds' in args and args.kinds:
        d['kind'] = args.kinds
    if 'tmin' in args and args.tmin:
        d['tmin'] = util.str_to_time_fillup(args.tmin)
    if 'tmax' in args and args.tmax:
        d['tmax'] = util.str_to_time_fillup(args.tmax)
    if 'time' in args and args.time:
        d['tmin'] = d['tmax'] = util.str_to_time_fillup(args.time)
    if 'codes' in args and args.codes:
        d['codes'] = tuple(args.codes.split('.'))

    if ('tmin' in d and 'time' in d) or ('tmax' in d and 'time' in d):
        raise error.SquirrelError(
            'Options --tmin/--tmax and --time are mutually exclusive.')
    return d


class SquirrelCommand(object):

    def add_parser(self, subparsers, *args, **kwargs):
        return add_parser(subparsers, *args, **kwargs)

    def add_selection_arguments(self, p):
        return add_selection_arguments(p)

    def add_query_arguments(self, p):
        return add_query_arguments(p)

    def squirrel_query_from_arguments(self, args):
        return squirrel_query_from_arguments(args)

    def squirrel_from_selection_arguments(self, args):
        return squirrel_from_selection_arguments(args)

    def fail(self, message):
        raise error.ToolError(message)


__all__ = ['SquirrelCommand']
