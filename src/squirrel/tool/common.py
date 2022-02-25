# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

import sys
import argparse
import logging

from pyrocko import util, progress
from pyrocko.squirrel import error


logger = logging.getLogger('psq.tool.common')

help_time_format = 'Format: "YYYY-MM-DD HH:MM:SS.FFF", truncation allowed.'


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


class SquirrelArgumentParser(PyrockoArgumentParser):
    '''
    Parser for CLI arguments with a some extras for Squirrel based apps.

    :param command:
        Implementation of the command.
    :type command:
        :py:class:`SquirrelCommand` (or module providing the same interface).

    :param subcommands:
        Implementations of subcommands.
    :type subcommands:
        :py:class:`list` of :py:class:`SquirrelCommand` (or modules providing
        the same interface).

    :param \\*args:
        Handed through to base class's init.

    :param \\*\\*kwargs:
        Handed through to base class's init.
    '''

    def __init__(self, *args, command=None, subcommands=[], **kwargs):

        self._command = command
        self._subcommands = subcommands
        self._have_selection_arguments = False
        self._have_query_arguments = False

        kwargs['add_help'] = False
        PyrockoArgumentParser.__init__(self, *args, **kwargs)
        add_standard_arguments(self)
        self._command = None
        self._subcommands = []
        if command:
            self.set_command(command)

        if subcommands:
            self.set_subcommands(subcommands)

    def set_command(self, command):
        command.setup(self)
        self.set_defaults(target=command.run)

    def set_subcommands(self, subcommands):
        subparsers = self.add_subparsers(
            metavar='SUBCOMMAND',
            title='Subcommands')

        for mod in subcommands:
            subparser = mod.make_subparser(subparsers)
            if subparser is None:
                raise Exception(
                    'make_subparser(subparsers) must return the created '
                    'parser.')

            mod.setup(subparser)
            subparser.set_defaults(target=mod.run, subparser=subparser)

    def parse_args(self, args=None, namespace=None):
        '''
        Parse arguments given on command line.

        Extends the functionality of
        :py:meth:`argparse.ArgumentParser.parse_args` to process and handle the
        standard options ``--loglevel``, ``--progress`` and ``--help``.
        '''

        args = PyrockoArgumentParser.parse_args(
            self, args=args, namespace=namespace)

        eff_parser = args.__dict__.get('subparser', self)

        process_standard_arguments(self, args)

        if eff_parser._have_selection_arguments:
            def make_squirrel():
                return squirrel_from_selection_arguments(args)

            args.make_squirrel = make_squirrel

        if eff_parser._have_query_arguments:
            try:
                args.squirrel_query = squirrel_query_from_arguments(args)
            except (error.SquirrelError, error.ToolError) as e:
                logger.fatal(str(e))
                sys.exit(1)

        return args

    def dispatch(self, args):
        '''
        Dispatch execution to selected command/subcommand.

        :param args:
            Parsed arguments obtained from :py:meth:`parse_args`.

        :returns:
            ``True`` if dispatching was successful, ``False`` othewise.

        If an exception of type
        :py:exc:`~pyrocko.squirrel.error.SquirrelError` or
        :py:exc:`~pyrocko.squirrel.error.ToolError` is caught, the error is
        logged and the program is terminated with exit code 1.
        '''
        eff_parser = args.__dict__.get('subparser', self)
        target = args.__dict__.get('target', None)

        if target:
            try:
                target(eff_parser, args)
                return True

            except (error.SquirrelError, error.ToolError) as e:
                logger.fatal(str(e))
                sys.exit(1)

        return False

    def run(self, args=None):
        '''
        Parse arguments and dispatch to selected command/subcommand.

        This simply calls :py:meth:`parse_args` and then :py:meth:`dispatch`
        with the obtained ``args``. A usage message is printed if no command is
        selected.
        '''
        args = self.parse_args(args)
        if not self.dispatch(args):
            self.print_help()

    def add_squirrel_selection_arguments(self):
        '''
        Set up command line options commonly used to configure a
        :py:class:`~pyrocko.squirrel.base.Squirrel` instance.

        This will  optional arguments ``--add``, ``--include``, ``--exclude``,
        ``--optimistic``, ``--format``, ``--kind``, ``--persistent``,
        ``--update``, and ``--kind`` to a given argument parser.

        Call ``args.make_squirrel()`` on the arguments returned from
        :py:meth:`parse_args` to finally instantiate and configure the
        :py:class:`~pyrocko.squirrel.base.Squirrel` instance.
        '''
        add_squirrel_selection_arguments(self)
        self._have_selection_arguments = True

    def add_squirrel_query_arguments(self, without=[]):
        '''
        Set up command line options commonly used in squirrel queries.

        This will add options ``--codes``, ``--tmin``, ``--tmax``, and
        ``--time``.

        Once finished with parsing, the query arguments are available as
        ``args.squirrel_query`` on the arguments returned from
        :py:meth:`prase_args`.

        :param without:
            Suppress adding given options.
        :type without:
            :py:class:`list` of :py:class:`str`, choices: ``'tmin'``,
            ``'tmax'``, ``'codes'``, and ``'time'``.
        '''

        add_squirrel_query_arguments(self, without=without)
        self._have_query_arguments = True


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


def add_standard_arguments(parser):
    group = parser.add_argument_group('General options')
    group.add_argument(
        '--help', '-h',
        action='help',
        help='Show this help message and exit.')

    group.add_argument(
        '--loglevel',
        choices=['critical', 'error', 'warning', 'info', 'debug'],
        default='info',
        metavar='LEVEL',
        help='Set logger level. Choices: %(choices)s. Default: %(default)s.')

    group.add_argument(
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


def add_squirrel_selection_arguments(parser):
    '''
    Set up command line options commonly used to configure a
    :py:class:`~pyrocko.squirrel.base.Squirrel` instance.

    This will  optional arguments ``--add``, ``--include``, ``--exclude``,
    ``--optimistic``, ``--format``, ``--kind``, ``--persistent``, ``--update``,
    and ``--kind`` to a given argument parser.

    Once finished with parsing, call
    :py:func:`squirrel_from_selection_arguments` to finally instantiate and
    configure the :py:class:`~pyrocko.squirrel.base.Squirrel` instance.

    :param parser:
        The argument parser to be configured.
    :type parser:
        argparse.ArgumentParser
    '''
    from pyrocko import squirrel as sq

    group = parser.add_argument_group('Data collection options')

    group.add_argument(
        '--add', '-a',
        dest='paths',
        metavar='PATH',
        nargs='+',
        help='Add files and directories with waveforms, metadata and events. '
             'Content is indexed and added to the temporary (default) or '
             'persistent (see --persistent) data selection.')

    group.add_argument(
        '--include',
        dest='include',
        metavar='REGEX',
        help='Only include files whose paths match the regular expression '
             'REGEX. Examples: --include=\'\\.MSEED$\' would only match files '
             'ending with ".MSEED". --include=\'\\.BH[EN]\\.\' would match '
             'paths containing ".BHE." or ".BHN.". --include=\'/2011/\' would '
             'match paths with a subdirectory "2011" in their path hierarchy.')

    group.add_argument(
        '--exclude',
        dest='exclude',
        metavar='REGEX',
        help='Only include files whose paths do not match the regular '
             'expression REGEX. Examples: --exclude=\'/\\.DS_Store/\' would '
             'exclude anything inside any ".DS_Store" subdirectory.')

    group.add_argument(
        '--optimistic', '-o',
        action='store_false',
        dest='check',
        default=True,
        help='Disable checking file modification times for faster startup.')

    group.add_argument(
        '--format', '-f',
        dest='format',
        metavar='FORMAT',
        default='detect',
        choices=sq.supported_formats(),
        help='Assume input files are of given FORMAT. Choices: %(choices)s. '
             'Default: %(default)s.')

    group.add_argument(
        '--add-only',
        type=csvtype(sq.supported_content_kinds()),
        dest='kinds_add',
        metavar='KINDS',
        help='Restrict meta-data scanning to given content kinds. '
             'KINDS is a comma-separated list of content kinds, choices: %s. '
             'By default, all content kinds are indexed.'
             % ', '.join(sq.supported_content_kinds()))

    group.add_argument(
        '--persistent', '-p',
        dest='persistent',
        metavar='NAME',
        help='Create/use persistent selection with given NAME. Persistent '
             'selections can be used to speed up startup of Squirrel-based '
             'applications.')

    group.add_argument(
        '--update', '-u',
        dest='update',
        action='store_true',
        default=False,
        help='Allow adding paths and datasets to existing persistent '
             'selection.')

    group.add_argument(
        '--dataset', '-d',
        dest='datasets',
        default=[],
        action='append',
        metavar='FILE',
        help='Add files, directories and remote sources from dataset '
             'description file. This option can be repeated to add multiple '
             'datasets. Run `squirrel template` to obtain examples of dataset '
             'description files.')


def squirrel_from_selection_arguments(args):
    '''
    Create a :py:class:`~pyrocko.squirrel.base.Squirrel` instance from command
    line arguments.

    Use :py:func:`add_squirrel_selection_arguments` to configure the parser
    with the necessary options.

    :param args:
        Parsed command line arguments, as returned by
        :py:meth:`argparse.ArgumentParser.parse_args`.

    :returns:
        :py:class:`pyrocko.squirrel.base.Squirrel` instance with paths,
        datasets and remote sources added.

    '''
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

    if args.paths:
        squirrel.add(
            args.paths,
            check=args.check,
            format=args.format,
            kinds=args.kinds_add or None,
            include=args.include,
            exclude=args.exclude)

    for ds in datasets:
        squirrel.add_dataset(ds, check=args.check)

    return squirrel


def add_squirrel_query_arguments(parser, without=[]):
    '''
    Set up command line options commonly used in squirrel queries.

    This will add options ``--codes``, ``--tmin``, ``--tmax``, and ``--time``.

    Once finished with parsing, call
    :py:func:`squirrel_query_from_arguments` to get the parsed values.

    :param parser:
        The argument parser to be configured.
    :type parser:
        argparse.ArgumentParser

    :param without:
        Suppress adding given options.
    :type without:
        :py:class:`list` of :py:class:`str`
    '''

    from pyrocko import squirrel as sq

    group = parser.add_argument_group('Data query options')

    if 'kinds' not in without:
        group.add_argument(
            '--kinds',
            type=csvtype(sq.supported_content_kinds()),
            dest='kinds',
            metavar='KINDS',
            help='Content kinds to query. KINDS is a comma-separated list of '
                 'content kinds, choices: %s. By default, all content kinds '
                 'are queried.' % ', '.join(sq.supported_content_kinds()))

    if 'codes' not in without:
        group.add_argument(
            '--codes',
            dest='codes',
            metavar='CODES',
            help='Code pattern to query (STA, NET.STA, NET.STA.LOC, '
                 'NET.STA.LOC.CHA, or NET.STA.LOC.CHA.EXTRA).')

    if 'tmin' not in without:
        group.add_argument(
            '--tmin',
            dest='tmin',
            metavar='TIME',
            help='Begin of time interval to query. %s' % help_time_format)

    if 'tmax' not in without:
        group.add_argument(
            '--tmax',
            dest='tmax',
            metavar='TIME',
            help='End of time interval to query. %s' % help_time_format)

    if 'time' not in without:
        group.add_argument(
            '--time',
            dest='time',
            metavar='TIME',
            help='Time instant to query. %s' % help_time_format)


def squirrel_query_from_arguments(args):
    '''
    Get common arguments to be used in squirrel queries from command line.

    Use :py:func:`add_squirrel_query_arguments` to configure the parser with
    the necessary options.

    :param args:
        Parsed command line arguments, as returned by
        :py:meth:`argparse.ArgumentParser.parse_args`.

    :returns:
        :py:class:`dict` with any parsed option values.
    '''

    from pyrocko import squirrel as sq

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
        d['codes'] = sq.to_codes_guess(args.codes)

    if ('tmin' in d and 'time' in d) or ('tmax' in d and 'time' in d):
        raise error.SquirrelError(
            'Options --tmin/--tmax and --time are mutually exclusive.')
    return d


class SquirrelCommand(object):
    '''
    Base class for Squirrel-based CLI programs and subcommands.
    '''

    def fail(self, message):
        '''
        Raises :py:exc:`~pyrocko.squirrel.error.ToolError`.

        :py:func:`~pyrocko.squirrel.tool.from_command` catches
        :py:exc:`~pyrocko.squirrel.error.ToolError`, logs the error message and
        terminates with an error exit state.
        '''
        raise error.ToolError(message)

    def make_subparser(self, subparsers):
        '''
        To be implemented in subcommand. Create subcommand parser.

        Must return a newly created parser obtained with
        :py:meth:`add_parser`, e.g.::

            def make_subparser(self, subparsers):
                return subparsers.add_parser(
                    'plot', help='Draw a nice plot.')

        '''
        return subparsers.add_parser(
            self.__class__.__name__, help='Undocumented.')

    def setup(self, parser):
        '''
        To be implemented in subcommand. Configure parser.

        :param parser:
            The argument parser to be configured.
        :type parser:
            argparse.ArgumentParser

        Example::

            def setup(self, parser):
                parser.add_squirrel_selection_arguments()
                parser.add_squirrel_query_arguments()
                parser.add_argument(
                    '--fmin',
                    dest='fmin',
                    metavar='FLOAT',
                    type=float,
                    help='Corner of highpass [Hz].')
        '''
        pass

    def run(self, parser, args):
        '''
        To be implemented in subcommand. Main routine of the command.

        :param parser:
            The argument parser to be configured.
        :type parser:
            argparse.ArgumentParser

        :param args:
            Parsed command line arguments, as returned by
            :py:meth:`argparse.ArgumentParser.parse_args`.

        Example::

            def run(self, parser, args):
                print('User has selected fmin = %g Hz' % args.fmin)

                # args.make_squirrel() is available if
                # parser.add_squirrel_selection_arguments() was called during
                # setup().

                sq = args.make_squirrel()

                # args.squirrel_query is available if
                # praser.add_squirrel_query_arguments() was called during
                # setup().

                stations = sq.get_stations(**args.squirrel_query)
        '''
        pass


__all__ = [
    'SquirrelArgumentParser',
    'SquirrelCommand',
    'add_squirrel_selection_arguments',
    'squirrel_from_selection_arguments',
    'add_squirrel_query_arguments',
    'squirrel_query_from_arguments',
]
