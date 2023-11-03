# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Implementation of :app:`gato array`.
'''

from pyrocko import progress
from pyrocko.util import glob_filter
from pyrocko.squirrel import SquirrelCommand, Squirrel
from pyrocko.gato.array import get_named_arrays, get_named_arrays_dataset
from pyrocko.squirrel.tool.common import ldq


def add_argument_array_names(parser, nargs):
    parser.add_argument(
        dest='array_names',
        nargs=nargs,
        metavar='NAMES',
        help='List only arrays with names matching given (glob-style) '
             'patterns.')


def get_matching_builtin_array_names(name_patterns):
    arrays = get_named_arrays()
    return sorted(glob_filter(name_patterns, arrays.keys()))


def get_matching_builtin_arrays(name_patterns):
    arrays = get_named_arrays()
    return [
        (name, arrays[name])
        for name in get_matching_builtin_array_names(name_patterns)]


class List(SquirrelCommand):

    def make_subparser(self, subparsers):
        return subparsers.add_parser(
            'list',
            help='List array setups.',
            description='List array setups.')

    def setup(self, parser):
        add_argument_array_names(parser, '*')

    def run(self, parser, args):
        for (name, array) in get_matching_builtin_arrays(args.array_names):
            print('%-25s %s' % (name, array.comment or ''))


class Show(SquirrelCommand):

    def make_subparser(self, subparsers):
        return subparsers.add_parser(
            'show',
            help='Print array setup.',
            description='Print array setup.')

    def setup(self, parser):
        add_argument_array_names(parser, '+')

    def run(self, parser, args):
        for (name, array) in get_matching_builtin_arrays(args.array_names):
            print('# %s' % name)
            print(array)


class Info(SquirrelCommand):

    def make_subparser(self, subparsers):
        return subparsers.add_parser(
            'info',
            help='Print information about array.',
            description='Print information about array.')

    def setup(self, parser):
        add_argument_array_names(parser, '+')

        style_choices = ['summary', 'yaml']

        parser.add_argument(
            '--style',
            dest='style',
            choices=style_choices,
            default='summary',
            help='Set style of presentation. Choices: %s' % ldq(style_choices))

        parser.add_squirrel_query_arguments(without=['codes', 'kinds'])

    def run(self, parser, args):

        arrays = dict(get_matching_builtin_arrays(args.array_names))
        names = sorted(arrays.keys())

        sq = Squirrel()
        sq.add_dataset(get_named_arrays_dataset(names))

        with progress.view():
            sq.update()

        for name, array in arrays.items():
            info = array.get_info(sq, **args.squirrel_query)
            if args.style == 'summary':
                print(info.summary)
            elif args.style == 'yaml':
                print('# ' + info.summary)
                print(info)


class Sensors(SquirrelCommand):

    def make_subparser(self, subparsers):
        return subparsers.add_parser(
            'sensors',
            help='Print information sensors of an array.',
            description='Print information sensors of an array.')

    def setup(self, parser):
        add_argument_array_names(parser, '+')

        style_choices = ['summary', 'yaml']

        parser.add_argument(
            '--style',
            dest='style',
            choices=style_choices,
            default='summary',
            help='Set style of presentation. Choices: %s' % ldq(style_choices))

        parser.add_squirrel_query_arguments(without=['codes', 'kinds'])

    def run(self, parser, args):

        arrays = dict(get_matching_builtin_arrays(args.array_names))
        names = sorted(arrays.keys())

        sq = Squirrel()
        sq.add_dataset(get_named_arrays_dataset(names))

        with progress.view():
            sq.update()

        for name, array in arrays.items():
            info = array.get_info(sq, **args.squirrel_query)
            for sensor in info.sensors:
                if args.style == 'summary':
                    print(sensor.summary)
                elif args.style == 'yaml':
                    print('# ' + sensor.summary)
                    print(sensor)


headline = 'Manage arrays setups.'


def make_subparser(subparsers):
    return subparsers.add_parser(
        'array',
        help=headline,
        subcommands=[List(), Show(), Info(), Sensors()],
        description=headline + '''

Manage seismic array setups: add, remove, show.
''')


def setup(parser):
    pass


def run(parser, args):
    parser.print_help()
