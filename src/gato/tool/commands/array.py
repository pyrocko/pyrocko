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
    return sorted((
        arrays[name]
        for name in get_matching_builtin_array_names(name_patterns)),
        key=lambda array: (array.type, array.name))


def get_matching_builtin_arrays_dict(name_patterns):
    return dict(
        (array.name, array)
        for array in get_matching_builtin_arrays(name_patterns))


class List(SquirrelCommand):

    def make_subparser(self, subparsers):
        return subparsers.add_parser(
            'list',
            help='List array setups.',
            description='List array setups.')

    def setup(self, parser):
        add_argument_array_names(parser, '*')

        style_choices = ['summary', 'yaml', 'name']

        parser.add_argument(
            '--style',
            dest='style',
            choices=style_choices,
            default='summary',
            help='Set style of presentation. Choices: %s' % ldq(style_choices))

    def run(self, parser, args):
        for array in get_matching_builtin_arrays(args.array_names):
            if args.style == 'name':
                print(array.name)
            elif args.style == 'summary':
                print(array.summary2)
            else:
                print('#', array.summary2)
                print(array)


class Info(SquirrelCommand):

    def make_subparser(self, subparsers):
        return subparsers.add_parser(
            'info',
            help='Print information about array.',
            description='Print information about array.')

    def setup(self, parser):
        add_argument_array_names(parser, '*')

        style_choices = ['summary', 'yaml']

        parser.add_argument(
            '--style',
            dest='style',
            choices=style_choices,
            default='summary',
            help='Set style of presentation. Choices: %s' % ldq(style_choices))

        parser.add_squirrel_query_arguments(without=['codes', 'kinds'])

    def run(self, parser, args):

        arrays = get_matching_builtin_arrays_dict(args.array_names)
        names = sorted(arrays.keys())

        sq = Squirrel()
        sq.add_dataset(get_named_arrays_dataset(names))

        with progress.view():
            sq.update()

        print('#', ' | '.join([
            'name',
            'type',
            'num. sites',
            'num. channels',
            'start date',
            'end date',
            'interstation distances min, 10%, 50%, 90%, max [km]',
            'channel group: num. sites']))

        for name, array in arrays.items():
            info = array.get_info(sq, **args.squirrel_query)
            if args.style == 'summary':
                print(' | '.join((array.summary, info.summary)))
            elif args.style == 'yaml':
                print('#', ' | '.join((array.summary, info.summary)))
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

        arrays = get_matching_builtin_arrays_dict(args.array_names)
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
        subcommands=[List(), Info(), Sensors()],
        description=headline + '''

Manage seismic array setups: add, remove, show.
''')


def setup(parser):
    pass


def run(parser, args):
    parser.print_help()
