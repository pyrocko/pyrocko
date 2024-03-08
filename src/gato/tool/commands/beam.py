# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Implementation of :app:`gato beam`.
'''
import os
from pyrocko import progress
from pyrocko.util import glob_filter
from pyrocko.squirrel import SquirrelCommand, Squirrel
from pyrocko.gato.array import get_named_arrays, get_named_arrays_dataset
from pyrocko.squirrel.tool.common import ldq
from pyrocko.guts import Object, Float, Int, String, Bool, load
guts_prefix = 'gato'


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

def add_argument_configuration(parser, nargs):
    parser.add_argument(
        dest='configuration',
        nargs=nargs,
        metavar='CONFIG',
        help='filename or keyword like "teleseismic", "regional", or "local"')

class ProcOpts(Object):

    # filter options
    highpass = Float.T(optional=True)
    lowpass = Float.T(optional=True)
    transfer_removal = Bool.T(default=False)
    transfer_f1 = Float.T(default=0.001)
    transfer_f2 = Float.T(default=0.01)
    transfer_f3 = Float.T(default=100.)
    transfer_f4 = Float.T(default=200.)
    # window options
    winlen = Float.T(default=60.)
    winstep = Float.T(default=30.)
    # stack option
    stacklen = Float.T(default=86400.)
    # preprocessing options:
    taper = Float.T(default=0.2)
    prewhiten = Bool.T(default=False)
    # to be continued ...

    def __init__(self, **kwargs):
        # call the guts initilizer
        Object.__init__(self, **kwargs)


def get_matching_configuration(name_patterns):

    config_name = name_patterns[0]
    if config_name == 'teleseismic':
        config = 'teleseismic'
    elif config_name == 'regional':
        config = 'regional'
    elif config_name == 'local':
        config = 'local'
    elif (os.path.exists(config_name) and os.path.isfile(config_name)):
        config = read_config_from_file(config_name)
    else:
        config = None
    return config

def read_config_from_file(path):
    # parse yaml file
    print("==================", path)
    procopts = None
    procopts = load(filename=path)
    return procopts


class DelayAndSumTD(SquirrelCommand):

    def make_subparser(self, subparsers):
        return subparsers.add_parser(
            'time_domain',
            help='computes delay and sum operation in time domain',
            description='time domain delay and sum')

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

        # now add another option that defines a resonable 
        # set of processing options - use parser command 'config'
        # and presets of 'teleseismic', 'regional', 'local' or
        # input via configuration file (?)
        add_argument_configuration(parser, '+')

    def run(self, parser, args):

        arrays = dict(get_matching_builtin_arrays(args.array_names))
        names = sorted(arrays.keys())

        config = get_matching_configuration(args.configuration)
        print(config)

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




class DelayAndSumFD(SquirrelCommand):

    def make_subparser(self, subparsers):
        return subparsers.add_parser(
            'frequency_domain',
            help='computes delay and sum operation in frequency domain',
            description='frequency domain delay and sum')

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


headline = 'Manage arrays setups.'


def make_subparser(subparsers):
    return subparsers.add_parser(
        'beam',
        help=headline,
        subcommands=[DelayAndSumTD(), DelayAndSumFD()],
        description=headline + '''

Manage seismic array setups: add, remove, show.
''')


def setup(parser):
    pass


def run(parser, args):
    parser.print_help()
