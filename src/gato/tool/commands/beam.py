# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Implementation of :app:`gato beam`.
'''
import os
import numpy as num
from pyrocko.squirrel import SquirrelCommand
from pyrocko.gato.array import SensorArray
from pyrocko.gato.io import load
from pyrocko.gato.grid.slowness import SlownessGrid
from pyrocko.gato.grid.location import UnstructuredLocationGrid
from pyrocko.gato.delay import GenericDelayTable
from pyrocko.gato.delay.plane_wave import PlaneWaveDM
from pyrocko.guts import Object, Float, Bool

guts_prefix = 'gato'


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
    tinc = Float.T(default=60.)
    winstep = Float.T(default=30.)
    # stack option
    stacklen = Float.T(default=86400.)
    # preprocessing options:
    taper = Float.T(default=0.2)
    prewhiten = Bool.T(default=False)
    slowness_grid = SlownessGrid.T()
    # to be continued ...


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
            'time-domain',
            help='Compute delay and sum operation in time domain.',
            description='Compute delay and sum operation in time domain.')

    def setup(self, parser):
        parser.add_squirrel_selection_arguments()
        parser.add_squirrel_query_arguments(without=['kinds'])

        parser.add_argument(
            dest='array_name',
            metavar='NAME',
            help='TODO: array name or file name')

        parser.add_argument(
            dest='config_name',
            metavar='CONFIG',
            help='TODO: processing conf name or file name')

    def run(self, parser, args):

        # TODO: allow builtin names and filenames
        array = load(args.array_name, want=SensorArray)
        config = load(args.config_name, want=ProcOpts)

        sq = args.make_squirrel()
        print(sq)

        info = array.get_info(sq, **args.squirrel_query)

        receiver_grid = UnstructuredLocationGrid.from_locations(
            info.sensors, ignore_position_duplicates=False)

        gdt = GenericDelayTable(
            source_grid=config.slowness_grid,
            receiver_grid=receiver_grid,
            method=PlaneWaveDM())

        print(gdt)
        print(receiver_grid.coordinates)

        delays = gdt.get_delays()

        tpad = num.max(num.abs(delays))
        tpad += 2.0 / config.transfer_f2

        for batch in sq.chopper_waveforms(
                tinc=config.tinc, tpad=tpad, **args.squirrel_query):
            pass

        print(array)
        print(config)


class DelayAndSumFD(SquirrelCommand):

    def make_subparser(self, subparsers):
        return subparsers.add_parser(
            'frequency-domain',
            help='Compute delay and sum operation in frequency domain.',
            description='Compute delay and sum operation in frequency domain.')

    def setup(self, parser):
        parser.add_squirrel_selection_arguments()
        parser.add_squirrel_query_arguments(without=['kinds'])

        parser.add_argument(
            dest='array_name',
            metavar='NAME',
            help='TODO: array name or file name')

        parser.add_argument(
            dest='config_name',
            metavar='CONFIG',
            help='TODO: processing conf name or file name')

    def run(self, parser, args):
        pass


headline = 'Run beamforming.'


def make_subparser(subparsers):
    return subparsers.add_parser(
        'beam',
        help=headline,
        subcommands=[DelayAndSumTD(), DelayAndSumFD()],
        description=headline + '''

Apply beamforming algorithm in time or frequency domain.
''')


def setup(parser):
    pass


def run(parser, args):
    parser.print_help()
