# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Implementation of :app:`gato simulate`.
'''

import numpy as num

from pyrocko import util, trace, model
from pyrocko.gato.tool.common import add_sensor_array_arguments, \
    sensor_arrays_from_arguments
from pyrocko.squirrel.storage import get_storage_scheme

from pyrocko.gato.error import GatoToolError
from pyrocko.gato.array import SensorArrayAndIncarnationContext, \
    get_named_arrays_dataset


headline = 'Synthesize test signals.'


def make_subparser(subparsers):
    return subparsers.add_parser(
        'simulate',
        help=headline,
        description=headline)


def setup(parser):
    add_sensor_array_arguments(parser)
    parser.add_squirrel_selection_arguments()
    parser.add_squirrel_query_arguments(without=['kinds'])

    parser.add_argument(
        '--out-storage-path',
        dest='out_storage_path',
        metavar='PATH',
        help='Store output in directory PATH.')


def run(parser, args):
    arrays = sensor_arrays_from_arguments(args)

    sq = args.make_squirrel()
    sq.add_dataset(get_named_arrays_dataset(
        sorted(array.name for array in arrays)))

    squirrel_query = dict(args.squirrel_query)
    tmin = squirrel_query['tmin']
    tmax = squirrel_query['tmax']
    tinc = 3600.

    storage = get_storage_scheme('default')
    if not args.out_storage_path:
        raise GatoToolError(
            'Specify output storage directory with --out-storage-path')

    storage.set_base_path(args.out_storage_path)

    for array in arrays:
        incarnation = array.get_incarnation(sq, **squirrel_query)

        if incarnation.n_codes == 0:
            raise GatoToolError(
                'No sensors match given combination of array definition '
                'and available metadata. Context:\n'
                + str(SensorArrayAndIncarnationContext(
                    array=array, incarnation=incarnation)))

        source_location = model.Location(
            lat=63.879167, lon=-22.387222,
            north_shift=0.0, east_shift=0.0)

        velocity = 2000.

        distance_max = max(
            source_location.distance_to(channel)
            for sensor in incarnation.sensors for channel in sensor.channels)

        deltat_min = min(
            channel.deltat
            for sensor in incarnation.sensors for channel in sensor.channels)

        tpad = 1.1 * distance_max / velocity
        nsamples = int(round(((tmax+tpad) - (tmin-tpad)) / deltat_min))
        signal_data = num.random.normal(size=nsamples)
        tr_signal = trace.Trace(
            '', '', '', '',
            tmin=tmin-tpad,
            ydata=signal_data,
            deltat=deltat_min)

        for batch in util.iter_windows(tmin=tmin, tmax=tmax, tinc=tinc):
            for isensor, sensor in enumerate(incarnation.sensors):
                for channel in sensor.channels:
                    nsamples = int(round(
                        (batch.tmax - batch.tmin) / channel.deltat))
                    data = num.random.normal(size=nsamples) * 0.001
                    tr = trace.Trace(
                        channel.codes.network,
                        channel.codes.station,
                        channel.codes.location,
                        channel.codes.channel,
                        tmin=batch.tmin,
                        deltat=channel.deltat,
                        ydata=data)

                    distance = source_location.distance_to(channel)
                    # time = batch.tmin + num.arange(nsamples) * channel.deltat
                    # for frequency in frequencies:
                    #     data += 1.0 / len(frequencies) * num.sin(
                    #         2.0 * num.pi * frequency * time
                    #         - distance / velocity)

                    if isensor <= batch.i:
                        tr_signal_copy = tr_signal.chop(
                            batch.tmin - tpad, batch.tmax + tpad,
                            inplace=False)

                        tr_signal_copy.downsample_to(
                            tr.deltat, allow_upsample_max=8)

                        tr_signal_copy.shift(distance/velocity)

                        tr.add(tr_signal_copy)

                    tr.ydata *= 1e6
                    tr.ydata = tr.ydata.astype(num.int32)

                    storage.save(tr)
