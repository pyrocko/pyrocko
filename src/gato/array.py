from collections import defaultdict

import numpy as num

from pyrocko import util
from pyrocko.guts import Object, String, List, Timestamp, Dict, Tuple, Float
from pyrocko.model.location import Location
from pyrocko.squirrel import CodesNSLCE, FDSNSource, Dataset, CodesNSL, Sensor

from pyrocko.has_paths import HasPaths, Path
from .grid.location import UnstructuredLocationGrid, distances_3d

guts_prefix = 'gato'
km = 1000.


def time_or_none_to_str(t):
    return util.time_to_str(t, '%Y-%m-%d') if t else '...'


class SensorArrayInfo(Object):
    name = String.T(optional=True)
    codes = List.T(CodesNSLCE.T())
    tmin = Timestamp.T(optional=True)
    tmax = Timestamp.T(optional=True)
    center = Location.T(optional=True)
    distances_stats = Tuple.T(5, Float.T(), optional=True)
    codes_nsl__ = List.T(CodesNSL.T())
    codes_nsl_by_channels = Dict.T(Tuple.T(String.T()), List.T(CodesNSL.T()))
    sensors = List.T(Sensor.T())

    @property
    def summary(self):
        return ' | '.join((
            self.name.ljust(15),
            '%2i' % len(self.codes_nsl),
            '%2i' % len(self.codes),
            time_or_none_to_str(self.tmin).ljust(10),
            time_or_none_to_str(self.tmax).ljust(10),
            ', '.join('%5.1f' % (v/km) for v in self.distances_stats)
            if self.distances_stats is not None else ' ' * 33,
            ', '.join('%s: %i' % (','.join(k), len(v))
                      for (k, v) in self.codes_nsl_by_channels.items())))

    @property
    def codes_nsl(self):
        return sorted(set(CodesNSL(c) for c in self.codes))

    @codes_nsl.setter
    def codes_nsl(self, _):
        pass


class SensorArray(Object):
    name = String.T(optional=True)
    codes = List.T(CodesNSLCE.T())
    comment = String.T(optional=True)

    def get_info(self, sq, channels=None, time=None, tmin=None, tmax=None):
        if isinstance(channels, str):
            channels = [channels]

        codes_query = set()
        if channels is not None:
            for c in self.codes:
                for cha in channels:
                    codes_query.add(c.replace(channel=cha))

            codes_query = sorted(codes_query)
        else:
            codes_query = self.codes

        tmins = []
        tmaxs = []
        codes = set()
        nsl_by_chas = defaultdict(set)

        sensors = sq.get_sensors(
                codes=codes_query, time=time, tmin=tmin, tmax=tmax)

        for sensor in sensors:
            for channel in sensor.channels:
                codes.add(channel.codes)

            sensor_chas = tuple(
                sorted(set(
                    channel.codes.channel
                    for channel in sensor.channels)))

            nsl_by_chas[sensor_chas].add(CodesNSL(sensor.codes))

            tmins.append(sensor.tmin)
            tmaxs.append(sensor.tmax)

        tmins = [tmin for tmin in tmins if tmin is not None]
        tmin = min(tmins) if tmins else None
        tmax = None if None in tmaxs or not tmaxs else max(tmaxs)

        if sensors:
            grid = UnstructuredLocationGrid.from_locations(
                sensors, ignore_position_duplicates=True)

            center = grid.get_center()
            distances = distances_3d(grid, grid).flatten()
            distances = distances[distances != 0.0]
            distances_stats = tuple(num.percentile(
                distances, [0., 10., 50., 90., 100.])) \
                if distances.size != 0 else None
        else:
            center = None
            distances_stats = None

        codes_nsl_by_channels = dict(
            (k, sorted(v)) for (k, v) in nsl_by_chas.items())

        return SensorArrayInfo(
            name=self.name,
            codes=sorted(codes),
            codes_nsl_by_channels=codes_nsl_by_channels,
            tmin=tmin,
            tmax=tmax,
            distances_stats=distances_stats,
            center=center,
            sensors=sensors)


class SensorArrayFromFDSN(SensorArray):
    sources = List.T(FDSNSource.T())


def to_codes(codes):
    return [CodesNSLCE(c) for c in codes]


def _make_fdsn_source(site, codes):
    return FDSNSource(site=site, codes=codes)


g_sensor_arrays = [
    SensorArrayFromFDSN(
        name=name,
        codes=to_codes(codes),
        sources=[_make_fdsn_source('iris', codes)])

    for (name, codes) in [
        ('h-dghan', ['IM.H08N?.*.?DH']),
        ('h-dghas', ['IM.H08S?.*.?DH']),
        ('i-bermuda', ['IM.I51H?.*.?DF']),
        ('i-cocos-island', ['IM.I06H?.*.?DF']),
        ('i-dgha-land', ['IM.I52H?.*.?DF']),
        ('i-fairbanks', ['IM.I53H?.*.?DF']),
        ('i-hia', ['IM.I59H?.*.?DF']),
        ('i-narrogin', ['IM.I04H?.*.?DF']),
        ('i-nia', ['IM.I56H?.*.?DF']),
        ('i-pfia', ['IM.I57H?.*.?DF', 'IM.I57L?.*.?DF']),
        ('i-tdc', ['IM.H09N?.*.?DF', 'IM.I49H?.*.?DF']),
        ('i-warramunga', ['IM.I07H?.*.?DF']),
        ('s-alice-springs', ['AU.AS*.*.?H?']),
        ('s-bca', ['IM.BC0?.*.?H?']),
        ('s-bma', ['IM.BM0?.*.?H?']),
        ('s-esk', ['IM.EKB?.*.?H?', 'IM.EKR*.*.?H?']),
        ('s-ilar', ['IM.IL*.*.?H?']),
        ('s-imar', ['IM.IM0?.*.?H?']),
        ('s-nvar', ['IM.NV*.*.?H?']),
        ('s-pdar', ['IM.PD0*.*.?H?', 'IM.PD1*.*.?H?']),
        ('s-pilbara', ['AU.PSA*.*.?H?']),
        ('s-txar', ['IM.TX*.*.?H?']),
        ('s-yka', ['CN.YKA*.*.?H?']),
    ]
] + [
    SensorArrayFromFDSN(
        name=name,
        codes=to_codes(codes),
        sources=[_make_fdsn_source('geofon', codes)])

    for (name, codes) in [
        ('s-rohrbach', ['6A.V*.*.?H?']),
        ('s-anta-onshore', ['AW.VNA*.*.?H?']),
    ]
] + [
    SensorArrayFromFDSN(
        name=name,
        codes=to_codes(codes),
        sources=[_make_fdsn_source('bgr', codes)])

    for (name, codes) in [
        ('s-geres', [
            'GR.GEA?.*.?H?',
            'GR.GEB?.*.?H?',
            'GR.GEC?.*.?H?',
            'GR.GED?.*.?H?']),
        ('s-grf', ['GR.GR??.*.?H?']),
    ]
]

g_sensor_arrays_dict = dict(
    (array.name, array) for array in g_sensor_arrays)


class SensorArrayFromFile(SensorArray, HasPaths):
    name = String.T()
    stations_path = Path.T()


def get_named_arrays_dataset(names=None):

    if isinstance(names, str):
        names = [names]

    sources = []
    for array in g_sensor_arrays:
        if names is None or array.name in names:
            sources.extend(array.sources)

    if names is None:
        comment = 'Aggregated dataset containing data sources for all ' \
            'built-in arrays of Gato.'
    else:
        if len(names) == 1:
            comment = 'Data source for Gato array: %s' % names[0]
        else:
            comment = 'Data sources for Gato array: %s' % ', ' .join(names)

    return Dataset(sources=sources, comment=comment)


def get_named_arrays():
    return g_sensor_arrays_dict


def get_named_array(name):
    return g_sensor_arrays_dict[name]


__all__ = [
    'SensorArrayInfo',
    'SensorArray',
    'SensorArrayFromFDSN',
    'get_named_arrays_dataset',
    'get_named_arrays',
    'get_named_array',
]
