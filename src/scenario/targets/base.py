import numpy as num
from ..base import LocationGenerator


class TargetGenerator(LocationGenerator):

    def get_time_range(self, sources):
        ''' Get the target's time range.

        In the easiest case this is the sources' time range, yet for waveform
        targets we have to consider vmin, vmax
        '''
        times = num.array([source.time for source in sources],
                          dtype=num.float)

        return num.min(times), num.max(times)

    def set_workdir(self, path):
        self._workdir = path

    def get_stations(self):
        return []

    def get_waveforms(self, engine, sources, tmin=None, tmax=None):
        return []

    def get_insar_scenes(self, engine, sources, tmin=None, tmax=None):
        return []

    def get_gps_offsets(self, engine, sources,  tmin=None, tmax=None):
        return []

    def dump_data(self, engine, sources, path,
                  tmin=None, tmax=None, overwrite=False):
        return []
