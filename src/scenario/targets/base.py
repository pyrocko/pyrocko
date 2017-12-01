from ..base import LocationGenerator


class TargetGenerator(LocationGenerator):

    def get_waveforms(self, engine, sources):
        return []

    def get_stations(self):
        return []

    def get_insar_scenes(self, engine, sources):
        return []

    def get_gps_offsets(self, engine, sources):
        return []

    def dump_data(self, engine, sources):
        pass
