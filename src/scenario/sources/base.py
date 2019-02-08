import os.path as op

from pyrocko import util, moment_tensor
from pyrocko.guts import Timestamp, Float, Int, Bool

from ..base import LocationGenerator

km = 1e3
guts_prefix = 'pf.scenario'


class SourceGenerator(LocationGenerator):

    nevents = Int.T(default=2)
    avoid_water = Bool.T(
        default=False,
        help='Avoid sources offshore under the ocean / lakes.')

    radius = Float.T(
        default=10*km)

    time_min = Timestamp.T(default=util.str_to_time('2017-01-01 00:00:00'))
    time_max = Timestamp.T(default=util.str_to_time('2017-01-03 00:00:00'))

    magnitude_min = Float.T(default=4.0)
    magnitude_max = Float.T(default=0.0)
    b_value = Float.T(
        optional=True, help='Gutenberg Richter magnitude distribution.')

    def __init__(self, *args, **kwargs):
        super(SourceGenerator, self).__init__(*args, **kwargs)
        if self.b_value and self.magnitude_max:
            raise Exception('b_value and magnitude_max are mutually exclusive')

    def draw_magnitude(self, rstate):
        if self.b_value is None:
            return rstate.uniform(self.magnitude_min, self.magnitude_max)
        else:
            return moment_tensor.rand_to_gutenberg_richter(
                rstate.rand(), self.b_value, magnitude_min=self.magnitude_min)

    def get_sources(self):
        sources = []
        for ievent in range(self.nevents):
            src = self.get_source(ievent)
            src.name = 'scenario_ev%03d' % (ievent + 1)
            sources.append(src)

        return sources

    def dump_data(self, path):
        fn_sources = op.join(path, 'sources.yml')
        with open(fn_sources, 'w') as f:
            for src in self.get_sources():
                f.write(src.dump())

        fn_events = op.join(path, 'events.txt')
        with open(fn_events, 'w') as f:
            for isrc, src in enumerate(self.get_sources()):
                f.write(src.pyrocko_event().dump())

        return [fn_events, fn_sources]

    def add_map_artists(self, automap):
        pass
