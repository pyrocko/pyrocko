import os
import math
import tarfile
import errno
import numpy as num
import time


from pyrocko.guts import Object, Timestamp
from pyrocko import gf, guts, util, pile, gmtpy, io

from .scenario import draw_scenario_gmt
from .base import ScenarioError

op = os.path
guts_prefix = 'pf.scenario'


def mtime(p):
    return os.stat(p).st_mtime


class ScenarioCollectionItem(Object):
    scenario_id = gf.StringID.T()
    time_created = Timestamp.T()

    def __init__(self, **kwargs):
        Object.__init__(self, **kwargs)
        self._path = None
        self._pile = None
        self._engine = None

    def set_base_path(self, path):
        self._path = path

    def init_modelling(self, engine):
        self._engine = engine

    def get_path(self, *entry):
        return op.join(*((self._path,) + entry))

    def get_generator(self):
        generator = guts.load(filename=self.get_path('generator.yaml'))
        generator.init_modelling(self._engine)
        return generator

    def have_waveforms(self, tmin, tmax):
        p = self.get_waveform_pile()
        trs_have = p.all(
            tmin=tmin, tmax=tmax, load_data=False, degap=False)

        return any(tr.data_len() > 0 for tr in trs_have)

    def get_waveform_pile(self):
        if self._pile is None:
            path_waveforms = self.get_path('waveforms')
            util.ensuredir(path_waveforms)
            fns = util.select_files(
                [path_waveforms], show_progress=False)
            self._pile = pile.Pile()
            if fns:
                self._pile.load_files(
                    fns, fileformat='mseed', show_progress=False)

        return self._pile

    def make_map(self, path_pdf):
        draw_scenario_gmt(self.get_generator(), path_pdf)

    def get_map(self, format='pdf'):
        path_pdf = self.get_path('map.pdf')

        if not op.exists(path_pdf):
            self.make_map(path_pdf)

        path = self.get_path('map.%s' % format)

        outdated = op.exists(path) and mtime(path) < mtime(path_pdf)
        if not op.exists(path) or outdated:
            gmtpy.convert_graph(path_pdf, path)

        return path

    def ensure_waveforms(self, tmin, tmax):
        path_waveforms = self.get_path('waveforms')
        path_traces = op.join(
            path_waveforms,
            '%(wmin_year)s',
            '%(wmin_month)s',
            '%(wmin_day)s',
            'waveform_%(network)s_%(station)s_'
            + '%(location)s_%(channel)s_%(tmin)s_%(tmax)s.mseed')

        generator = self.get_generator()
        tmin_all, tmax_all = generator.get_time_range()

        tmin = tmin if tmin is not None else tmin_all
        tmax = tmax if tmax is not None else tmax_all

        tinc = generator.get_useful_time_increment()

        tmin = math.floor(tmin / tinc) * tinc
        tmax = math.ceil(tmax / tinc) * tinc

        nwin = int(round((tmax - tmin) / tinc))

        p = self.get_waveform_pile()

        for iwin in range(nwin):
            tmin_win = max(tmin, tmin + iwin*tinc)
            tmax_win = min(tmax, tmin + (iwin+1)*tinc)
            if tmax_win <= tmin_win:
                continue

            if self.have_waveforms(tmin_win, tmax_win):
                continue
            trs = generator.get_waveforms(tmin_win, tmax_win)
            tts = util.time_to_str

            fns = io.save(
                trs, path_traces,
                additional=dict(
                    wmin_year=tts(tmin_win, format='%Y'),
                    wmin_month=tts(tmin_win, format='%m'),
                    wmin_day=tts(tmin_win, format='%d'),
                    wmin=tts(tmin_win, format='%Y-%m-%d_%H-%M-%S'),
                    wmax_year=tts(tmax_win, format='%Y'),
                    wmax_month=tts(tmax_win, format='%m'),
                    wmax_day=tts(tmax_win, format='%d'),
                    wmax=tts(tmax_win, format='%Y-%m-%d_%H-%M-%S')))

            if fns:
                p.load_files(fns, fileformat='mseed', show_progress=False)

        return p

    def ensure_insar_scenes(self, tmin=None, tmax=None):
        from kite import Scene

        path_insar = self.get_path('insar')
        util.ensuredir(path_insar)

        generator = self.get_generator()
        tmin, tmax = generator.get_time_range()

        tts = util.time_to_str
        fn = op.join(path_insar, 'insar-scene-{track_direction}_%s_%s'
                     % (tts(tmin), tts(tmax)))

        def scene_fn(track):
            return fn.format(track_direction=track.lower())

        for track in ('ascending', 'descending'):

            if op.exists('%s.npz' % scene_fn(track)):
                continue

            scenes = generator.get_insar_scenes(tmin, tmax)
            for sc in scenes:
                print(sc.config)
                sc.save(scene_fn(sc.meta.orbit_direction))
            return scenes

        scenes = []
        for track in ('ascending', 'descending'):
            scenes.append(Scene.load(scene_fn(track)))

        return scenes

    def get_time_range(self):
        return self.get_generator().get_time_range()

    def get_archive(self):
        path_tar = self.get_path('archive.tar')
        if not op.exists(path_tar):
            path_base = self.get_path()
            path_waveforms = self.get_path('waveforms')

            tmin, tmax = self.get_time_range()
            self.ensure_waveforms(tmin, tmax)
            fns = util.select_files(
                [path_waveforms], show_progress=False)

            f = tarfile.TarFile(path_tar, 'w')
            for fn in fns:
                fna = fn[len(path_base)+1:]
                f.add(fn, fna)

            f.close()

        return path_tar


class ScenarioCollection(object):

    def __init__(self, path, engine):
        self._scenario_suffix = 'scenario'
        self._path = path
        util.ensuredir(self._path)
        self._engine = engine
        self._load_scenarios()

    def _load_scenarios(self):
        scenarios = []
        base_path = self.get_path()
        for path_entry in os.listdir(base_path):
            scenario_id, suffix = op.splitext(path_entry)
            if suffix == '.' + self._scenario_suffix:
                path = op.join(base_path, path_entry, 'scenario.yaml')
                scenario = guts.load(filename=path)
                assert scenario.scenario_id == scenario_id
                scenario.set_base_path(op.join(base_path, path_entry))
                scenario.init_modelling(self._engine)
                scenarios.append(scenario)

        self._scenarios = scenarios
        self._scenarios.sort(key=lambda s: s.time_created)

    def get_path(self, scenario_id=None, *entry):
        if scenario_id is not None:
            return op.join(self._path, '%s.%s' % (
                scenario_id, self._scenario_suffix), *entry)
        else:
            return self._path

    def add_scenario(self, scenario_id, scenario_generator):

        if scenario_generator.seed is None:
            scenario_generator = guts.clone(scenario_generator)
            scenario_generator.seed = num.random.randint(1, 2**32-1)

        path = self.get_path(scenario_id)
        try:
            os.mkdir(path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                raise ScenarioError(
                    'scenario id is already in use: %s' % scenario_id)
            else:
                raise

        scenario = ScenarioCollectionItem(
            scenario_id=scenario_id,
            time_created=time.time())

        scenario_path = self.get_path(scenario_id, 'scenario.yaml')
        guts.dump(scenario, filename=scenario_path)

        generator_path = self.get_path(scenario_id, 'generator.yaml')
        guts.dump(scenario_generator, filename=generator_path)

        scenario.set_base_path(self.get_path(scenario_id))
        scenario.init_modelling(self._engine)

        self._scenarios.append(scenario)

    def list_scenarios(self, ilo=None, ihi=None):
        return self._scenarios[ilo:ihi]

    def get_scenario(self, scenario_id):
        for scenario in self._scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario

        raise KeyError(scenario_id)
