from __future__ import division, print_function, absolute_import
import unittest
from tempfile import mkdtemp
import shutil

from pyrocko import scenario, util, gf

km = 1000.


def have_store(store_id):
    engine = gf.get_engine()
    try:
        engine.get_store(store_id)
        return True
    except gf.NoSuchStore:
        return False


class ScenarioTestCase(unittest.TestCase):
    store_id = 'crust2_m5_hardtop_8Hz_fine'
    store_id_static = 'ak135_static'

    tempdirs = []

    generator = scenario.ScenarioGenerator(
            seed=20,
            center_lat=42.6,
            center_lon=13.3,
            # center_lat=28.6,
            # center_lon=-115.3,
            radius=80*km,
            station_generator=scenario.RandomStationGenerator(
                nstations=5),
            source_generator=scenario.generators.DCSourceGenerator(
                time_min=util.str_to_time('2017-01-01 00:00:00'),
                time_max=util.str_to_time('2017-01-01 02:00:00'),
                radius=10*km,
                depth_min=1*km,
                depth_max=10*km,
                magnitude_min=3.0,
                magnitude_max=4.5,
                strike=120.,
                dip=45.,
                rake=90.,
                perturbation_angle_std=15.,
                nevents=3),
            insar_generator=scenario.generators.InSARDisplacementGenerator(
                swath_width=100*km,
                track_length=50*km,
                resolution=(200, 200)),
            store_id=store_id,
            store_id_static=store_id_static,
            seismogram_quantity='velocity')

    @classmethod
    def tearDownClass(cls):
        for d in cls.tempdirs:
            continue
            shutil.rmtree(d)

    @unittest.skipUnless(
            have_store(store_id),
            'GF Store "%s" is not available' % store_id)
    def test_scenario_waveforms(self):
        tempdir = mkdtemp(prefix='pyrocko-scenario')
        self.tempdirs.append(tempdir)

        vmin = 2500.
        generator = self.generator

        def twin(source):
            tmin = source.time
            tmax = source.time + 100*km / vmin
            return tmin, tmax

        engine = gf.get_engine()
        generator.init_modelling(engine)

        ref_sources = generator.get_sources()
        ref_trs_list = []
        for source in ref_sources:
            trs = generator.get_waveforms(*twin(source))
            trs.sort(key=lambda tr: tr.nslc_id)
            ref_trs_list.append(trs)

        collection = scenario.ScenarioCollection(tempdir, engine)
        collection.add_scenario('one', generator)
        with self.assertRaises(scenario.ScenarioError):
            collection.add_scenario('one', generator)

        assert len(collection.list_scenarios()) == 1
        assert collection.list_scenarios()[0].scenario_id == 'one'

        s = collection.get_scenario('one')

        for ref_trs, source in zip(
                ref_trs_list,
                s.get_generator().get_sources()):

            trs = generator.get_waveforms(*twin(source))
            trs.sort(key=lambda tr: tr.nslc_id)
            self.assert_traces_almost_equal(trs, ref_trs)

        collection2 = scenario.ScenarioCollection(tempdir, engine)

        assert len(collection2.list_scenarios()) == 1
        assert collection2.list_scenarios()[0].scenario_id == 'one'

        s = collection2.get_scenario('one')

        for ref_trs, source in zip(
                ref_trs_list,
                s.get_generator().get_sources()):

            trs = generator.get_waveforms(*twin(source))
            trs.sort(key=lambda tr: tr.nslc_id)
            self.assert_traces_almost_equal(trs, ref_trs)

        p = s.get_waveform_pile()
        tmin, tmax = s.get_time_range()
        s.ensure_waveforms(tmin, tmax)

        for ref_trs, source in zip(
                ref_trs_list,
                s.get_generator().get_sources()):

            tmin, tmax = twin(source)
            trs = p.all(tmin=tmin, tmax=tmax, include_last=False)
            trs.sort(key=lambda tr: tr.nslc_id)
            self.assert_traces_almost_equal(trs, ref_trs)

    def test_scenario_insar(self):
        tempdir = mkdtemp(prefix='pyrocko-scenario')
        self.tempdirs.append(tempdir)

        generator = self.generator
        engine = gf.get_engine()
        generator.init_modelling(engine)

        collection = scenario.ScenarioCollection(tempdir, engine)
        collection.add_scenario('insar', generator)

        s = collection.get_scenario('insar')
        s.ensure_insar_scenes()

    def test_scenario_map(self):
        tempdir = mkdtemp(prefix='pyrocko-scenario')
        self.tempdirs.append(tempdir)

        generator = self.generator
        engine = gf.get_engine()

        collection = scenario.ScenarioCollection(tempdir, engine)
        collection.add_scenario('plot', generator)

        s = collection.get_scenario('plot')
        s.get_map()

    def assert_traces_almost_equal(self, trs1, trs2):
        assert len(trs1) == len(trs2)
        for (tr1, tr2) in zip(trs1, trs2):
            tr1.assert_almost_equal(tr2)


if __name__ == '__main__':
    util.setup_logging('test_scenario', 'warning')
    unittest.main()
