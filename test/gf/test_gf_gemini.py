import os
import unittest
import shutil
import tempfile

import numpy as num

from pyrocko import util, gf, cake
from pyrocko.guts import load_string
from pyrocko.fomosto import gemini

km = 1000.


class GeminiModelTestCase(unittest.TestCase):

    def test_model_from_cake(self):
        mod = cake.load_model()
        gmod = gemini.GeminiModel.from_cake(mod)
        gmod.validate()

        assert len(gmod.layers) <= gemini.gemini_nlayer
        dev, _ = gmod.max_deviation(mod)
        assert dev < 0.005

        names = [d.name for d in gmod.to_cake().discontinuities()]
        for name in ('moho', 'cmb', 'icb'):
            assert name in names

    def test_model_discontinuity_names(self):
        gmod = gemini.GeminiModel.from_cake(cake.load_model())

        # move all layers below the moho 1000 km down, so that the fluid
        # layer is far from the core-mantle and inner-core boundaries
        r_moho = gmod.r_surface - 35.
        for layer in gmod.layers:
            if 0.0 < layer.r_bottom < r_moho:
                layer.r_bottom = max(1.0, layer.r_bottom - 1000.)

        gmod.validate()
        names = [d.name for d in gmod.to_cake().discontinuities()]
        assert 'moho' in names
        assert 'cmb' not in names
        assert 'icb' not in names

    def test_model_string_roundtrip(self):
        gmod = gemini.GeminiModel.from_cake(cake.load_model())
        gmod2 = gemini.GeminiModel.from_string(gmod.to_string())

        assert len(gmod.layers) == len(gmod2.layers)
        assert gmod.r_surface == gmod2.r_surface
        for a, b in zip(gmod.layers, gmod2.layers):
            num.testing.assert_allclose(
                [a.r_bottom, a.qmu, a.qkappa],
                [b.r_bottom, b.qmu, b.qkappa], rtol=1e-12)
            num.testing.assert_allclose(a.coefs, b.coefs, rtol=1e-12)

    def test_model_transversely_isotropic(self):
        gmod = gemini.GeminiModel.from_cake(cake.load_model())
        gmod_ti = gemini.GeminiModel.from_string(gmod.to_string())
        gmod_ti.transversely_isotropic = True

        # with vph == vpv, vsh == vsv and eta == 1, the Voigt average must
        # reproduce the isotropic velocities
        for (layer, rb, rt), (layer_ti, _, _) in zip(
                gmod.layer_bounds(), gmod_ti.layer_bounds()):

            r = num.linspace(rb, rt, 5)
            num.testing.assert_allclose(
                gmod.isotropic_properties(layer, r),
                gmod_ti.isotropic_properties(layer_ti, r),
                rtol=1e-10)

    def test_cli_conversion(self):
        tmpdir = tempfile.mkdtemp(prefix='pyrocko.gemini')
        try:
            fn_gemini = os.path.join(tmpdir, 'model.gemini')
            fn_nd = os.path.join(tmpdir, 'model.nd')
            gemini.main(['nd-to-gemini', 'ak135-f-continental.m', fn_gemini])
            gemini.main(['gemini-to-nd', fn_gemini, fn_nd])

            gmod = gemini.GeminiModel.from_string(open(fn_gemini).read())
            mod = cake.load_model(fn_nd)
            dev, _ = gmod.max_deviation(mod)
            assert dev < 0.005

            with self.assertRaises(SystemExit):
                gemini.main(['nd-to-gemini', fn_gemini, fn_nd])

        finally:
            shutil.rmtree(tmpdir)

    def test_config_yaml(self):
        conf = gemini.GeminiConfig(
            earthmodel=gemini.GeminiModel.from_cake(cake.load_model()))

        conf2 = load_string(conf.dump())
        assert isinstance(conf2.earthmodel, gemini.GeminiModel)
        conf3 = load_string(conf2.dump())
        assert conf3.earthmodel.to_string() == conf2.earthmodel.to_string()


@unittest.skipUnless(
    gemini.have_backend(), 'backend gemini not available')
class GeminiTestCase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pyrocko.gemini')

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_gemini_build(self):
        gemini.init(self.tmpdir, None, config_params=dict(
            source_depth_max=10e3,
            distance_min=500e3,
            distance_max=600e3))

        store = gf.store.Store(self.tmpdir, 'r')
        store.make_travel_time_tables()
        gemini.build(self.tmpdir)

        engine = gf.LocalEngine(store_dirs=[self.tmpdir])

        source = gf.DCSource(
            lat=0.,
            lon=0.,
            depth=10e3,
            magnitude=6.0)

        targets = [gf.Target(
            quantity='displacement',
            codes=('', 'STA', '', comp),
            lat=0.,
            lon=0.,
            north_shift=500e3,
            east_shift=100e3) for comp in 'NEZ']

        response = engine.process(source, targets)
        for tr in response.pyrocko_traces():
            assert num.all(num.isfinite(tr.ydata))
            assert num.any(tr.ydata != 0.0)


if __name__ == '__main__':
    util.setup_logging('test_gf_gemini', 'warning')
    unittest.main()
