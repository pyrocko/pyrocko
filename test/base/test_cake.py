
import math
import unittest
import numpy as num
from io import BytesIO

from pyrocko import cake, util

km = 1000.


class CakeTestCase(unittest.TestCase):

    def test_copy(self):
        m1 = cake.Material(vp=1., vs=1.)
        m2 = cake.Material(vp=4., vs=4.)

        s = cake.Surface(0., m1)
        self.assertEqual(s.__dict__, s.copy().__dict__)

        i = cake.Interface(1., m1, m2, name='a')
        self.assertEqual(i.__dict__, i.copy().__dict__)

    def test_layer_resize(self):
        depth_min_init = 0.
        depth_max_init = 3.
        m1 = cake.Material(vp=1., vs=1.)
        m2 = cake.Material(vp=4., vs=4.)

        tests = [
            (cake.HomogeneousLayer, 1., None, 2., m1, m1),
            (cake.HomogeneousLayer, None, 2., 1., m1, m1),
            (cake.HomogeneousLayer, 1., 2., 1.5, m1, m1),
            (cake.GradientLayer, 1., None, 2., cake.Material(2., 2.), m2),
            (cake.GradientLayer, None, 2., 1., m1, cake.Material(3., 3.)),
            (cake.GradientLayer, 1., 2., 1.5,
                cake.Material(2., 2.,), cake.Material(3., 3.)),
            (cake.GradientLayer, None, 4., 2., m1, cake.Material(5., 5.)),
        ]

        for t in tests:
            cls, depth_min, depth_max, depth_mid, nmtop, nmbot = t
            if cls == cake.HomogeneousLayer:
                layer = cls(ztop=depth_min_init, zbot=depth_max_init, m=m1)
            if cls == cake.GradientLayer:
                layer = cls(ztop=depth_min_init, zbot=depth_max_init,
                            mtop=m1, mbot=m2)
            depth_min = depth_min or depth_min_init
            depth_max = depth_max or depth_max_init

            layer.resize(depth_min=depth_min, depth_max=depth_max)
            self.assertEqual(layer.ztop, depth_min)
            self.assertEqual(layer.zbot, depth_max)
            self.assertEqual(layer.zmid, num.mean((depth_min, depth_max)))

            self.assertEqual(layer.mtop, nmtop)
            self.assertEqual(layer.mbot, nmbot)

    def test_single_layer_extract(self):
        mod = cake.load_model()
        zmin = 100.
        zmax = 200.
        new_mod = mod.extract(zmin, zmax)
        elements = list(new_mod.elements())
        interface_material_top = mod.material(zmin)
        interface_material_bot = mod.material(zmax)
        if not isinstance(elements[0], cake.Surface):
            self.assertEqual(elements[0].mtop, interface_material_top)
        self.assertEqual(elements[-1].mbot, interface_material_bot)
        self.assertEqual(elements[0].ztop, zmin)
        self.assertEqual(elements[-1].zbot, zmax)

    def test_random_model_extract(self):
        nz = 100
        mod = cake.load_model()
        layers = list(mod.elements())
        zmin = layers[0].ztop
        zmax = layers[mod.nlayers-1].zbot
        zmins = num.random.uniform(zmin, zmax, nz)
        zmaxs = num.random.uniform(zmin, zmax, nz)
        for i in range(nz):
            zmin = min(zmins[i], zmaxs[i])
            zmax = max(zmins[i], zmaxs[i])
            new_mod = mod.extract(zmin, zmax)

            elements = list(new_mod.elements())
            n_layers = len([e for e in elements if isinstance(e, cake.Layer)])
            interface_material_top = mod.material(zmin)
            interface_material_bot = mod.material(zmax)
            if not isinstance(elements[0], cake.Surface):
                self.assertEqual(elements[0].mtop, interface_material_top)
            if isinstance(elements[-1], cake.Layer):
                self.assertEqual(elements[-1].ilayer, n_layers-1)
                for k, v in elements[-1].mbot.__dict__.items():
                    self.assertAlmostEqual(
                        v, interface_material_bot.__dict__[k], 6)
            self.assertEqual(elements[0].ztop, zmin)
            self.assertEqual(elements[-1].zbot, zmax)

    def test_interface_model_extract(self):
        nz = 100
        mod = cake.load_model()
        layers = list(mod.elements())
        for i in range(nz):
            i = num.random.randint(0, len(layers)-3)
            i2 = num.random.randint(i+2, len(layers)-1)
            z1 = layers[i].zbot
            z2 = layers[i2].zbot
            zmin = min(z1, z2)
            zmax = max(z1, z2)
            new_mod = mod.extract(zmin, zmax)
            interface_material_top = mod.material(zmin)
            interface_material_bot = mod.material(zmax)
            elements = list(new_mod.elements())
            n_layers = len([e for e in elements if isinstance(e, cake.Layer)])
            if isinstance(elements[0], cake.Layer):
                self.assertEqual(elements[0].mtop, interface_material_top)
                self.assertEqual(elements[0].ilayer, 0)
            if isinstance(elements[-1], cake.Layer):
                self.assertEqual(elements[-1].ilayer, n_layers-1)
                self.assertEqual(elements[-1].mbot, interface_material_bot)
            self.assertEqual(elements[0].ztop, zmin)
            self.assertEqual(elements[-1].zbot, zmax)
            if zmin == 0.:
                assert isinstance(elements[0], cake.Surface)

    def test_material(self):
        mat = cake.Material(
            poisson=0.20, rho=3000., qp=100.)
        mat1 = cake.Material(
            vp=mat.vp, poisson=mat.poisson(), rho=mat.rho, qp=mat.qp)
        mat2 = cake.Material(
            vp=mat.vp, vs=mat1.vs, rho=mat1.rho, qs=mat1.qs)
        mat3 = cake.Material(
            lame=mat2.lame(), rho=mat2.rho, qp=mat2.qp, qs=mat2.qs)
        mat4 = cake.Material(
            vs=mat3.vs, poisson=mat3.poisson(), rho=mat3.rho,
            qk=mat3.qk(), qmu=mat3.qmu())

        mat5 = eval('cake.'+repr(mat))

        for matx in (mat1, mat2, mat3, mat4, mat5):
            self.assertEqual(mat.vp, matx.vp)
            self.assertEqual(mat.vs, matx.vs)
            self.assertEqual(mat.rho, matx.rho)
            self.assertEqual(mat.qp, matx.qp)
            self.assertEqual(mat.qs, matx.qs)
            self.assertEqual(mat.describe(), matx.describe())

    def test_classic(self):
        phase = cake.PhaseDef.classic('PP')[0]
        assert str(phase) == '''
Phase definition "P<(cmb)(moho)pP<(cmb)(moho)p":
 - P mode propagation, departing downward \
(may not propagate deeper than interface cmb)
 - passing through moho on upgoing path
 - P mode propagation, departing upward
 - surface reflection
 - P mode propagation, departing downward \
(may not propagate deeper than interface cmb)
 - passing through moho on upgoing path
 - P mode propagation, departing upward
 - arriving at target from below'''.strip()

        mod = cake.load_model()
        rays = mod.arrivals(
            phases=[phase], distances=[5000*km*cake.m2d], zstart=500.)

        assert str(rays[0]).split() == '''10.669 s/deg    5000 km  601.9 s  \
33.8   33.8  17%  12% P<(cmb)(moho)pP<(cmb)(moho)p (P^0P)            \
0_1_2_3_(4-5)_(6-7)_8_(7-6)_(5-4)_3_2_1_0|\
0_1_2_3_(4-5)_(6-7)_8_(7-6)_(5-4)_3_2_1_0'''.split()

        assert abs(rays[0].t - 601.9) < 0.2

    def test_simplify(self):
        mod1 = cake.load_model()
        mod2 = mod1.simplify()
        phases = cake.PhaseDef.classic('Pdiff')
        rays1 = mod1.arrivals(phases=phases, distances=[120.], zstart=500.)
        rays2 = mod2.arrivals(phases=phases, distances=[120.], zstart=500.)
        assert abs(rays1[0].t - 915.9) < 0.1
        assert abs(rays2[0].t - 915.9) < 0.1

    def test_path(self):
        mod = cake.load_model()
        phase = cake.PhaseDef('P')
        ray = mod.arrivals(phases=[phase], distances=[70.], zstart=100.)

        z, x, t = ray[0].zxt_path_subdivided()
        assert z[0].size == 681

    def test_to_phase_defs(self):
        pdefs = cake.to_phase_defs(['p,P', cake.PhaseDef('PP')])
        assert len(pdefs) == 3
        for pdef in pdefs:
            assert isinstance(pdef, cake.PhaseDef)

        pdefs = cake.to_phase_defs(cake.PhaseDef('PP'))
        assert len(pdefs) == 1
        for pdef in pdefs:
            assert isinstance(pdef, cake.PhaseDef)

        pdefs = cake.to_phase_defs('P,p')
        assert len(pdefs) == 2
        for pdef in pdefs:
            assert isinstance(pdef, cake.PhaseDef)

    def test_model_io(self):
        mod = cake.load_model()
        s = cake.write_nd_model_str(mod)
        assert isinstance(s, str)

    def test_dump(self):
        f = BytesIO()
        mod = cake.load_model()
        mod.profile('vp').dump(f)
        f.close()

    def test_angles(self):
        mod = cake.load_model()
        data = [
            [1.0*km, 1.0*km, 1.0*km, 90., 90., 'P'],
            [1.0*km, 2.0*km, 1.0*km, 45., 135., 'P\\'],
            [2.0*km, 1.0*km, 1.0*km, 135., 45., 'p'],
            [1.0*km, 2.0*km, math.sqrt(3.)*km, 60., 120., 'P\\'],
            [2.0*km, 1.0*km, math.sqrt(3.)*km, 120., 60., 'p']]

        for (zstart, zstop, dist, takeoff_want, incidence_want, pdef_want) \
                in data:

            rays = mod.arrivals(
                zstart=zstart,
                zstop=zstop,
                phases=[cake.PhaseDef(sphase)
                        for sphase in 'P,p,P\\,p\\'.split(',')],
                distances=[dist*cake.m2d])

            for ray in rays:
                takeoff = round(ray.takeoff_angle())
                incidence = round(ray.incidence_angle())
                pdef = ray.used_phase().definition()

                assert takeoff == takeoff_want
                assert incidence == incidence_want
                assert pdef == pdef_want


if __name__ == '__main__':
    util.setup_logging('test_cake', 'warning')
    unittest.main()
