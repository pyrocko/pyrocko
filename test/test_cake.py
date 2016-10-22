import unittest
import numpy as num

from pyrocko import cake, util


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
        for i in xrange(nz):
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
        for i in xrange(nz):
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


if __name__ == "__main__":
    util.setup_logging('test_cake', 'warning')
    unittest.main()
