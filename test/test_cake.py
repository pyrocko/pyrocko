import unittest
import numpy as num

from pyrocko import cake, util



class CakeTestCase(unittest.TestCase):

    def test_random_model_cut(self):
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
            interface_material_top = mod.material(zmin)
            interface_material_bot = mod.material(zmax)
            self.assertEqual(elements[0].mtop, interface_material_top)
            self.assertEqual(elements[-1].mbot, interface_material_bot)
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
            if zmin == 0.:
                assert isinstance(elements[0], cake.Surface)
            else:
                self.assertEqual(elements[0].mtop, interface_material_top)
            self.assertEqual(elements[-1].mbot, interface_material_bot)
            self.assertEqual(elements[0].ztop, zmin)
            self.assertEqual(elements[-1].zbot, zmax)

if __name__ == "__main__":
    util.setup_logging('test_moment_tensor', 'warning')
    unittest.main()
