import math
import shutil
import tempfile
import unittest

import numpy as num

from pyrocko import fomosto, util, gf
from pyrocko.apps.fomosto import main as fomosto_main
from pyrocko.fomosto import ahfullgreen, gemini, qssp


class FomostoBackendsTestCase(unittest.TestCase):

    def test_backend_variants(self):
        entries = fomosto.get_backend_variants()
        missing = dict(entries)

        for modelling_code_id in (
                'ahfullgreen', 'gemini.2.2', 'qseis2d.2014', 'qssp.2017'):
            assert modelling_code_id in missing

        # all backend modules can be imported
        assert all(m is not None for m in missing.values())

        # converters without init are not listed
        assert not any(i.startswith('elastic10') for i in missing)

        # built into Pyrocko, needs no executables
        assert missing['ahfullgreen'] == []

    def test_init_requires_variant(self):
        tmpdir = tempfile.mkdtemp(prefix='pyrocko.fomosto')
        try:
            for modelling_code_id in ('qssp', 'gemini', 'gemini.3.0'):
                with self.assertRaises(SystemExit):
                    fomosto_main(
                        ['init', modelling_code_id, tmpdir + '/store'])

            fomosto_main(['init', 'gemini.2.2', tmpdir + '/store'])

        finally:
            shutil.rmtree(tmpdir)

    def test_init_api_requires_variant(self):
        tmpdir = tempfile.mkdtemp(prefix='pyrocko.fomosto')
        try:
            for module, variant in [
                    (qssp, None),
                    (gemini, None),
                    (gemini, '3.0'),
                    (ahfullgreen, '1.0')]:

                with self.assertRaises(gf.StoreError):
                    module.init(tmpdir + '/store', variant)

            ahfullgreen.init(tmpdir + '/store', None)

        finally:
            shutil.rmtree(tmpdir)

    def test_correct_rectangle_rule(self):
        deltat = 1.0
        t = num.arange(200) * deltat
        t0, sigma = 80.3, 1.5

        # velocity: gaussian pulse, displacement: step with static offset
        v = num.exp(-0.5 * ((t - t0) / sigma)**2) \
            / (math.sqrt(2. * math.pi) * sigma)
        u = 0.5 * (1.0 + num.array(
            [math.erf((x - t0) / (math.sqrt(2.) * sigma)) for x in t]))

        # running sum over preceding samples as done by QSSP
        y = num.zeros(t.size)
        y[1:] = num.cumsum(v[:-1]) * deltat

        def misfit(a):
            return num.max(num.abs(a - u))

        assert misfit(y) > 0.1
        assert misfit(fomosto.correct_rectangle_rule(y, deltat)) < 1e-3


if __name__ == '__main__':
    util.setup_logging('test_fomosto_backends', 'warning')
    unittest.main()
