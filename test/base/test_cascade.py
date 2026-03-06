import os
import shutil
import tempfile
import unittest

import numpy as num

from pyrocko import util
from pyrocko import carpet as pcarpet
from pyrocko import squirrel
from pyrocko.io import rug
from pyrocko.squirrel import database as dbm
from pyrocko.squirrel.cascade import cascade
from pyrocko.squirrel.model import CodesNSLCE


class CascadeTestCase(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp(prefix='pyrocko-test-cascade-')

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def make_squirrel(self):
        database = squirrel.Database()
        self.addCleanup(dbm.close_database, database)
        return squirrel.Squirrel(database=database)

    def test_cascade(self):
        nsamples = 20000
        nlevels = 4

        # Sampling intervals not exactly representable as floats and start
        # times typical for real data, aligned to the coarsest level's
        # sampling grid.
        for deltat, tmin in [
                (0.1, 1e9),
                (0.3, 4.8 * 354166666),
                (0.001, 1e9),
                (1.0, 0.0)]:

            dirname = os.path.join(self.tempdir, '%g' % deltat)
            codes = CodesNSLCE('XX.ABC..SPEC')
            cp = pcarpet.Carpet(
                codes=codes,
                component_codes=[codes] * 2,
                component_axes={'frequency': num.arange(2.)},
                data=num.tile(num.arange(nsamples, dtype=float), (2, 1)),
                tmin=tmin,
                deltat=deltat)

            path_in = os.path.join(dirname, 'in.rug')
            util.ensuredirs(path_in)
            rug.save([cp], path_in)

            sq = self.make_squirrel()
            sq.add(path_in)
            cascade(
                sq,
                nlevels=nlevels,
                storage_path=os.path.join(dirname, 'out'),
                methods=['mean', 'min'])

            sq_out = self.make_squirrel()
            sq_out.add(os.path.join(dirname, 'out'))

            for ilevel in range(nlevels):
                nfold = 2**(ilevel+1)
                for method, offset in [
                        ('mean', (nfold - 1) / 2.),
                        ('min', 0.)]:

                    cps = sq_out.get_carpets(
                        codes='*.*.*.*.-L%s%02i' % (method, ilevel))

                    # Complete and continuous: no samples lost or duplicated
                    # at window boundaries.
                    assert len(cps) == 1
                    assert cps[0].nsamples == nsamples // nfold
                    assert abs(cps[0].tmin - tmin) < 1e-3 * deltat
                    num.testing.assert_allclose(
                        cps[0].data[0],
                        num.arange(0, nsamples, nfold) + offset)

    def test_cascade_offgrid(self):
        # Input sampled off the absolute time grid, e.g. carpets made from
        # traces. Levels must still be complete and continuous.
        nsamples = 20000
        nlevels = 4
        deltat = 0.1

        for offset in [0.3, 0.5, -0.49]:
            tmin = 1e9 + offset * deltat
            dirname = os.path.join(self.tempdir, '%g' % offset)
            codes = CodesNSLCE('XX.ABC..SPEC')
            cp = pcarpet.Carpet(
                codes=codes,
                component_codes=[codes],
                component_axes={'frequency': num.arange(1.)},
                data=num.arange(nsamples, dtype=float)[num.newaxis, :],
                tmin=tmin,
                deltat=deltat)

            path_in = os.path.join(dirname, 'in.rug')
            util.ensuredirs(path_in)
            rug.save([cp], path_in)

            sq = self.make_squirrel()
            sq.add(path_in)
            cascade(
                sq,
                nlevels=nlevels,
                storage_path=os.path.join(dirname, 'out'),
                methods=['min'])

            sq_out = self.make_squirrel()
            sq_out.add(os.path.join(dirname, 'out'))

            for ilevel in range(nlevels):
                nfold = 2**(ilevel+1)
                cps = sq_out.get_carpets(codes='*.*.*.*.-Lmin%02i' % ilevel)
                assert len(cps) == 1
                values = cps[0].data[0]
                assert num.all(num.diff(values) == nfold)
                assert values[0] < nfold
                assert values[-1] > nsamples - 1 - 2 * nfold


if __name__ == '__main__':
    util.setup_logging('test_cascade', 'warning')
    unittest.main()
