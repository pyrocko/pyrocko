
import os
import tempfile
import unittest

import numpy as num

from pyrocko import trace, carpet
from pyrocko import util
from pyrocko.io import rug
from pyrocko.io.io_common import FileSaveError
from pyrocko.squirrel.model import CodesNSLCE


class CarpetTestCase(unittest.TestCase):

    def random_traces(
            self,
            tmin=util.str_to_time('1970-01-01 00:00:00'),
            n=10,
            deltat_choices=[0.5, 1., 2.],
            toffset_range=[-5., 5.],
            tlen_range=[200., 300.],
            dtype_choices=[float, int, num.int32]):

        tpad = max(
            trace.downsample_tpad(
                deltat, max(deltat_choices), allow_upsample_max=5)
            for deltat in deltat_choices)

        tpad += num.max(num.abs(toffset_range))

        tmin = tmin - tpad

        toffset = num.random.uniform(*toffset_range, n)
        deltats = num.random.choice(deltat_choices, n)
        tlens = num.random.uniform(*tlen_range, n) + 2 * tpad
        nsamples = (tlens / deltats).astype(int)
        dtypes = num.random.choice(dtype_choices, n)

        trs = []
        for i in range(n):
            data = num.random.normal(0., 1000., nsamples[i]).astype(dtypes[i])
            trs.append(trace.Trace(
                'NX', 'S%03i' % i, 'RAW', 'Z',
                tmin=tmin+toffset[i],
                deltat=deltats[i],
                ydata=data))

        return trs

    def test_creation(self):
        tmin = util.str_to_time('2023-06-13 00:00:00')
        tlen_range = [200., 300.]
        traces = self.random_traces(n=100, tmin=tmin, tlen_range=tlen_range)
        traces = trace.make_traces_compatible(traces)
        mt = carpet.Carpet(traces)
        assert mt.tmin <= tmin
        assert mt.tmax >= tmin + tlen_range[0]
        # mt.snuffle()

    def test_downsample(self):
        traces = self.random_traces(n=100)
        for tr in traces:
            tpad = trace.downsample_tpad(tr.deltat, 2.)
            tr_orig = tr.copy()
            tr.downsample_to(2., cut=True, snap=True)
            assert tr.tmin <= tr_orig.tmin + tpad
            assert tr_orig.tmax - tpad <= tr.tmax

    def _make_rug_carpet(self, tmin, nsamples, value, deltat=1.0, ncomp=3):
        codes = CodesNSLCE('XX.ABC..SPEC')
        return carpet.Carpet(
            codes=codes,
            component_codes=[codes] * ncomp,
            component_axes={'frequency': num.arange(ncomp, dtype=float)},
            nsamples=nsamples,
            data=num.full((ncomp, nsamples), value, dtype=float),
            tmin=tmin,
            deltat=deltat)

    def _save_initial_rug_file(self, path):
        rug.save(
            [self._make_rug_carpet(0., 10, 1.)], path, overwrite=True)

    def test_save_check_append_hook_permit(self):
        # A hook returning True behaves like no hook at all: the
        # append proceeds normally.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.rug')
            self._save_initial_rug_file(path)

            rug.save(
                [self._make_rug_carpet(20., 10, 2.)],
                path,
                append=True,
                check_append_hook=lambda fn: True)

            carpets = list(cp for (_, cp) in rug.iload(path))
            self.assertEqual(len(carpets), 2)

    def test_save_check_append_hook_deny_overwrite(self):
        # A hook returning False vetoes the append; with overwrite
        # allowed, this falls back to truncating the file instead.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.rug')
            self._save_initial_rug_file(path)

            rug.save(
                [self._make_rug_carpet(20., 10, 2.)],
                path,
                append=True,
                overwrite=True,
                check_append_hook=lambda fn: False)

            carpets = list(cp for (_, cp) in rug.iload(path))
            self.assertEqual(len(carpets), 1)
            self.assertEqual(carpets[0].tmin, 20.)

    def test_save_check_append_hook_deny_no_overwrite(self):
        # A hook returning False vetoes the append; with overwrite
        # also forbidden, saving must fail outright.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.rug')
            self._save_initial_rug_file(path)

            with self.assertRaises(FileSaveError):
                rug.save(
                    [self._make_rug_carpet(20., 10, 2.)],
                    path,
                    append=True,
                    overwrite=False,
                    check_append_hook=lambda fn: False)

    def test_save_check_append_merge(self):
        # Overlapping carpets are rejected on append by default...
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.rug')
            self._save_initial_rug_file(path)

            with self.assertRaises(FileSaveError):
                rug.save(
                    [self._make_rug_carpet(5., 10, 2.)],
                    path,
                    append=True,
                    check_append=True)

            # ...but merge in cleanly, with the new data taking
            # precedence over the overlapping part of the old, when
            # check_append_merge is enabled.
            rug.save(
                [self._make_rug_carpet(5., 10, 2.)],
                path,
                append=True,
                check_append=True,
                check_append_merge=True)

            carpets = list(cp for (_, cp) in rug.iload(path))
            self.assertEqual(len(carpets), 1)
            cp = carpets[0]
            self.assertEqual(cp.tmin, 0.)
            self.assertEqual(cp.nsamples, 15)
            self.assertTrue(num.all(cp.data[0, :5] == 1.))
            self.assertTrue(num.all(cp.data[0, 5:] == 2.))
