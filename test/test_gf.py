import sys
import random
import math
from pyrocko import guts
import unittest
from tempfile import mkdtemp
import numpy as num

from pyrocko import gf, util, cake

r2d = 180. / math.pi
d2r = 1.0 / r2d
km = 1000.


def numeq(a, b, eps):
    return (num.all(num.asarray(a).shape == num.asarray(b).shape and
            num.abs(num.asarray(a) - num.asarray(b)) < eps))


class PulseConfig(guts.Object):
    fwhm = guts.Float.T(default=0.02)
    velocity = guts.Float.T(default=1000.)

    def evaluate(self, distance, t):
        tarr = distance / self.velocity
        denom = self.fwhm / (2.*math.sqrt(math.log(2.)))
        return num.exp(-((t-tarr)/denom)**2)


class GFTestCase(unittest.TestCase):

    if sys.version_info < (2, 7):
        from contextlib import contextmanager

        @contextmanager
        def assertRaises(self, exc):

            gotit = False
            try:
                yield None
            except exc:
                gotit = True

            assert gotit, 'expected to get a %s exception' % exc

        def assertIsNone(self, value):
            assert value is None, 'expected None but got %s' % value

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.tempdirs = []
        self.pulse_store_dir = None
        self.regional_ttt_store_dir = None

    def __del__(self):
        import shutil

        for d in self.tempdirs:
            shutil.rmtree(d)

    def create(self, deltat=1.0, nrecords=10):
        d = mkdtemp(prefix='gfstore')
        store = gf.BaseStore.create(d, deltat, nrecords, force=True)

        store = gf.BaseStore(d, mode='w')
        for i in range(nrecords):
            data = num.asarray(num.random.random(random.randint(0, 7)),
                               dtype=gf.gf_dtype)

            tr = gf.GFTrace(data=data, itmin=1+i)
            store.put(i, tr)

        store.close()
        self.tempdirs.append(d)
        return d

    def get_pulse_store_dir(self):
        if self.pulse_store_dir is None:
            self.pulse_store_dir = self._create_pulse_store()

        return self.pulse_store_dir

    def get_regional_ttt_store_dir(self):
        if self.regional_ttt_store_dir is None:
            self.regional_ttt_store_dir = self._create_regional_ttt_store()

        return self.regional_ttt_store_dir

    def _create_regional_ttt_store(self):

        conf = gf.ConfigTypeA(
            id='empty_regional',
            source_depth_min=0.,
            source_depth_max=20*km,
            source_depth_delta=10*km,
            distance_min=1000*km,
            distance_max=2000*km,
            distance_delta=10*km,
            sample_rate=2.0,
            ncomponents=10,
            earthmodel_1d=cake.load_model(),
            tabulated_phases=[
                gf.TPDef(id=id, definition=defi) for (id, defi) in [
                    ('depthp', 'p'),
                    ('pS', 'pS'),
                    ('P', 'P'),
                    ('S', 'S')
                ]
            ])

        store_dir = mkdtemp(prefix='gfstore')
        self.tempdirs.append(store_dir)

        gf.Store.create(store_dir, config=conf)
        store = gf.Store(store_dir)
        store.make_ttt()

        store.close()
        return store_dir

    def _create_pulse_store(self):

        conf = gf.ConfigTypeB(
            id='pulse',
            receiver_depth_min=0.,
            receiver_depth_max=10.,
            receiver_depth_delta=10.,
            source_depth_min=0.,
            source_depth_max=1000.,
            source_depth_delta=10.,
            distance_min=10.,
            distance_max=1000.,
            distance_delta=10.,
            sample_rate=200.,
            ncomponents=2.,
            component_scheme='elastic2')

        pulse = PulseConfig()

        #fnyq_spatial = pulse.velocity / math.sqrt(conf.distance_delta**2 +
        #                                       conf.source_depth_delta**2)

        store_dir = mkdtemp(prefix='gfstore')
        self.tempdirs.append(store_dir)

        gf.Store.create(store_dir, config=conf, force=True,
                        extra={'pulse': pulse})

        deltat = conf.deltat

        store = gf.Store(store_dir, mode='w')
        for args in store.config.iter_nodes(level=-1):

            rdepth, sdepth, surfdist = args
            dist = math.sqrt((rdepth-sdepth)**2 + surfdist**2)

            tarr = dist / pulse.velocity

            tmin = tarr - 5 * pulse.fwhm
            tmax = tarr + 5 * pulse.fwhm
            itmin = int(num.floor(tmin/deltat))
            itmax = int(num.ceil(tmax/deltat))
            tmin = itmin * deltat
            tmax = itmax * deltat
            nsamples = itmax - itmin + 1

            t = tmin + num.arange(nsamples) * deltat

            data = pulse.evaluate(dist, t)

            phi = math.atan2(rdepth-sdepth, surfdist)
            data = [data*math.cos(phi), data*math.sin(phi)]
            for icomponent, data in enumerate(data):
                is_zero = num.all(data == 0.0)

                tr = gf.GFTrace(data=data, itmin=itmin, deltat=deltat,
                                is_zero=is_zero)

                store.put(args + (icomponent,), tr)

        store.close()
        return store_dir

    def test_get_spans(self):
        nrecords = 8
        random.seed(0)
        num.random.seed(0)

        store = gf.BaseStore(self.create(nrecords=nrecords))
        for deci in (1, 2, 3, 4):
            for i in range(nrecords):
                tr = store.get(i, decimate=deci)
                itmin, itmax = store.get_span(i, decimate=deci)
                self.assertEqual(tr.itmin, itmin)
                self.assertEqual(tr.data.size, itmax-itmin + 1)

        store.close()

    def test_partial_get(self):
        nrecords = 8
        random.seed(0)
        num.random.seed(0)

        store = gf.BaseStore(self.create(nrecords=nrecords))
        for deci in (1, 2, 3, 4):
            for i in range(0, nrecords):
                tr = store.get(i, decimate=deci)
                itmin_gf, nsamples_gf = tr.itmin, tr.data.size
                for itmin in range(itmin_gf - nsamples_gf,
                                   itmin_gf + nsamples_gf+1):

                    for nsamples in range(0, nsamples_gf*3):
                        tr2 = store.get(i, itmin, nsamples, decimate=deci)
                        self.assertEqual(tr2.itmin, max(tr.itmin, itmin))
                        self.assertEqual(tr2.itmin + tr2.data.size,
                                         max(min(tr.itmin + tr.data.size,
                                                 itmin + nsamples),
                                             tr2.itmin + tr2.data.size))

                        ilo = max(tr.itmin, tr2.itmin)
                        ihi = min(tr.itmin+tr.data.size,
                                  tr2.itmin+tr2.data.size)

                        a = tr.data[ilo-tr.itmin:ihi-tr.itmin]
                        b = tr2.data[ilo-tr2.itmin:ihi-tr2.itmin]

                        self.assertTrue(numeq(a, b, 0.001))

        store.close()

    def test_sum(self):

        nrecords = 8
        random.seed(0)
        num.random.seed(0)

        store = gf.BaseStore(self.create(nrecords=nrecords))

        for deci in (1, 2, 3, 4):
            for i in range(300):
                n = random.randint(0, 5)
                indices = num.random.randint(nrecords, size=n)
                weights = num.random.random(n)
                shifts = num.random.random(n)*nrecords
                shifts[::2] = num.round(shifts[::2])

                for itmin, nsamples in [(None, None),
                                        (random.randint(0, nrecords),
                                         random.randint(0, nrecords))]:

                        a = store.sum(indices, shifts, weights,
                                      itmin=itmin,
                                      nsamples=nsamples,
                                      decimate=deci)

                        b = store.sum_reference(indices, shifts, weights,
                                                itmin=itmin,
                                                nsamples=nsamples,
                                                decimate=deci)

                        self.assertEqual(a.itmin, b.itmin)
                        self.assertTrue(numeq(a.data, b.data, 0.01))

        store.close()

    def test_pulse(self):
        store_dir = self.get_pulse_store_dir()

        engine = gf.LocalEngine(store_dirs=[store_dir])

        sources = [
            gf.ExplosionSource(
                time=0.0,
                depth=depth,
                moment=moment)

            for moment in (1.0, 2.0, 3.0) for depth in [100., 200., 300.]
        ]

        targets = [
            gf.Target(
                codes=('', 'STA', '', component),
                north_shift=500.,
                east_shift=0.)

            for component in 'ZNE'
        ]

        pulse = engine.get_store_extra(None, 'pulse')

        response = engine.process(sources=sources, targets=targets)
        for source, target, tr in response.iter_results():
            t = tr.get_xdata()

            dist = math.sqrt((source.depth - target.depth)**2 +
                             source.distance_to(target)**2)

            data = pulse.evaluate(dist, t-source.time)

            phi = math.atan2((source.depth - target.depth),
                             source.distance_to(target)) * r2d

            azi, bazi = source.azibazi_to(target)

            data *= source.moment

            if tr.channel.endswith('N'):
                data *= math.cos(phi*d2r) * math.cos(azi*d2r)
            elif tr.channel.endswith('E'):
                data *= math.cos(phi*d2r) * math.sin(azi*d2r)
            elif tr.channel.endswith('Z'):
                data *= math.sin(phi*d2r)

            tr2 = tr.copy(data=False)
            tr2.set_ydata(data)
            tr2.set_codes(location='X')

            self.assertTrue(numeq(data, tr.ydata, 0.01))

    def test_timing(self):
        store_dir = self.get_regional_ttt_store_dir()

        store = gf.Store(store_dir)

        args = (10*km, 1500*km)
        assert(store.t('P', args) is not None)
        self.assertEqual(store.t('last(S|P)', args), store.t('S', args))
        self.assertEqual(store.t('(S|P)', args), store.t('S', args))
        self.assertEqual(store.t('(P|S)', args), store.t('P', args))
        self.assertEqual(store.t('first(S|P)', args), store.t('P', args))
        with self.assertRaises(gf.NoSuchPhase):
            store.t('nonexistant', args)

        with self.assertRaises(AssertionError):
            store.t('P', (10*km,))

        with self.assertRaises(gf.OutOfBounds):
            print store.t('P', (10*km, 5000*km))

        with self.assertRaises(gf.OutOfBounds):
            print store.t('P', (30*km, 1500*km))


if __name__ == '__main__':
    util.setup_logging('test_gf', 'warning')
    unittest.main()
