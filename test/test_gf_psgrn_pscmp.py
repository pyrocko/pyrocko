import math
import unittest
import logging
from tempfile import mkdtemp
import numpy as num
import os

from pyrocko import orthodrome as ortd
from pyrocko import util, gf, cake  # noqa
from pyrocko.fomosto import psgrn_pscmp
from pyrocko.guts import Object, Float, List, String
from pyrocko.guts_array import Array

logger = logging.getLogger('test_gf_psgrn_pscmp')

km = 1000.

r2d = 180. / math.pi
d2r = 1.0 / r2d
km = 1000.


class StaticTarget(Object):
    lat = Float.T()
    lon = Float.T()
    north_shifts = Array.T(shape=(None,), dtype=num.float)
    east_shifts = Array.T(shape=(None,), dtype=num.float)
    depth = Float.T(default=0.0)
    components = List.T(String.T())
    store_id = gf.StringID.T()
    interpolation = gf.InterpolationMethod.T(
        default='nearest_neighbor',
        help='interpolation method to use')

    optimization = gf.OptimizationMethod.T(
        default='enable',
        optional=True,
        help='disable/enable optimizations in weight-delay-and-sum operation')


def statics(engine, source, starget):
    store = engine.get_store(starget.store_id)
    dsource = source.discretize_basesource(store, starget)

    assert len(starget.north_shifts) == len(starget.east_shifts)
    out = num.zeros((len(starget.north_shifts), len(starget.components)))
    sfactor = source.get_factor()
    for i, (north, east) in enumerate(
            zip(starget.north_shifts, starget.east_shifts)):

        receiver = gf.Receiver(
            lat=starget.lat,
            lon=starget.lon,
            north_shift=north,
            east_shift=east,
            depth=starget.depth)

        values = store.statics(
            dsource, receiver, starget.components,
            interpolation=starget.interpolation,
            optimization=starget.optimization)

        for icomponent, value in enumerate(values):
            out[i, icomponent] = value * sfactor

    return out


class GFPsgrnPscmpTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.tempdirs = []

    def __del__(self):
        import shutil

        for d in self.tempdirs:
            shutil.rmtree(d)

    def test_fomosto_vs_psgrn_pscmp(self):

        mod = cake.LayeredModel.from_scanlines(cake.read_nd_model_str('''
 0. 5.8 3.46 2.6 1264. 600.
 20. 5.8 3.46 2.6 1264. 600.
 20. 6.5 3.85 2.9 1283. 600.
 35. 6.5 3.85 2.9 1283. 600.
mantle
 35. 8.04 4.48 3.58 1449. 600.
 77.5 8.045 4.49 3.5 1445. 600.
 77.5 8.045 4.49 3.5 180.6 75.
 120. 8.05 4.5 3.427 180. 75.
 120. 8.05 4.5 3.427 182.6 76.06
 165. 8.175 4.509 3.371 188.7 76.55
 210. 8.301 4.518 3.324 201. 79.4
 210. 8.3 4.52 3.321 336.9 133.3
 410. 9.03 4.871 3.504 376.5 146.1
 410. 9.36 5.08 3.929 414.1 162.7
 660. 10.2 5.611 3.918 428.5 172.9
 660. 10.79 5.965 4.229 1349. 549.6'''.lstrip()))

        store_dir = mkdtemp(prefix='gfstore')
        self.tempdirs.append(store_dir)
        store_id = 'psgrn_pscmp_test'

        version = '2008a'

        c = psgrn_pscmp.PsGrnPsCmpConfig()
        c.psgrn_config.sampling_interval = 1.
        c.psgrn_config.version = version
        c.pscmp_config.version = version

        config = gf.meta.ConfigTypeA(
            id=store_id,
            ncomponents=10,
            sample_rate=1. / c.pscmp_config.snapshots.deltat,
            receiver_depth=0. * km,
            source_depth_min=0. * km,
            source_depth_max=5. * km,
            source_depth_delta=0.1 * km,
            distance_min=0. * km,
            distance_max=40. * km,
            distance_delta=0.1 * km,
            modelling_code_id='psgrn_pscmp.%s' % version,
            earthmodel_1d=mod,
            tabulated_phases=[])

        config.validate()
        gf.store.Store.create_editables(
            store_dir, config=config, extra={'psgrn_pscmp': c})

        store = gf.store.Store(store_dir, 'r')
        store.close()

        # build store
        try:
            psgrn_pscmp.build(store_dir, nworkers=1)
        except psgrn_pscmp.PsCmpError, e:
            if str(e).find('could not start psgrn/pscmp') != -1:
                logger.warn('psgrn/pscmp not installed; '
                            'skipping test_pyrocko_gf_vs_pscmp')
                return
            else:
                raise

        origin = gf.Location(
            lat=10.,
            lon=-15.)

        # test GF store
        TestRF = dict(
            lat=origin.lat, lon=origin.lon,
            depth=2. * km,
            width=2. * km,
            length=5. * km,
            rake=90., dip=45., strike=45.,
            slip=1.,
                    )

        source = gf.RectangularSource(**TestRF)

        nnorth = 40
        neast = 40

        norths = num.linspace(-20., 20., nnorth) * km
        easts = num.linspace(-20., 20., neast) * km
        norths2 = num.repeat(norths, len(easts))
        easts2 = num.tile(easts, len(norths))

        engine = gf.LocalEngine(store_dirs=[store_dir])

        starget = StaticTarget(
            lat=origin.lat,
            lon=origin.lon,
            east_shifts=easts2,
            north_shifts=norths2,
            store_id=store_id,
            components=['displacement.d'],
            optimization='enable',
            interpolation='multilinear')

        uz = statics(engine, source, starget)[:, 0]

        # test against direct pscmp output
        lats2, lons2 = ortd.ne_to_latlon(
            origin.lat, origin.lon, norths2, easts2)
        pscmp_sources = [psgrn_pscmp.PsCmpRectangularSource(**TestRF)]

        cc = c.pscmp_config
        cc.observation = psgrn_pscmp.PsCmpScatter(lats=lats2, lons=lons2)
        cc.rectangular_source_patches = pscmp_sources

        ccf = psgrn_pscmp.PsCmpConfigFull(**cc.items())
        ccf.psgrn_outdir = os.path.join(store_dir, c.gf_outdir) + '/'

        runner = psgrn_pscmp.PsCmpRunner(keep_tmp=False)
        runner.run(ccf)
        ps2du = runner.get_results(component='displ')[0]

        uz_ps2d = ps2du[:, 2]

        uz = -uz.reshape((nnorth, neast))
        uz2d = -uz_ps2d.reshape((nnorth, neast))

        num.testing.assert_allclose(uz, uz2d, atol=0.001)

        # plotting

#        uz_min = num.min(uz)
#        uz_max = num.max(uz)
#        uz2d_min = num.min(uz2d)
#        uz2d_max = num.max(uz2d)

#        uz_absmax = max(abs(uz_min), abs(uz_max))
#        uz2d_absmax = max(abs(uz2d_min), abs(uz2d_max))

#        levels = num.linspace(-uz_absmax, uz_absmax, 21)
#        levels2d = num.linspace(-uz2d_absmax, uz2d_absmax, 21)

#        from matplotlib import pyplot as plt

#        fontsize = 10.
#        plot.mpl_init(fontsize=fontsize)

#        cmap = plt.cm.get_cmap('coolwarm')
#        fig = plt.figure(figsize=plot.mpl_papersize('a4', 'landscape'))
#        plot.mpl_margins(fig, w=14., h=6., units=fontsize)

#        axes1 = fig.add_subplot(1, 2, 1, aspect=1.0)
#        cs1 = axes1.contourf(
#            easts / km, norths / km, uz, levels=levels, cmap=cmap)
#        plt.colorbar(cs1)

#        axes2 = fig.add_subplot(1, 2, 2, aspect=1.0)
#        cs2 = axes2.contourf(
#            easts / km, norths / km, uz2d, levels=levels2d, cmap=cmap)
#        plt.colorbar(cs2)

#        axes1.set_xlabel('Easting [km]')
#        axes1.set_ylabel('Northing [km]')

#        fig.savefig('staticGFvs2d_Afmu_diff.pdf')


if __name__ == '__main__':
    util.setup_logging('test_gf_psgrn_pscmp', 'warning')
    unittest.main()
