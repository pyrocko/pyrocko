
import unittest
import logging
import os

import numpy as num
from pyrocko import gf, util, trace, moment_tensor as pmt, orthodrome as pod
from pyrocko.gf.seismosizer import map_anchor
from pyrocko.modelling import make_okada_coefficient_matrix

logger = logging.getLogger('pyrocko.test.test_gf_source_types')

km = 1e3
r2d = 180. / num.pi
d2r = num.pi / 180.

show_plot = int(os.environ.get('MPL_SHOW', 0))


def have_store(store_id):
    engine = gf.get_engine()
    try:
        engine.get_store(store_id)
        return True, ''
    except gf.NoSuchStore:
        return False, 'GF store "%s" not available' % store_id


class GFSourceTypesTestCase(unittest.TestCase):

    def test_rectangular_source(self):
        # WIP
        nsrc = 5
        rect_sources = []

        for n in range(nsrc):
            src = gf.RectangularSource(
                lat=0., lon=0.,
                anchor='bottom',
                north_shift=5000., east_shift=9000., depth=4.*km,
                width=2.*km, length=8.*km,
                dip=0., rake=0., strike=(180./nsrc + 1) * n,
                slip=1.)
            rect_sources.append(src)

    @staticmethod
    def plot_rectangular_source(src, store):
        from matplotlib import pyplot as plt
        from matplotlib.patches import Polygon
        ax = plt.gca()
        ne = src.outline(cs='xy')
        p = Polygon(num.fliplr(ne), fill=False, color='r', alpha=.7)
        ax.add_artist(p)

        mt = src.discretize_basesource(store)
        ax.scatter(mt.east_shifts, mt.north_shifts, alpha=1)
        ax.scatter(src.east_shift, src.north_shift, color='r')

        plt.axis('equal')
        plt.show()

    @unittest.skipUnless(*have_store('iceland_reg_v2'))
    def test_pseudo_dynamic_rupture(self):
        from matplotlib import pyplot as plt

        store_id = 'iceland_reg_v2'

        engine = gf.get_engine()
        store = engine.get_store(store_id)

        moment = 1e19
        rake = 0.
        nucleation_x, nucleation_y = 0., 0.

        pdr = gf.PseudoDynamicRupture(
            length=20000., width=10000., depth=2000.,
            anchor='top', gamma=0.8, dip=45., strike=60.,
            slip=1., rake=rake,
            nx=5, ny=3, smooth_rupture=False,
            decimation_factor=1000)

        # Check magnitude calculations
        pdr.rescale_slip(
            magnitude=pmt.moment_to_magnitude(moment), store=store)
        assert pdr.get_magnitude(store) == pmt.moment_to_magnitude(moment)

        # Check magnitude scaling based on tractions
        pdr.slip = None
        pdr.rake = None
        pdr.tractions = gf.tractions.DirectedTractions(traction=1., rake=rake)
        pdr.tractions.traction *= moment / pdr.get_moment(store)

        assert pdr.get_magnitude(store) == pmt.moment_to_magnitude(moment)

        # Check nucleation setting
        pdr.nucleation = num.array([[nucleation_x, nucleation_y]])
        assert pdr.nucleation_x == num.array(nucleation_x)
        assert pdr.nucleation_y == num.array(nucleation_y)
        num.testing.assert_equal(
            pdr.nucleation, num.array([[nucleation_x, nucleation_y]]))

        # Check new nucleation time
        pdr.nucleation_time = 1.
        points_old, _, vr_old, times_old = pdr.discretize_time(store)
        assert times_old.shape == vr_old.shape
        assert points_old.shape[0] == times_old.shape[0] * times_old.shape[1]

        pdr._interpolators = {}
        pdr.nucleation_time = 0.

        points_new, _, vr_new, times_new = pdr.discretize_time(store)
        assert times_new.shape == times_old.shape
        num.testing.assert_allclose(times_new, times_old - 1., rtol=1e-7)
        num.testing.assert_allclose(vr_new, vr_old, rtol=1e-7)
        num.testing.assert_allclose(points_new, points_old, rtol=1e-7)

        pdr.discretize_patches(
            store=store,
            interpolation='nearest_neighbor')

        # Check moment calculation discretize basesource vs moment_rate_patches
        mom_rate_old, times_old = pdr.get_moment_rate_patches(store=store)
        mom_rate_new, times_new = pdr.get_moment_rate(store=store)

        cum_mom_old = (mom_rate_old * num.concatenate([
            (num.diff(times_old)[0],), num.diff(times_old)])).sum()

        num.testing.assert_allclose(cum_mom_old, moment, rtol=1.5e-1)
        num.testing.assert_equal(times_new, times_old)

        # Check magnitude scaling of slip and slip rate
        disloc_tmax = pdr.get_slip()
        disloc_tmax_max = num.linalg.norm(disloc_tmax, axis=1).max()

        # pdr.magnitude = None
        pdr.slip = disloc_tmax_max

        # Large rtol due to interpolation and rounding differences induced by
        # get_moment calculation
        num.testing.assert_allclose(
            pdr.get_moment(store=store),
            moment,
            rtol=5e-2)

        deltat = pdr.get_patch_attribute('time').max() * 0.5
        deltaslip, times = pdr.get_delta_slip(deltat=deltat, delta=False)

        num.testing.assert_allclose(disloc_tmax, pdr.get_slip())
        num.testing.assert_allclose(disloc_tmax, deltaslip[:, -1, :])

        pdr.slip = None
        pdr.rake = None

        pdr.tractions = gf.tractions.HomogeneousTractions(
            strike=0.,
            dip=0.,
            normal=-.5e6)

        disloc_est = pdr.get_slip(t=deltat)

        # Check get delta slip function
        deltaslip, times_old = pdr.get_delta_slip(deltat=deltat)
        cumslip, times_new = pdr.get_delta_slip(deltat=deltat, delta=False)

        num.testing.assert_equal(times_old, times_new)
        num.testing.assert_allclose(
            cumslip, num.cumsum(deltaslip, axis=1), rtol=1e-10)

        # Check coefficient matrix calculation
        coef_mat = make_okada_coefficient_matrix(pdr.patches)
        assert (coef_mat == pdr.coef_mat).all()

        if show_plot:
            level = num.arange(0., 15., 1.5)

            x_val = points_old[:times_old.shape[1], 0]
            y_val = points_old[::times_old.shape[1], 2]

            plt.gcf().add_subplot(1, 1, 1, aspect=1.0)
            plt.imshow(
                vr_old,
                extent=[
                    num.min(x_val), num.max(x_val),
                    num.max(y_val), num.min(y_val)])
            plt.contourf(
                x_val, y_val, times_old, level, cmap='gray', alpha=0.7)
            plt.colorbar(label='Rupture Propagation Time [s]')
            plt.show()

            x_val = num.array([
                src.northing for src in pdr.patches])[:pdr.nx]
            y_val = num.array([
                src.depth for src in pdr.patches])[::pdr.nx]

            plt.gcf().add_subplot(1, 1, 1, aspect=1.0)
            im = plt.imshow(
                disloc_est[2::3].reshape(y_val.shape[0], x_val.shape[0]),
                extent=[
                    num.min(x_val), num.max(x_val),
                    num.max(y_val), num.min(y_val)])
            plt.contourf(
                x_val, y_val,
                pdr.get_patch_attribute('time').reshape(pdr.ny, pdr.nx),
                level,
                cmap='gray', alpha=0.5)
            plt.colorbar(im, label='Opening [m] after %.2f s' % deltat)
            plt.show()

    @unittest.skipUnless(*have_store('iceland_reg_v2'))
    def test_pseudo_dynamic_rupture_cfs(self):
        from pyrocko.modelling import okada_ext
        store_id = 'iceland_reg_v2'

        engine = gf.get_engine()
        store = engine.get_store(store_id)

        source = gf.PseudoDynamicRupture(
            length=20000., width=10000., depth=2000.,
            anchor='top', gamma=0.8, dip=45., strike=60.,
            slip=1., rake=0.,
            nx=5, ny=3, smooth_rupture=False,
            decimation_factor=1000)

        source.discretize_patches(store)

        target_strike = -92.
        target_dip = 73.
        target_rake = -8.

        nnorths = 50
        neasts = 50
        norths = num.linspace(-50., 50., nnorths) * 1e3
        easts = num.linspace(-50., 50., neasts) * 1e3
        depth_target = 10e3

        receiver_points = num.zeros((nnorths * neasts, 3))
        receiver_points[:, 0] = num.repeat(norths, neasts)
        receiver_points[:, 1] = num.tile(easts, nnorths)
        receiver_points[:, 2] = num.ones(nnorths * neasts) * depth_target

        friction = 0.6
        pressure = 0.

        cfs_init = source.get_coulomb_failure_stress(
            receiver_points, friction=friction, pressure=pressure,
            strike=target_strike, dip=target_dip, rake=target_rake,
            nthreads=1)

        # Calculate CFS from scratch
        source_slip = source.get_slip(time=None, scale_slip=True)

        # source planes
        source_patches = num.array([
            src.source_patch() for src in source.patches])

        # earth model
        lambda_mean = num.mean([src.lamb for src in source.patches])
        mu_mean = num.mean([src.shearmod for src in source.patches])

        # Dislocation and spatial derivatives from okada in NED
        results = okada_ext.okada(
            source_patches,
            source_slip,
            receiver_points,
            lambda_mean,
            mu_mean,
            rotate_sdn=False,
            stack_sources=0)

        # resolve stress tensor (sum!)
        diag_ind = [0, 4, 8]
        kron = num.zeros(9)
        kron[diag_ind] = 1.
        kron = kron[num.newaxis, num.newaxis, :]

        eps = 0.5 * (
            results[:, :, 3:] +
            results[:, :, (3, 6, 9, 4, 7, 10, 5, 8, 11)])

        dilatation \
            = eps[:, :, diag_ind].sum(axis=-1)[:, :, num.newaxis]

        stress = kron*lambda_mean*dilatation + 2.*mu_mean*eps
        # stress shape (n_sources, n_receivers, n_stress_components)

        # superposed stress of all sources at receiver locations
        stress_sum = num.sum(stress, axis=0)
        # stress_sum shape: (n_receivers, n_stress_components)

        # get shear and normal stress from stress tensor
        st0 = d2r * target_strike
        di0 = d2r * target_dip
        ra0 = d2r * target_rake

        n_rec = receiver_points.shape[0]
        stress_normal = num.zeros(n_rec)
        tau = num.zeros(n_rec)

        for irec in range(n_rec):
            ns = num.zeros(3)
            rst = num.zeros(3)
            rdi = num.zeros(3)

            ns[0] = num.sin(di0) * num.cos(st0 + 0.5 * num.pi)
            ns[1] = num.sin(di0) * num.sin(st0 + 0.5 * num.pi)
            ns[2] = -num.cos(di0)

            rst[0] = num.cos(st0)
            rst[1] = num.sin(st0)
            rst[2] = 0.0

            rdi[0] = num.cos(di0) * num.cos(st0 + 0.5 * num.pi)
            rdi[1] = num.cos(di0) * num.sin(st0 + 0.5 * num.pi)
            rdi[2] = num.sin(di0)

            ts = rst * num.cos(ra0) - rdi * num.sin(ra0)

            for j in range(3):
                for i in range(3):
                    stress_normal[irec] += \
                        ns[i] * stress_sum[irec, j*3 + i] * ns[j]
                    tau[irec] += ts[i] * stress_sum[irec, j*3 + i] * ns[j]

        # calculate cfs using formula above and return
        cfs_comp = tau + friction * (stress_normal + pressure)

        num.testing.assert_allclose(cfs_init, cfs_comp, atol=1.)

    def test_pseudo_dynamic_rupture_outline(self):
        length = 20000.
        width = 10000.
        depth = 3400.
        north_shift = 1500.
        east_shift = -2000.

        pdr = gf.PseudoDynamicRupture(
            length=length, width=width, depth=depth,
            north_shift=north_shift, east_shift=east_shift,
            anchor='top', gamma=0.8, dip=90., strike=0.)

        points = num.array(
            [[-0.5 * length + north_shift, east_shift, depth],
             [0.5 * length + north_shift, east_shift, depth],
             [0.5 * length + north_shift, east_shift, depth + width],
             [-0.5 * length + north_shift, east_shift, depth + width],
             [-0.5 * length + north_shift, east_shift, depth]])

        num.testing.assert_allclose(
            points, pdr.outline(cs='xyz'), atol=1.)

        num.testing.assert_allclose(
            points[:, :2], pdr.outline(cs='xy'), atol=1.)

        latlon = pod.ne_to_latlon(pdr.lat, pdr.lon, points[:, 0], points[:, 1])
        latlon = num.array(latlon).T

        num.testing.assert_allclose(
            latlon,
            pdr.outline(cs='latlon'), atol=1.)

        num.testing.assert_allclose(
            latlon[:, ::-1],
            pdr.outline(cs='lonlat'), atol=1.)

        num.testing.assert_allclose(
            num.concatenate((latlon, points[:, 2].reshape((-1, 1))), axis=1),
            pdr.outline(cs='latlondepth'), atol=1.)

    def test_pseudo_dynamic_rupture_points_on_source(self):
        length = 20000.
        width = 10000.
        depth = 3400.
        north_shift = 1500.
        east_shift = -2000.

        pdr = gf.PseudoDynamicRupture(
            length=length, width=width, depth=depth,
            north_shift=north_shift, east_shift=east_shift,
            anchor='top', gamma=0.8, dip=90., strike=0.)

        random_points = num.concatenate((
            num.random.uniform(-1, 1, size=(10, 1)),
            num.random.uniform(-1., 1., size=(10, 1))), axis=1)

        points_xy = num.array(list(map_anchor.values()))
        points_xy = num.concatenate((points_xy, random_points))

        points = num.zeros((points_xy.shape[0], 3))
        points[:, 0] = 0.5 * points_xy[:, 0] * length + north_shift
        points[:, 1] = east_shift
        points[:, 2] = 0.5 * (points_xy[:, 1] + 1.) * width + depth

        num.testing.assert_allclose(
            points, pdr.points_on_source(
                points_x=points_xy[:, 0], points_y=points_xy[:, 1],
                cs='xyz'), atol=1.)

        num.testing.assert_allclose(
            points[:, :2], pdr.points_on_source(
                points_x=points_xy[:, 0], points_y=points_xy[:, 1],
                cs='xy'), atol=1.)

        latlon = pod.ne_to_latlon(pdr.lat, pdr.lon, points[:, 0], points[:, 1])
        latlon = num.array(latlon).T

        num.testing.assert_allclose(
            latlon,
            pdr.points_on_source(
                points_x=points_xy[:, 0], points_y=points_xy[:, 1],
                cs='latlon'), atol=1.)

        num.testing.assert_allclose(
            latlon[:, ::-1],
            pdr.points_on_source(
                points_x=points_xy[:, 0], points_y=points_xy[:, 1],
                cs='lonlat'), atol=1.)

        num.testing.assert_allclose(
            num.concatenate((latlon, points[:, 2].reshape((-1, 1))), axis=1),
            pdr.points_on_source(
                points_x=points_xy[:, 0], points_y=points_xy[:, 1],
                cs='latlondepth'), atol=1.)

    @unittest.skipUnless(*have_store('iceland_reg_v2'))
    def test_oversampling(self):
        store_id = 'iceland_reg_v2'

        comp_map = {
            'E': 0,
            'N': 1,
            'Z': 2
        }

        nazimuths = 12
        azimuths = num.linspace(0, num.pi*2, nazimuths)
        dist = 15*km

        coords = num.zeros((2, nazimuths))
        coords[0, :] = num.cos(azimuths)
        coords[1, :] = num.sin(azimuths)
        coords *= dist

        targets = [
            gf.Target(
                north_shift=coords[0, iazi],
                east_shift=coords[1, iazi],
                interpolation='multilinear',

                store_id=store_id,
                codes=('', 'S%01d' % iazi, '', channel))
            for iazi in range(nazimuths)
            for channel in comp_map.keys()]

        for t, azi in zip(targets, azimuths):
            t.azimuth = azi * r2d

        nsizes = 20
        length = num.linspace(10, 20, nsizes)
        width = num.linspace(5, 10, nsizes)

        fault_sizes = length * width * km**2

        engine = gf.get_engine()

        misfit_setup = trace.MisfitSetup(
            norm=2,
            taper=trace.CosFader(xfade=.5))

        mf = num.zeros((nazimuths*3, nsizes, 2))

        all_traces = []
        for isize in range(nsizes):
            source = gf.RectangularSource(
                depth=1.5*km,
                strike=0.,
                dip=90.,
                anchor='top',

                length=length[isize]*km,
                width=width[isize]*km,
                slip=2.)

            source_os = source.clone(aggressive_oversampling=True)

            resp = engine.process(source, targets, nthreads=6)
            resp_os = engine.process(source_os, targets, nthreads=6)

            traces = resp.pyrocko_traces()
            traces_os = resp_os.pyrocko_traces()

            for itr, (tr, tr_os) in enumerate(zip(traces, traces_os)):
                assert tr.channel == tr_os.channel
                assert tr.station == tr_os.station
                tr.set_location(str(isize))
                tr_os.set_location(str(isize))
                misfit, norm = tr.misfit(tr_os, misfit_setup)

                assert misfit <= .1
                mf[itr, isize, :] = (misfit, norm)

            all_traces += traces
            all_traces += traces_os

        if show_plot:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import FuncFormatter
            from pyrocko import io

            logger.info('saving oversampling data to /tmp/test_os.mseed')
            io.save(all_traces, '/tmp/test_os.mseed')

            ax = plt.gca()

            for itr in range(mf.shape[0]):
                ax.plot(fault_sizes, mf[itr, :, 0], label='%d' % itr)

            ax.set_title(
                'Misfit aggressive oversampling\n'
                '%d azimuths at %d km distance, ENZ components' %
                (nazimuths, dist/km))
            ax.set_ylim(0, 1)
            ax.grid(alpha=.3)
            ax.set_ylabel('Misfit [time_domain | Lp = 2]')
            ax.set_xlabel('Fault area [km**2]')
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, v: x / km**2))

            plt.show()


if __name__ == '__main__':
    util.setup_logging('test_gf_source_types', 'warning')
    unittest.main(defaultTest='GFSourceTypesTestCase')
