from __future__ import division, print_function, absolute_import
from builtins import range

import unittest
import logging
import random
import time
import math
import multiprocessing
import os.path as op
from pyrocko import gf, util, model
from pyrocko import moment_tensor as pmt
from pyrocko import orthodrome as od

logger = logging.getLogger('pyrocko.test.test_gf_scenarios')
km = 1000.
d2r = math.pi/180.

transchan = {
    'e': 'E',
    'n': 'N',
    'u': 'Z'}


def rand(mi, ma):
    mi = float(mi)
    ma = float(ma)
    return random.random()*(ma-mi) + mi


def to_kiwi_source(source):
    from tunguska import source as kiwi_source

    return kiwi_source.Source(
        time=source.time + source.length / source.velocity / 2.,
        depth=source.depth,
        moment=source.get_moment(),
        strike=source.strike,
        dip=source.dip,
        slip_rake=source.rake,
        length_a=(1. - source.nucleation_x)/2. * source.length,
        length_b=(source.nucleation_x - -1.)/2. * source.length,
        width=source.width,
        rise_time=source.stf.duration)


def have_store(store_id):
    engine = gf.get_engine()
    try:
        engine.get_store(store_id)
        return True
    except gf.NoSuchStore:
        return False


def have_kiwi():
    try:
        import tunguska  # noqa
        return True
    except ImportError:
        return False


class GFScenariosTestCase(unittest.TestCase):
    store_id = 'crust2_mf'
    store_id2 = 'chile_70km_crust'

    @unittest.skipUnless(
            have_store(store_id),
            'GF Store "%s" is not available' % store_id)
    def test_regional(self):
        engine = gf.get_engine()

        nsources = 10
        nstations = 10

        print('cache source channels par wallclock seismograms_per_second')
        nprocs_max = multiprocessing.cpu_count()

        for sourcetype, channels in [
                ['point', 'Z'],
                ['point', 'NEZ'],
                ['rect', 'Z'],
                ['rect', 'NEZ']]:

            for nprocs in [1, 2, 4, 8, 16, 32]:
                if nprocs > nprocs_max:
                    continue

                sources = []
                for isource in range(nsources):
                    m = pmt.MomentTensor.random_dc()
                    strike, dip, rake = map(float, m.both_strike_dip_rake()[0])

                    if sourcetype == 'point':
                        source = gf.DCSource(
                            north_shift=rand(-20.*km, 20*km),
                            east_shift=rand(-20.*km, 20*km),
                            depth=rand(10*km, 20*km),
                            strike=strike,
                            dip=dip,
                            rake=rake,
                            magnitude=rand(4.0, 5.0))

                    elif sourcetype == 'rect':
                        source = gf.RectangularSource(
                            north_shift=rand(-20.*km, 20*km),
                            east_shift=rand(-20.*km, 20*km),
                            depth=rand(10*km, 20*km),
                            length=10*km,
                            width=5*km,
                            nucleation_x=0.,
                            nucleation_y=-1.,
                            strike=strike,
                            dip=dip,
                            rake=rake,
                            magnitude=rand(4.0, 5.0))
                    else:
                        assert False

                    sources.append(source)

                targets = []
                for istation in range(nstations):
                    dist = rand(40.*km, 900*km)
                    azi = rand(-180., 180.)

                    north_shift = dist * math.cos(azi*d2r)
                    east_shift = dist * math.sin(azi*d2r)

                    for cha in channels:
                        target = gf.Target(
                            codes=('', 'S%04i' % istation, '', cha),
                            north_shift=north_shift,
                            east_shift=east_shift,
                            quantity='displacement',
                            interpolation='multilinear',
                            # optimization='disable',
                            store_id=GFScenariosTestCase.store_id)

                        targets.append(target)

                ntraces = len(targets) * len(sources)
                for temperature in ['cold', 'hot']:
                    t0 = time.time()
                    resp = engine.process(sources, targets, nprocs=nprocs)
                    # print resp.stats

                    t1 = time.time()
                    duration = t1 - t0
                    sps = ntraces / duration
                    if temperature == 'hot':
                        if nprocs == 1:
                            sps_ref = sps
                        print('%-5s %-6s %-8s %3i %9.3f %12.1f %12.1f' % (
                            temperature, sourcetype, channels, nprocs, t1-t0,
                            sps, sps/sps_ref))

                    del resp

    @unittest.skipUnless(
            have_store(store_id2),
            'GF Store "%s" is not available' % store_id2)
    @unittest.skipUnless(
            have_kiwi(),
            'KIWI Tools not available')
    def test_against_kiwi(self):
        engine = gf.get_engine()
        store_id = GFScenariosTestCase.store_id2
        try:
            store = engine.get_store(store_id)
        except gf.NoSuchStore:
            logger.warn('GF Store %s not available - skipping test' % store_id)
            return

        base_source = gf.RectangularSource(
            depth=15*km,
            strike=0.,
            dip=90.,
            rake=0.,
            magnitude=4.5,
            nucleation_x=-1.,
            length=10*km,
            width=0*km,
            stf=gf.BoxcarSTF(duration=1.0))

        base_event = base_source.pyrocko_event()

        channels = 'NEZ'
        nstations = 20
        stations = []
        targets = []
        for istation in range(nstations):
            dist = rand(40.*km, 900*km)
            azi = rand(-180., 180.)
            north_shift = dist * math.cos(azi*d2r)
            east_shift = dist * math.sin(azi*d2r)
            lat, lon = od.ne_to_latlon(0., 0., north_shift, east_shift)
            sta = 'S%02i' % istation
            station = model.Station(
                '', sta, '',
                lat=lat,
                lon=lon)

            station.set_channels_by_name('N', 'E', 'Z')
            stations.append(station)

            for cha in channels:
                target = gf.Target(
                    codes=station.nsl() + (cha,),
                    lat=lat,
                    lon=lon,
                    quantity='displacement',
                    interpolation='multilinear',
                    # optimization='disable',
                    store_id=store_id)

                targets.append(target)

        from tunguska import glue

        nsources = 10

        # nprocs_max = multiprocessing.cpu_count()
        nprocs = 1

        try:
            seis = glue.start_seismosizer(
                gfdb_path=op.join(store.store_dir, 'db'),
                event=base_event,
                stations=stations,
                hosts=['localhost']*nprocs,
                balance_method='123321',
                effective_dt=0.5,
                verbose=False)

            ksource = to_kiwi_source(base_source)

            seis.set_source(ksource)

            recs = seis.get_receivers_snapshot(('syn',), (), 'plain')
            trs = []
            for rec in recs:
                for tr in rec.get_traces():
                    tr.set_codes(channel=transchan[tr.channel])
                    trs.append(tr)

            # trs2 = engine.process(base_source, targets).pyrocko_traces()
            engine.process(base_source, targets).pyrocko_traces()

            # trace.snuffle(trs + trs2)

            seis.set_synthetic_reference()

            for sourcetype in ['point', 'rect']:
                sources = []
                for isource in range(nsources):
                    m = pmt.MomentTensor.random_dc()
                    strike, dip, rake = map(float, m.both_strike_dip_rake()[0])

                    if sourcetype == 'point':
                        source = gf.RectangularSource(
                            north_shift=rand(-20.*km, 20*km),
                            east_shift=rand(-20.*km, 20*km),
                            depth=rand(10*km, 20*km),
                            nucleation_x=0.0,
                            nucleation_y=0.0,
                            strike=strike,
                            dip=dip,
                            rake=rake,
                            magnitude=rand(4.0, 5.0),
                            stf=gf.BoxcarSTF(duration=1.0))

                    elif sourcetype == 'rect':
                        source = gf.RectangularSource(
                            north_shift=rand(-20.*km, 20*km),
                            east_shift=rand(-20.*km, 20*km),
                            depth=rand(10*km, 20*km),
                            length=10*km,
                            width=5*km,
                            nucleation_x=-1.,
                            nucleation_y=0,
                            strike=strike,
                            dip=dip,
                            rake=rake,
                            magnitude=rand(4.0, 5.0),
                            stf=gf.BoxcarSTF(duration=1.0))
                    else:
                        assert False

                    sources.append(source)

                for temperature in ['cold', 'hot']:
                    t0 = time.time()
                    resp = engine.process(sources, targets, nprocs=nprocs)
                    t1 = time.time()
                    if temperature == 'hot':
                        dur_pyrocko = t1 - t0

                    del resp

                ksources = map(to_kiwi_source, sources)

                for temperature in ['cold', 'hot']:
                    t0 = time.time()
                    seis.make_misfits_for_sources(
                        ksources, show_progress=False)
                    t1 = time.time()
                    if temperature == 'hot':
                        dur_kiwi = t1 - t0

                print('pyrocko %-5s %5.2fs  %5.1fx' % (
                    sourcetype, dur_pyrocko, 1.0))
                print('kiwi    %-5s %5.2fs  %5.1fx' % (
                    sourcetype, dur_kiwi, dur_pyrocko/dur_kiwi))

        finally:
            seis.close()
            del seis


if __name__ == '__main__':
    util.setup_logging('test_gf_scenarios', 'warning')
    unittest.main()
