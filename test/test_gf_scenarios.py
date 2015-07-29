import unittest
import logging
import random
import time
import math
import multiprocessing
from pyrocko import gf, util
from pyrocko import moment_tensor as pmt

logger = logging.getLogger('test_gf_scenarios')
km = 1000.
d2r = math.pi/180.


def rand(mi, ma):
    mi = float(mi)
    ma = float(ma)
    return random.random()*(ma-mi) + mi


class GFScenariosTestCase(unittest.TestCase):

    def test_regional(self):
        engine = gf.get_engine()
        store_id = 'crust2_mf'
        try:
            engine.get_store(store_id)
        except gf.NoSuchStore:
            logger.warn('GF Store %s not available - skipping test' % store_id)
            return

        nsources = 10
        nstations = 10

        print 'cache source channels par wallclock seismograms_per_second'
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
                for isource in xrange(nsources):
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
                for istation in xrange(nstations):
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
                            store_id=store_id)

                        targets.append(target)

                ntraces = len(targets) * len(sources)
                for temperature in ['cold', 'hot']:
                    t0 = time.time()
                    engine.process(sources, targets, nprocs=nprocs)
                    t1 = time.time()
                    duration = t1 - t0
                    sps = ntraces / duration
                    if temperature == 'hot':
                        if nprocs == 1:
                            sps_ref = sps
                        print '%-5s %-6s %-8s %3i %9.3f %12.1f %12.1f' % (
                            temperature, sourcetype, channels, nprocs, t1-t0,
                            sps, sps/sps_ref)


if __name__ == '__main__':
    util.setup_logging('test_gf_scenarios', 'warning')
    unittest.main()
