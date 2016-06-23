import numpy as num
import logging
import os
import shutil
import math
import copy
import signal

from tempfile import mkdtemp
from subprocess import Popen, PIPE
from os.path import join as pjoin

from pyrocko.guts import Float, Int, Tuple, List, Complex, Bool, Object, String
from pyrocko import trace, util, cake, model, moment_tensor
from pyrocko import gf
from pyrocko.ahfullgreen import add_seismogram, Impulse, Gauss

km = 1000.

guts_prefix = 'pf'

Timing = gf.meta.Timing

from pyrocko.moment_tensor import MomentTensor, symmat6

logger = logging.getLogger('fomosto.ahfullgreen')

# how to call the programs
program_bins = {
    'ahfullgreen': 'ahfullgreen',
}

components = 'r t z'.split()

def nextpow2(i):
    return 2**int(math.ceil(math.log(i)/math.log(2.)))

def example_model():
    material = cake.Material(vp=3000., vs=1000., rho=3000., qp=200., qs=100.)
    layer = cake.HomogeneousLayer(
        ztop=0., zbot=30*km, m=material, name='fullspace')
    mod = cake.LayeredModel()
    mod.append(layer)
    return mod


class AhfullgreenConfig(Object):

    time_region = Tuple.T(2, Timing.T(), default=(
        Timing('-10'), Timing('+890')))

    cut = Tuple.T(2, Timing.T(), optional=True)
    fade = Tuple.T(4, Timing.T(), optional=True)


    def items(self):
        return dict(self.T.inamevals(self))


class AhfullgreenConfigFull(AhfullgreenConfig):

    source_depth = Float.T(default=10.0)
    receiver_depth = Float.T(default=0.0)
    receiver_distances = List.T(Float.T())
    nsamples = Int.T(default=256)

    earthmodel_1d = gf.meta.Earthmodel1D.T(optional=True)

    @staticmethod
    def example():

        conf = AhfullgreenConfigFull()
        conf.receiver_distances = [2000.]
        #conf.time_start = -5.0
        #conf.time_reduction_velocity = 15.0
        conf.earthmodel_1d = example_model()
        return conf


class AhfullgreenError(gf.store.StoreError):
    pass


class Interrupted(gf.store.StoreError):
    def __str__(self):
        return 'Interrupted.'


class AhfullgreenRunner:

    def __init__(self, tmp=None, keep_tmp=False):
        self.tempdir = mkdtemp(prefix='ahfullrun-', dir=tmp)
        self.keep_tmp = keep_tmp
        self.config = None
        self.traces = []

    def run(self, config):
        elements = list(config.earthmodel_1d.elements())

        if len(elements) != 2:
            raise AhfullgreenError('More than one layer in earthmodel')
        if not isinstance(elements[1], cake.HomogeneousLayer):
            raise AhfullgreenError('Layer has to be a HomogeneousLayer')

        l = elements[1].m
        vp, vs, density, qp, qs = (l.vp, l.vs, l.rho, l.qp, l.qs)
        f = (0., 0., 0.)   # single force
        m6 = config.source_mech.m6()
        ns = config.nsamples
        deltat = config.deltat
        #vred = config.time_reduction_velocity
        tmin = 0.
        for i_distance, d in enumerate(config.receiver_distances):
            outx = num.zeros(ns)
            outy = num.zeros(ns)
            outz = num.zeros(ns)
            x = (d, 0.0, config.source_depth)
            add_seismogram(vp, vs, density, qp, qs, x, f, m6, 'displacement',
                           deltat, 0.0, outx, outy, outz, stf=Impulse())

            for i_comp, o in enumerate((outx, outy, outz)):
                comp = components[i_comp]
                tr = trace.Trace('', '%04i' % i_distance, '', comp,
                                tmin=tmin, ydata=o, deltat=deltat,
                                             meta=dict(distance=d,
                                                       azimuth=0.))
                self.traces.append(tr)

    def get_traces(self):
        # return traces and empty traces list
        tmp = self.traces
        self.traces = []
        return tmp

    def __del__(self):
        if self.tempdir:
            if not self.keep_tmp:
                shutil.rmtree(self.tempdir)
                self.tempdir = None
            else:
                logger.warn(
                    'not removing temporary directory: %s' % self.tempdir)


class AhfullGFBuilder(gf.builder.Builder):
    def __init__(self, store_dir, step, shared, block_size=None, tmp=None):

        self.store = gf.store.Store(store_dir, 'w')

        if block_size is None:
            block_size = (1, 1, 2000)

        if len(self.store.config.ns) == 2:
            block_size = block_size[1:]

        gf.builder.Builder.__init__(
            self, self.store.config, step, block_size=block_size)

        baseconf = self.store.get_extra('ahfullgreen')

        conf = AhfullgreenConfigFull(**baseconf.items())
        conf.earthmodel_1d = self.store.config.earthmodel_1d
        deltat = 1.0/self.store.config.sample_rate
        conf.deltat = deltat

        if 'time_window_min' not in shared:
            d = self.store.make_timing_params(
                conf.time_region[0], conf.time_region[1])

            shared['time_window_min'] = d['tlenmax_vred']

        time_window_min = shared['time_window_min']

        conf.nsamples = nextpow2(int(round(time_window_min / deltat)) + 1)

        self.ahfullgreen_config = conf

        self.tmp = tmp
        if self.tmp is not None:
            util.ensuredir(self.tmp)

    def work_block(self, index):
        if len(self.store.config.ns) == 2:
            (sz, firstx), (sz, lastx), (ns, nx) = \
                self.get_block_extents(index)

            rz = self.store.config.receiver_depth
        else:
            (rz, sz, firstx), (rz, sz, lastx), (nr, ns, nx) = \
                self.get_block_extents(index)

        conf = copy.deepcopy(self.ahfullgreen_config)

        logger.info('Starting block %i / %i' %
                    (index+1, self.nblocks))

        conf.source_depth = float(sz)
        conf.receiver_depth = float(rz)

        runner = AhfullgreenRunner(tmp=self.tmp)

        dx = self.gf_config.distance_delta

        distances = num.linspace(firstx, firstx + (nx-1)*dx, nx).tolist()

        #mmt1 = (MomentTensor(m=symmat6(1, 0, 0, 1, 0, 0)),
        #        {'r': (0, +1), 't': (3, +1), 'z': (5, +1)})
        #mmt2 = (MomentTensor(m=symmat6(0, 0, 0, 0, 1, 1)),
        #        {'r': (1, +1), 't': (4, +1), 'z': (6, +1)})
        #mmt3 = (MomentTensor(m=symmat6(0, 0, 1, 0, 0, 0)),
        #        {'r': (2, +1), 'z': (7, +1)})
        mmt1 = (MomentTensor(m=symmat6(1, 0, 0, 1, 0, 0)),
                {'r': (0, +1), 't': (3, +1), 'z': (5, +1)})
        mmt2 = (MomentTensor(m=symmat6(0, 0, 0, 0, 1, 1)),
                {'r': (1, +1), 't': (4, +1), 'z': (6, +1)})
        mmt3 = (MomentTensor(m=symmat6(0, 0, 1, 0, 0, 0)),
                {'r': (2, +1), 'z': (7, +1)})
        mmt4 = (MomentTensor(m=symmat6(0, 1, 0, 0, 0, 0)),
                {'r': (8, +1), 'z': (9, +1)})

        component_scheme = self.store.config.component_scheme
        #if component_scheme != 'elastic10':
        #    raise Exception('Invalid component scheme')
        #off = 8

        #msf = (None, {
        #    'fz.tr': (off+0, +1),
        #    'fh.tr': (off+1, +1),
        #    'fh.tt': (off+2, -1),
        #    'fz.tz': (off+3, +1),
        #    'fh.tz': (off+4, +1)})

        if component_scheme == 'elastic8':
            gfmapping = [mmt1, mmt2, mmt3]

        if component_scheme == 'elastic10':
            gfmapping = [mmt1, mmt2, mmt3, mmt4]

        #elif component_scheme == 'elastic13':
        #    gfmapping = [mmt1, mmt2, mmt3, msf]

        #elif component_scheme == 'elastic15':
        #    gfmapping = [mmt1, mmt2, mmt3, mmt4, msf]


        conf.receiver_distances = distances

        for mt, gfmap in gfmapping:
            if mt:
                m = mt.m()
                f = float
                conf.source_mech = mt
            else:
                conf.source_mech = None

            if conf.source_mech is not None:
                runner.run(conf)

            rawtraces = runner.get_traces()
            interrupted = []

            def signal_handler(signum, frame):
                interrupted.append(True)

            original = signal.signal(signal.SIGINT, signal_handler)
            self.store.lock()
            duplicate_inserts = 0
            try:
                for itr, tr in enumerate(rawtraces):
                    if tr.channel not in gfmap:
                        logger.debug('%s not in gfmap' % tr.channel)
                        continue

                    x = tr.meta['distance']
                    if x > firstx + (nx-1)*dx:
                        logger.error("x out of range")
                        continue

                    ig, factor = gfmap[tr.channel]

                    if len(self.store.config.ns) == 2:
                        args = (sz, x, ig)
                    else:
                        args = (rz, sz, x, ig)
                    if conf.cut:
                        tmin = self.store.t(conf.cut[0], args[:-1])
                        tmin = math.floor(tmin/conf.deltat)*conf.deltat
                        tmax = self.store.t(conf.cut[1], args[:-1])
                        tmax = math.ceil(tmax/conf.deltat)*conf.deltat
                        if None in (tmin, tmax):
                            # maybe notify the user, that begin / end timings
                            # are inappropriate?
                            logger.error("tmin or tmax from begin and end is None")
                            continue
                        tr.chop(tmin, tmax)

                    tr = tr.snap()

                    if conf.fade:
                        ta, tb, tc, td = [
                            self.store.t(v, args[:-1]) for v in conf.fade]

                        if None in (ta, tb, tc, td):
                            logger.error("ta, tb, tx, td is None")
                            continue

                        if not (ta <= tb and tb <= tc and tc <= td):
                            raise AhfullgreenError(
                                'invalid fade configuration')

                        t = tr.get_xdata()
                        fin = num.interp(t, [ta, tb], [0., 1.])
                        fout = num.interp(t, [tc, td], [1., 0.])
                        anti_fin = 1. - fin
                        anti_fout = 1. - fout

                        y = tr.ydata

                        sum_anti_fin = num.sum(anti_fin)
                        sum_anti_fout = num.sum(anti_fout)

                        if sum_anti_fin != 0.0:
                            yin = num.sum(anti_fin*y) / sum_anti_fin
                        else:
                            yin = 0.0

                        if sum_anti_fout != 0.0:
                            yout = num.sum(anti_fout*y) / sum_anti_fout
                        else:
                            yout = 0.0

                        y2 = anti_fin*yin + fin*fout*y + anti_fout*yout

                        if conf.relevel_with_fade_in:
                            y2 -= yin

                        tr.set_ydata(y2)
                    gf_tr = gf.store.GFTrace.from_trace(tr)
                    gf_tr.data *= factor

                    try:
                        self.store.put(args, gf_tr)
                    except gf.store.DuplicateInsert:
                        duplicate_inserts += 1

            finally:
                if duplicate_inserts:
                    logger.warn('%i insertions skipped (duplicates)' %
                                duplicate_inserts)

                self.store.unlock()
                signal.signal(signal.SIGINT, original)

            if interrupted:
                raise KeyboardInterrupt()

            conf.gf_sw_source_types = (0, 0, 0, 0, 0, 0)

        logger.info('Done with block %i / %i' %
                    (index+1, self.nblocks))


def init(store_dir, variant):
    assert variant is None

    modelling_code_id = 'ahfullgreen'

    ahfull = AhfullgreenConfig()
    ahfull.time_region = (
        gf.meta.Timing('begin-2'),
        gf.meta.Timing('end+2'))

    ahfull.cut = (
        gf.meta.Timing('begin-2'),
        gf.meta.Timing('end+2'))

    store_id = os.path.basename(os.path.realpath(store_dir))

    config = gf.meta.ConfigTypeA(
        id=store_id,
        ncomponents=10,
        sample_rate=20.,
        receiver_depth=0*km,
        source_depth_min=1*km,
        source_depth_max=10*km,
        source_depth_delta=1*km,
        distance_min=1*km,
        distance_max=20*km,
        distance_delta=1*km,
        earthmodel_1d=example_model(),
        modelling_code_id=modelling_code_id,
        tabulated_phases=[
            gf.meta.TPDef(
                id='begin',
                definition='p'),
            gf.meta.TPDef(
                id='end',
                definition='s'),
            gf.meta.TPDef(
                id='p',
                definition='p'),
            gf.meta.TPDef(
                id='s',
                definition='s')])

    config.validate()
    return gf.store.Store.create_editables(
        store_dir, config=config, extra={'ahfullgreen': ahfull})


def build(store_dir, force=False, nworkers=None, continue_=False, step=None,
          iblock=None):

    return AhfullGFBuilder.build(
        store_dir, force=force, nworkers=nworkers, continue_=continue_,
        step=step, iblock=iblock)
