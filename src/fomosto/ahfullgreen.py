# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

import numpy as num
import logging
import os
import math
import signal

from pyrocko.guts import Object, clone
from pyrocko import trace, cake, gf
from pyrocko.ahfullgreen import add_seismogram, AhfullgreenSTFImpulse, \
    AhfullgreenSTF
from pyrocko.moment_tensor import MomentTensor, symmat6

km = 1000.

guts_prefix = 'pf'

logger = logging.getLogger('pyrocko.fomosto.ahfullgreen')

# how to call the programs
program_bins = {
    'ahfullgreen': 'ahfullgreen',
}

components = 'x y z'.split()


def example_model():
    material = cake.Material(vp=3000., vs=1000., rho=3000., qp=200., qs=100.)
    layer = cake.HomogeneousLayer(
        ztop=0., zbot=30*km, m=material, name='fullspace')
    mod = cake.LayeredModel()
    mod.append(layer)
    return mod


class AhfullgreenError(gf.store.StoreError):
    pass


class AhfullgreenConfig(Object):
    stf = AhfullgreenSTF.T(default=AhfullgreenSTFImpulse.D())


def make_traces(material, source_mech, deltat, norths, easts,
                source_depth, receiver_depth, stf):

    if isinstance(source_mech, MomentTensor):
        m6 = source_mech.m6()
        f = (0., 0., 0.)
    elif isinstance(source_mech, tuple):
        m6 = (0., 0., 0., 0., 0., 0.)
        f = source_mech

    npad = 120

    traces = []
    for isource, (north, east) in enumerate(zip(norths, easts)):
        d3d = math.sqrt(
            north**2 + east**2 + (receiver_depth - source_depth)**2)

        tmin = (math.floor(d3d / material.vp / deltat) - npad) * deltat
        tmax = (math.ceil(d3d / material.vs / deltat) + npad) * deltat
        ns = int(round((tmax - tmin) / deltat))

        outx = num.zeros(ns)
        outy = num.zeros(ns)
        outz = num.zeros(ns)

        x = (north, east, receiver_depth-source_depth)

        add_seismogram(
            material.vp, material.vs, material.rho, material.qp, material.qs,
            x, f, m6, 'displacement',
            deltat, tmin, outx, outy, outz,
            stf=stf)

        for i_comp, o in enumerate((outx, outy, outz)):
            comp = components[i_comp]
            tr = trace.Trace('', '%04i' % isource, '', comp,
                             tmin=tmin, ydata=o, deltat=deltat,
                             meta=dict(isource=isource))

            traces.append(tr)

    return traces


class AhfullGFBuilder(gf.builder.Builder):
    def __init__(self, store_dir, step, shared, block_size=None, force=False):

        self.store = gf.store.Store(store_dir, 'w')

        block_size = {
            'A': (1, 2000),
            'B': (1, 1, 2000),
            'C': (1, 1, 2)}[self.store.config.short_type]

        gf.builder.Builder.__init__(
            self, self.store.config, step, block_size=block_size, force=force)

        self.ahfullgreen_config = self.store.get_extra('ahfullgreen')

    def cleanup(self):
        self.store.close()

    def work_block(self, index):
        store_type = self.store.config.short_type

        if store_type == 'A':
            (sz, firstx), (sz, lastx), (ns, nx) = \
                self.get_block_extents(index)

            rz = self.store.config.receiver_depth
            sy = 0.0
        elif store_type == 'B':
            (rz, sz, firstx), (rz, sz, lastx), (nr, ns, nx) = \
                self.get_block_extents(index)
            sy = 0.0
        elif store_type == 'C':
            (sz, sy, firstx), (sz, sy, lastx), (nz, ny, nx) = \
                self.get_block_extents(index)
            rz = self.store.config.receiver.depth

        logger.info('Starting block %i / %i' %
                    (index+1, self.nblocks))

        dx = self.gf_config.deltas[-1]

        xs = num.linspace(firstx, firstx + (nx-1)*dx, nx).tolist()

        mmt1 = (MomentTensor(m=symmat6(1, 0, 0, 1, 0, 0)),
                {'x': (0, +1), 'y': (3, +1), 'z': (5, +1)})
        mmt2 = (MomentTensor(m=symmat6(0, 0, 0, 0, 1, 1)),
                {'x': (1, +1), 'y': (4, +1), 'z': (6, +1)})
        mmt3 = (MomentTensor(m=symmat6(0, 0, 1, 0, 0, 0)),
                {'x': (2, +1), 'z': (7, +1)})
        mmt4 = (MomentTensor(m=symmat6(0, 1, 0, 0, 0, 0)),
                {'x': (8, +1), 'z': (9, +1)})
        mmt0 = (MomentTensor(m=symmat6(1, 1, 1, 0, 0, 0)),
                {'x': (0, +1), 'z': (1, +1)})

        component_scheme = self.store.config.component_scheme

        if component_scheme == 'elastic8':
            gfmapping = [mmt1, mmt2, mmt3]

        elif component_scheme == 'elastic10':
            gfmapping = [mmt1, mmt2, mmt3, mmt4]

        elif component_scheme == 'elastic2':
            gfmapping = [mmt0]

        elif component_scheme == 'elastic5':
            gfmapping = [
                ((1., 1., 0.), {'x': (1, +1), 'z': (4, +1), 'y': (2, +1)}),
                ((0., 0., 1.), {'x': (0, +1), 'z': (3, +1)})]
        elif component_scheme == 'elastic18':
            gfmapping = []
            for im in range(6):
                m6 = [0.0] * 6
                m6[im] = 1.0

                gfmapping.append(
                    (MomentTensor(m=symmat6(*m6)),
                     {'x': (im, +1),
                      'y': (6+im, +1),
                      'z': (12+im, +1)}))

        else:
            raise gf.UnavailableScheme(
                'fomosto backend "ahfullgreen" cannot handle component scheme '
                '"%s"' % component_scheme)

        for source_mech, gfmap in gfmapping:

            if component_scheme != 'elastic18':
                norths = xs
                easts = num.zeros_like(xs)
            else:
                receiver = self.store.config.receiver
                data = []
                for x in xs:
                    source = clone(self.store.config.source_origin)
                    source.north_shift = x
                    source.east_shift = sy
                    source.depth = sz
                    data.append(receiver.offset_to(source))

                norths, easts = -num.array(data).T

            rawtraces = make_traces(
                self.store.config.earthmodel_1d.require_homogeneous(),
                source_mech, 1.0/self.store.config.sample_rate,
                norths, easts, sz, rz, self.ahfullgreen_config.stf)

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

                    x = xs[tr.meta['isource']]
                    if x > firstx + (nx-1)*dx:
                        logger.error('x out of range')
                        continue

                    ig, factor = gfmap[tr.channel]

                    args = {
                        'A': (sz, x, ig),
                        'B': (rz, sz, x, ig),
                        'C': (sz, sy, x, ig)}[store_type]

                    tr = tr.snap()

                    gf_tr = gf.store.GFTrace.from_trace(tr)
                    gf_tr.data *= factor

                    try:
                        self.store.put(args, gf_tr)
                    except gf.store.DuplicateInsert:
                        duplicate_inserts += 1

            finally:
                if duplicate_inserts:
                    logger.warning('%i insertions skipped (duplicates)' %
                                   duplicate_inserts)

                self.store.unlock()
                signal.signal(signal.SIGINT, original)

            if interrupted:
                raise KeyboardInterrupt()

        logger.info('Done with block %i / %i' %
                    (index+1, self.nblocks))


def init(store_dir, variant, config_params=None):
    assert variant is None

    modelling_code_id = 'ahfullgreen'

    store_id = os.path.basename(os.path.realpath(store_dir))

    ahfullgreen = AhfullgreenConfig()

    d = dict(
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
                id='anyP',
                definition='P,p,\\P,\\p'),
            gf.meta.TPDef(
                id='anyS',
                definition='S,s,\\S,\\s')])

    if config_params is not None:
        d.update(config_params)

    config = gf.meta.ConfigTypeA(**d)
    config.validate()
    return gf.store.Store.create_editables(
        store_dir, config=config, extra={'ahfullgreen': ahfullgreen})


def build(store_dir, force=False, nworkers=None, continue_=False, step=None,
          iblock=None):

    return AhfullGFBuilder.build(
        store_dir, force=force, nworkers=nworkers, continue_=continue_,
        step=step, iblock=iblock)
