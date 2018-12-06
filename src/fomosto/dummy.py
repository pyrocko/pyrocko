# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function, division

from builtins import range

import logging
import os
import signal

import numpy as num

from pyrocko import trace, gf

guts_prefix = 'pf'

logger = logging.getLogger('pyrocko.fomosto.dummy')


class Interrupted(gf.store.StoreError):
    def __str__(self):
        return 'Interrupted.'


class DummyGFBuilder(gf.builder.Builder):
    def __init__(self, store_dir, step, shared, force=False):
        self.store = gf.store.Store(store_dir, 'w')
        gf.builder.Builder.__init__(
            self, self.store.config, step, block_size=(1, 51), force=force)

    def work_block(self, index):
        (sz, firstx), (sz, lastx), (ns, nx) = \
            self.get_block_extents(index)

        logger.info(
            'Starting block %i / %i' % (index+1, self.nblocks))

        interrupted = []

        def signal_handler(signum, frame):
            interrupted.append(True)

        original = signal.signal(signal.SIGINT, signal_handler)
        self.store.lock()
        duplicate_inserts = 0
        try:
            for x in num.linspace(firstx, lastx, nx):
                for ig in range(self.store.config.ncomponents):
                    args = (sz, x, ig)
                    irec = self.store.config.irecord(*args)
                    tr = trace.Trace(
                        deltat=self.store.config.deltat,
                        ydata=num.zeros(10000)+float(irec))

                    gf_tr = gf.store.GFTrace.from_trace(tr)

                    try:
                        self.store.put(args, gf_tr)
                    except gf.store.DuplicateInsert:
                        duplicate_inserts += 1

        finally:
            if duplicate_inserts:
                logger.warn(
                    '%i insertions skipped (duplicates)' % duplicate_inserts)

            self.store.unlock()
            signal.signal(signal.SIGINT, original)

        if interrupted:
            raise KeyboardInterrupt()

        logger.info('Done with block %i / %i' % (index+1, self.nblocks))


km = 1000.


def init(store_dir, variant):
    if variant is not None:
        raise gf.store.StoreError('unsupported variant: %s' % variant)

    store_id = os.path.basename(os.path.realpath(store_dir))

    config = gf.meta.ConfigTypeA(
        id=store_id,
        ncomponents=2,
        sample_rate=1.0,
        receiver_depth=0*km,
        source_depth_min=0*km,
        source_depth_max=400*km,
        source_depth_delta=4*km,
        distance_min=4*km,
        distance_max=400*km,
        distance_delta=4*km,
        modelling_code_id='dummy')

    config.validate()
    return gf.store.Store.create_editables(store_dir, config=config)


def build(
        store_dir,
        force=False,
        nworkers=None,
        continue_=False,
        step=None,
        iblock=None):

    return DummyGFBuilder.build(
        store_dir, force=force, nworkers=nworkers, continue_=continue_,
        step=step, iblock=iblock)
