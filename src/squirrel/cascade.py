# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

import re
import math
import logging
import numpy as num
from pyrocko import util, progress
from pyrocko.trace import NoData
from pyrocko.plot import nice_time_tick_inc_approx_secs
from .storage import get_storage_scheme

logger = logging.getLogger('psq.cascade')


def level_codes_check(ilevel, codes, method):
    if ilevel == 0:
        assert not re.match(r'-L%s\d\d$' % method, codes.extra)
        return codes.replace(extra=codes.extra + '-L%s00' % method)
    else:
        assert codes.extra.endswith('-L%s%02i' % (method, ilevel-1))
        return codes.replace(
            extra=codes.extra[:-(4+len(method))] + '-L%s%02i' % (
                method, ilevel))


def cascade(
        squirrel,
        nlevels,
        storage_path,
        nfold=2,
        kinds=None,
        codes=None,
        tmin=None,
        tmax=None,
        methods=['max', 'mean', 'min'],
        accessor_id='default',
        storage_scheme='rug-store-100'):

    assert kinds is None or kinds == ['carpet']

    method_functions = {
        'min': num.nanmin,
        'max': num.nanmax,
        'mean': num.nanmean,
    }

    storage = get_storage_scheme(storage_scheme)
    storage.set_base_path(storage_path)

    tmin_content, tmax_content = squirrel.get_time_span(['carpet'])
    nsamples_block = nfold * 1000

    task_levels = progress.task('Cascading (level)', logger=logger)
    for ilevel in task_levels(range(nlevels)):
        task_channels = progress.task('Cascading (channel)', logger=logger)

        codes_next = set()

        paths = []
        codes_info = list(squirrel.get_codes_info('carpet', codes))
        for (pattern, kind_codes_id, codes_cand, deltat) in task_channels(
                codes_info):

            tinc = nice_time_tick_inc_approx_secs(deltat * nsamples_block)
            tpad = deltat * nfold

            task_time = progress.task(
                'Cascading (time)', logger=logger)

            for batch in task_time(util.iter_windows(
                    tmin=tmin,
                    tmax=tmax,
                    tinc=tinc,
                    tmin_content=tmin_content,
                    tmax_content=tmax_content,
                    snap_window=True)):

                for carpet in squirrel.get_carpets(
                        tmin=batch.tmin-tpad,
                        tmax=batch.tmax+tpad,
                        kind_codes_ids=[kind_codes_id],
                        accessor_id=accessor_id):

                    # carpet.data[carpet.data > 10000.] = num.nan

                    if ilevel == 0:
                        x_methods = methods
                    else:
                        m = re.search(
                            r'-L(%s)\d\d$' % '|'.join(methods),
                            carpet.codes.extra)

                        x_methods = [m.group(1)]

                    for method in x_methods:
                        try:
                            folded = carpet.fold(
                                carpet.deltat*nfold,
                                method=method_functions[method])

                            # probably not a clean solution to prevent the
                            # rounding errors...

                            eps = 1e-5
                            folded.tmin += eps

                            folded = folded.chop(
                                batch.tmin,
                                batch.tmax,
                                snap=(math.ceil, math.ceil))

                            folded.tmin -= eps

                        except NoData:
                            continue

                        if folded.nsamples == 0:
                            continue

                        folded.codes = level_codes_check(
                            ilevel, carpet.codes, method)
                        codes_next.add(folded.codes)

                        folded.data = folded.data.astype(num.float32)

                        paths.extend(storage.save_carpets(folded))

                squirrel.advance_accessor(accessor_id, 'carpet')

        squirrel.add(paths)
        codes = list(codes_next)
