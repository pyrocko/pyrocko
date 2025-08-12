# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

import os
import re
import math
import logging
import numpy as num
from pyrocko import util, progress
from pyrocko.io import rug
from pyrocko.trace import NoData
from pyrocko.plot import nice_time_tick_inc_approx_secs

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
        nfold=2,
        kinds=None,
        codes=None,
        tmin=None,
        tmax=None,
        method='max',
        accessor_id='default'):

    assert kinds is None or kinds == ['carpet']

    method_functions = {
        'min': num.nanmin,
        'max': num.nanmax,
        'mean': num.nanmean,
    }

    tmin_content, tmax_content = squirrel.get_time_span(['carpet'])
    nsamples_block = nfold * 1000
    time_format = '%Y-%m-%d_%H-%M-%S'

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

            task_time = None

            for batch in util.iter_windows(
                    tmin=tmin,
                    tmax=tmax,
                    tinc=tinc,
                    tmin_content=tmin_content,
                    tmax_content=tmax_content,
                    snap_window=True):

                if task_time is None:
                    task_time = progress.task(
                        'Cascading (time)', n=batch.n, logger=logger)

                task_time.update(batch.i)

                for carpet in squirrel.get_carpets(
                        tmin=batch.tmin-tpad,
                        tmax=batch.tmax+tpad,
                        kind_codes_ids=[kind_codes_id],
                        accessor_id=accessor_id):

                    carpet.data[carpet.data > 10000.] = num.nan

                    try:
                        folded = carpet.fold(
                            carpet.deltat*nfold,
                            method=method_functions[method])

                        folded = folded.chop(
                            batch.tmin,
                            batch.tmax,
                            snap=(math.ceil, math.ceil))
                    except NoData:
                        continue

                    if folded.nsamples == 0:
                        continue

                    folded.codes = level_codes_check(
                        ilevel, carpet.codes, method)
                    codes_next.add(folded.codes)

                    path = os.path.join(
                        'cascade',
                        'L%02i' % ilevel,
                        'cascade_{tmin}_{tmax}_{codes}.rug'.format(
                            tmin=util.time_to_str(
                                folded.tmin, format=time_format),
                            tmax=util.time_to_str(
                                folded.tmax, format=time_format),
                            codes=folded.codes))

                    util.ensuredirs(path)
                    paths.extend(rug.save([folded], path))

                squirrel.advance_accessor(accessor_id, 'carpet')

            if task_time is not None:
                task_time.done()

        squirrel.add(paths)
        codes = list(codes_next)
