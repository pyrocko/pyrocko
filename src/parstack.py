import numpy as num
import parstack_ext


def parstack(arrays, offsets, shifts, weights, method,
             lengthout=-1,
             offsetout=0,
             result=None,
             nparallel=None,
             impl='openmp'):

    if nparallel is None:
        import multiprocessing
        nparallel = multiprocessing.cpu_count()

    narrays = offsets.size
    assert(len(arrays) == narrays)
    nshifts = shifts.size / narrays
    assert shifts.shape == (nshifts, narrays)
    shifts = num.reshape(shifts, (nshifts*narrays))
    assert weights.shape == (nshifts, narrays)
    weights = num.reshape(weights, (nshifts*narrays))

    if impl == 'openmp':
        parstack_impl = parstack_ext.parstack
    elif impl == 'numpy':
        parstack_impl = parstack_numpy

    result, offset = parstack_impl(
        arrays, offsets, shifts, weights, method,
        lengthout, offsetout, result, nparallel)

    if method == 0:
        nsamps = result.size / nshifts
        result = result.reshape((nshifts, nsamps))

    return result, offset


def parstack_numpy(arrays, offsets, shifts, weights, method, lengthout, offsetout, result, nparallel):

    # nparallel is ignored here

    narrays = offsets.size

    lengths = num.array([a.size for a in arrays], dtype=num.int)
    if lengthout < 0:

        imin = offsets[0] + shifts[0]
        imax = imin + lengths[0]
        for iarray in xrange(len(arrays)):
            istarts = offsets[iarray] + shifts[iarray::narrays]
            iends = istarts + lengths[iarray]
            imin = min(imin, num.amin(istarts))
            imax = max(imax, num.amax(iends))

        nsamp = imax - imin
        offsetout = imin
    else:
        nsamp = lengthout
        imin = offsetout

    nshifts = shifts.size / narrays
    result = num.zeros(nsamp*nshifts, dtype=num.float)

    for ishift in xrange(nshifts):
        for iarray in xrange(narrays):
            istart = offsets[iarray] + shifts[ishift*narrays + iarray]
            weight = weights[ishift*narrays + iarray]
            istart_r = ishift*nsamp + istart - imin
            jstart = max(0, imin - istart)
            jstop = max(jstart, min(nsamp - istart + imin, lengths[iarray]))
            result[istart_r+jstart:istart_r+jstop] += \
                arrays[iarray][jstart:jstop] * weight

    if method == 0:
        return result, offsetout
    elif method == 1:
        return num.amax(result.reshape((nshifts, nsamp)), axis=1), offsetout
