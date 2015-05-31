from pyrocko import trace
import numpy as num


def Variance_of_trace(traceobj,event_t):
    '''Calculate variance of the input trace object from time tmin until given event time
    :param: traceobj: :py:class:`Trace` object
    :param: event:    pyrocko event object including event time
    '''

    trace_event_t_samples = num.floor((event_t-traceobj.tmin)/traceobj.deltat)
    Variance = num.var(traceobj.ydata[0:trace_event_t_samples])
    return Variance

def SubCovariance(traceobj,Tzero):
    '''Calculate SubCovariance Matrix of trace object following Duputel et al. 2012 GJI
    "Uncertainty estimations for seismic source inversions" p. 5
    :param: traceobj - :py:class:`Trace` object
    :param: T0       - shortest period of waves in trace
    Cd(i,j) = (Variance of trace)*exp(-abs(ti-tj)/(shortest period T0 of waves))
    i,j are samples of the seismic trace

    Here, without the variance part, as it can be just multiplied to the array, which 
    is trace independent if the frequency content is the same. So it doesnt need to be 
    recalculated.
    '''

    trace_length = traceobj.data_len()
    Csub = num.zeros((trace_length,trace_length))
    for k in range(trace_length):
        for l in range(k,trace_length):
            if k == l:
                Csub[k,l] = 1;
            else:
                Csub[k,l] = num.exp(-(num.abs(k-l)*traceobj.deltat)/Tzero)
                Csub[l,k] = Csub[k,l]
    return Csub

def CovInvcov(Variance,Csub):
    '''Calculate Covariance matrix and Inverse of the Covariance matrix 
    following Duputel et al. 2012 GJI "Uncertainty estimations for seismic source inversions" p. 5
    :param: Variance: Variance of the data
    :param: Csub:     Subcovariance Matrix of a trace with a certain length 
                      (Output of SubCovariance function)
    '''
    Cd = Variance * Csub
    InvCd = num.linalg.inv(Cd)
    return Cd, InvCd

def llh(rt_data, can_data, InvCovariance):
    '''Calculate data likelihood that data (rt_data) is explained by a given model (can_data)
    :param adata (Observation) :py:class:`Trace` object
    :param adata (Model) :py:class:`Trace` object from GF store
    :param InvCovariance: Inverse Covariance matrix of the data
    '''
    res = rt_data - can_data
    part1 = num.dot([res.transpose()], InvCovariance)  #llk = -0.5*AT*Cd*A
    data_llh = -0.5 * num.dot(part1,res)
    return data_llh

def data_setup(rt, candidate, setup, nocache=False, debug=False):
    """
    Filter and taper data with input of taper and filter objects defined in misfit setup.

    :param rt-reference trace: :py:class:`Trace` object
    :param candidate: :py:class:`Trace` object
    :param setup: :py:class:`MisfitSetup` object
    :returns: filtered and tapered data and synthetic trace

    If the sampling rates of *self* and *candidate* differ, the trace with
    the higher sampling rate will be downsampled.
    """

    a = rt
    b = candidate

    for tr in (a, b):
        if not tr._pchain:
            tr.init_chain()

    deltat = max(a.deltat, b.deltat)
    tmin = min(a.tmin, b.tmin) - deltat
    tmax = max(a.tmax, b.tmax) + deltat

    rt_data, aproc = a.run_chain(tmin, tmax, deltat, setup, nocache)
    can_data, bproc = b.run_chain(tmin, tmax, deltat, setup, nocache)

    return rt_data, can_data

