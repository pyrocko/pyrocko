from pyrocko import trace
import numpy as num


def sub_covariance(n, dt, tzero):
    '''Calculate SubCovariance Matrix of trace object following Duputel et al. 2012 GJI
    "Uncertainty estimations for seismic source inversions" p. 5
    :param: n - length of trace/ samples of quadratic Covariance matrix
    :param: dt - time step of samples 
    :param: tzero - shortest period of waves in trace
    Cd(i,j) = (Variance of trace)*exp(-abs(ti-tj)/(shortest period T0 of waves))
    i,j are samples of the seismic trace

    Here, without the variance part, as it can be just multiplied to the array, which 
    is trace independent if the frequency content is the same. So it doesnt need to be 
    recalculated.
    '''

    return num.exp(-num.abs(num.arange(n)[:,num.newaxis]-num.arange(n)[num.newaxis,:])*dt / tzero)

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


