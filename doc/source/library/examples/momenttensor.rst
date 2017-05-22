Moment Tensor 
================================

Transformations to and from Moment Tensor using the :py:class:`pyrocko.moment_tensor` modules.
Conversion between Global Centroid Moment tensors format
with pointing upward, southward, and eastward (USE) system to
geographic Cartesian tensor (NED) transforms by:

.. math::

    Mrr=Mdd, Mt\theta\theta=Mnn, M\phi\phi= Mee, Mr\theta =Mnd, Mr\phi=-Med,M\theta\phi=-Mne

Convert from Moment Tensor components to strike, dip and rake
-------------------------

Moment Tensor construction is shown by using the moment components
and conversion to strike, dip and rake.

::

    from pyrocko import moment_tensor as mtm
    magnitude = 6.3  # Magnitude of the earthquake
    
    exp = mtm.magnitude_to_moment(magnitude)  # convert the mag to moment in [Nm]
    
    m = mtm.MomentTensor()  # init pyrocko moment tensor 
    m.mnn = 2.34*exp
    m.mee = -2.64*exp
    m.mdd = 0.295*exp
    m.mne = 1.49*exp
    m.mnd = 0.182*exp
    m.med = -0.975*exp
    print m  # print moment tensor
    (s1, d1, _), (s2, d2, _) = m.both_strike_dip_rake()  # gives out both nodal pl.



Strike, dip and rake to MT:
----------------------------

Conversion from strike, dip and rake to the Moment Tensor. Afterwards
we normalize the Moment Tensor. 

::

    from pyrocko import moment_tensor as mtm
    
    magnitude = 6.3  # Magnitude of the earthquake
    
    m0 = mtm.magnitude_to_moment(magnitude)  # convert the mag to moment
    
    strike = 130
    dip = 40
    rake = 110
    mt = mtm.MomentTensor(strike=strike, dip=dip, rake=rake, scalar_moment=m0)
    
    m6 = [mt.mnn, mt.mee, mt.mdd, mt.mne, mt.mnd, mt.med]  # The six MT components
    print m6/mt.scalar_moment()  # normalized MT components

