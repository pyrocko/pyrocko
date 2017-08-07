from pyrocko import moment_tensor as mtm

magnitude = 6.3  # Magnitude of the earthquake

m0 = mtm.magnitude_to_moment(magnitude)  # convert the mag to moment

strike = 130
dip = 40
rake = 110
mt = mtm.MomentTensor(strike=strike, dip=dip, rake=rake, scalar_moment=m0)

m6 = [mt.mnn, mt.mee, mt.mdd, mt.mne, mt.mnd, mt.med]  # The six MT components
print(m6/mt.scalar_moment())  # normalized MT components
