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
print(m) # print moment tensor
(s1, d1, _), (s2, d2, _) = m.both_strike_dip_rake()  # gives out both nodal pl.
