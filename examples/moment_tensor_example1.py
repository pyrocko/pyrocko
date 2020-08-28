from pyrocko import moment_tensor as pmt
import numpy as num

r2d = 180. / num.pi

magnitude = 6.3  # Magnitude of the earthquake

exp = pmt.magnitude_to_moment(magnitude)  # convert the mag to moment in [Nm]

# init pyrocko moment tensor
m = pmt.MomentTensor(
    mnn=2.34*exp,
    mee=-2.64*exp,
    mdd=0.295*exp,
    mne=1.49*exp,
    mnd=0.182*exp,
    med=-0.975*exp)

print(m)  # print moment tensor

# gives out both nodal planes:
(s1, d1, r1), (s2, d2, r2) = m.both_strike_dip_rake()

print('strike1=%g, dip1=%g, rake1=%g' % (s1, d1, r1))
print('strike2=%g, dip2=%g, rake2=%g' % (s2, d2, r2))


# p-axis normal vector in north-east-down coordinates
p_ned = m.p_axis().A[0]

print('p_ned=(%g, %g, %g)' % tuple(p_ned))

# convert to azimuth and dip
p_azimuth = num.arctan2(p_ned[1], p_ned[0]) * r2d
p_dip = num.arcsin(p_ned[2]) * r2d

print('p_azimuth=%g, p_dip=%g' % (p_azimuth, p_dip))
