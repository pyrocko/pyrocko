
import sys
import random
import math
import numpy as num

dynecm = 1e-7

def rotation_from_axis_and_angle(angle, axis):

    if len(axis) == 2:
        theta, phi = axis
        ux = math.sin(theta)*math.cos(phi)
        uy = math.sin(theta)*math.sin(phi)
        uz = math.cos(theta)

    elif len(axis) == 3:
        axis = num.asarray(axis)
        uabs = math.sqrt(num.sum(axis**2))
        ux, uy, uz = axis / uabs
    else:
        assert False

    ct = math.cos(angle)
    st = math.sin(angle)
    return num.matrix([
        [ct + ux**2*(1.-ct), ux*uy*(1.-ct)-uz*st, ux*uz*(1.-ct)+uy*st],
        [uy*ux*(1.-ct)+uz*st, ct+uy**2*(1.-ct), uy*uz*(1.-ct)-ux*st],
        [uz*ux*(1.-ct)-uy*st, uz*uy*(1.-ct)+ux*st, ct+uz**2*(1.-ct)]
        ])

def random_rotation(x=None):

    # after Arvo 1992 - "Fast random rotation matrices"

    if x:
        x1,x2,x3 = x
    else:
        x1,x2,x3 = num.random.random(3)

    phi = math.pi*2.0*x1

    zrot = num.matrix([
            [ math.cos(phi), math.sin(phi), 0.],
            [ -math.sin(phi), math.cos(phi), 0.],
            [ 0., 0., 1.]])

    lam = math.pi*2.0*x2

    v = num.matrix([[ 
        math.cos(lam)*math.sqrt(x3), 
        math.sin(lam)*math.sqrt(x3), 
        math.sqrt(1.-x3) ]]).T

    house = num.identity(3) - 2.0 * v * v.T
    return -house*zrot

def to6(m):
    return num.array([m[0,0], m[1,1], m[2,2], m[0,1], m[0,2], m[1,2]])

def symmat6(*vals):
    '''Create symmetric 3x3 matrix from its 6 non-redundant values.
     
    vals = [ Axx, Ayy, Azz, Axy, Axz, Ayz ]'''
    
    return num.matrix([[vals[0], vals[3], vals[4]],
                       [vals[3], vals[1], vals[5]],
                       [vals[4], vals[5], vals[2]]], dtype=num.float)

def moment_to_magnitude( moment ):
    return num.log10(moment*1.0e7)/1.5 - 10.7
    # global CMT uses 10.7333333... instead of 10.7, based on [Kanamori 1977]
    # 10.7 comes from [Hanks and Kanamori 1979]

def magnitude_to_moment( magnitude ):
    return 10.0**(1.5*(magnitude+10.7))*1.0e-7

magnitude_1Nm = moment_to_magnitude(1.0)
    
def euler_to_matrix( alpha, beta, gamma ):
    '''Given the euler angles alpha,beta,gamma, create rotation matrix
        
    Given coordinate system (x,y,z) and rotated system (xs,ys,zs)
    the line of nodes is the intersection between the x-y and the xs-ys
    planes.
    
    :param alpha: is the angle between the z-axis and the zs-axis.
    :param beta:  is the angle between the x-axis and the line of nodes.
    :param gamma: is the angle between the line of nodes and the xs-axis.

    Usage for moment tensors::

        m_unrot = numpy.matrix([[0,0,-1],[0,0,0],[-1,0,0]])
        euler_to_matrix(dip,strike,-rake, rotmat)
        m = rotmat.T * m_unrot * rotmat

    '''
    
    ca = math.cos(alpha)
    cb = math.cos(beta)
    cg = math.cos(gamma)
    sa = math.sin(alpha)
    sb = math.sin(beta)
    sg = math.sin(gamma)

    mat = num.matrix( [[cb*cg-ca*sb*sg,  sb*cg+ca*cb*sg,  sa*sg],
                       [-cb*sg-ca*sb*cg, -sb*sg+ca*cb*cg, sa*cg],
                       [sa*sb,           -sa*cb,          ca]], dtype=num.float )
    return mat


def matrix_to_euler( rotmat ):
    '''Inverse of euler_to_matrix().'''
    
    ex = cvec(1.,0.,0.)
    ez = cvec(0.,0.,1.)
    exs = rotmat.T * ex
    ezs = rotmat.T * ez
    enodes = num.cross(ez.T,ezs.T).T
    if num.linalg.norm(enodes) < 1e-10:
        enodes = exs
    enodess = rotmat*enodes
    cos_alpha = float((ez.T*ezs))
    if cos_alpha > 1.: cos_alpha = 1.
    if cos_alpha < -1.: cos_alpha = -1.
    alpha = math.acos(cos_alpha)
    beta  = num.mod( math.atan2( enodes[1,0], enodes[0,0] ), math.pi*2. )
    gamma = num.mod( -math.atan2( enodess[1,0], enodess[0,0] ), math.pi*2. )
    
    return unique_euler(alpha,beta,gamma)


def unique_euler( alpha, beta, gamma ):
    '''Uniquify euler angle triplet.
    
Put euler angles into ranges compatible with (dip,strike,-rake) in seismology:
    
    alpha (dip)   : [0, pi/2]
    beta (strike) : [0, 2*pi)
    gamma (-rake) : [-pi, pi)
    
If alpha is near to zero, beta is replaced by beta+gamma and gamma is set to
zero, to prevent that additional ambiguity.

If alpha is near to pi/2, beta is put into the range [0,pi).'''
    
    pi = math.pi 
    
    alpha = num.mod( alpha, 2.0*pi )
    
    if 0.5*pi < alpha and alpha <= pi:
        alpha = pi - alpha
        beta  = beta + pi
        gamma = 2.0*pi - gamma
    elif pi < alpha and alpha <= 1.5*pi:
        alpha = alpha - pi
        gamma = pi - gamma
    elif 1.5*pi < alpha and alpha <= 2.0*pi:
        alpha = 2.0*pi - alpha
        beta  = beta + pi
        gamma = pi + gamma
    
   
    alpha = num.mod( alpha, 2.0*pi )
    beta  = num.mod( beta,  2.0*pi )
    gamma = num.mod( gamma+pi, 2.0*pi )-pi
    
    # If dip is exactly 90 degrees, one is still
    # free to choose between looking at the plane from either side.
    # Choose to look at such that beta is in the range [0,180)
    
    # This should prevent some problems, when dip is close to 90 degrees:
    if abs(alpha - 0.5*pi) < 1e-10: alpha = 0.5*pi
    if abs(beta - pi) < 1e-10: beta = pi
    if abs(beta - 2.*pi) < 1e-10: beta = 0.
    if abs(beta) < 1e-10: beta = 0.
    
    if alpha == 0.5*pi and beta >= pi:
        gamma = - gamma
        beta  = num.mod( beta-pi,  2.0*pi )
        gamma = num.mod( gamma+pi, 2.0*pi )-pi
        assert 0. <= beta < pi
        assert -pi <= gamma < pi
        
    if alpha < 1e-7:
        beta = num.mod(beta + gamma, 2.0*pi)
        gamma = 0.
    
    return (alpha, beta, gamma)

def cvec(x,y,z):
    return num.matrix( [[x,y,z]], dtype=num.float ).T

def rvec(x,y,z):
    return num.matrix( [[x,y,z]], dtype=num.float )

def eigh_check(a):
    evals, evecs = num.linalg.eigh(a)
    assert evals[0] <= evals[1] <= evals[2]
    return evals, evecs

r2d = 180./math.pi
d2r = 1./r2d

    

def sm(m):
    return "/ %5.2F %5.2F %5.2F \\\n" % (m[0,0], m[0,1], m[0,2]) +\
    "| %5.2F %5.2F %5.2F |\n"  % (m[1,0], m[1,1], m[1,2]) +\
    "\\ %5.2F %5.2F %5.2F /\n" % (m[2,0] ,m[2,1], m[2,2])

class MomentTensor:

    _flip_dc = num.matrix( [[0.,0.,-1.],[0.,-1.,0.],[-1.,0.,0.]], dtype=num.float )
    _to_up_south_east = num.matrix( [[0.,0.,-1.],[-1.,0.,0.],[0.,1.,0.]], dtype=num.float ).T
    _m_unrot = num.matrix( [[0.,0.,-1.],[0.,0.,0.],[-1.,0.,0.]], dtype=num.float )
    _u_evals, _u_evecs = eigh_check(_m_unrot)

    @classmethod
    def random_dc(self, x=None, scalar_moment=1.):
        rotmat = random_rotation(x)
        return MomentTensor(m=rotmat * MomentTensor._m_unrot * rotmat.T,
                scalar_moment=scalar_moment)

    def __init__(self, m=None, m_up_south_east=None, strike=0., dip=0., rake=0., scalar_moment=1. ):
        '''Create moment tensor object based on 3x3 moment tensor matrix or orientation of 
           fault plane and scalar moment.
           
        In:
           m -- Matrix in north-east-down convention
           m_up_south_east -- Matrix in up-south-east convention
           strike, dip, rake -- Fault plane angles in [degrees]
           scalar_moment -- Scalar moment in [Nm]
        ''' 
        
        strike = d2r*strike
        dip = d2r*dip
        rake = d2r*rake
        
        if m_up_south_east is not None:
            m = self._to_up_south_east * m_up_south_east * self._to_up_south_east.T
        
        if m is not None:
            m_evals, m_evecs = eigh_check(m)
            rotmat1 = (m_evecs * MomentTensor._u_evecs.T).T
            if num.linalg.det(rotmat1) < 0.:
                rotmat1 *= -1.
            
        else:
            rotmat1 = euler_to_matrix( dip, strike, -rake )
            m = rotmat1.T * MomentTensor._m_unrot * rotmat1 * scalar_moment
            m_evals, m_evecs = eigh_check(m)
        
        self._m = m
        self._m_eigenvals = m_evals
        self._m_eigenvecs = m_evecs
        def cmp_mat(a,b):
            c = 0
            for x,y in zip(a.flat, b.flat):
                c = cmp(abs(x),abs(y))
                if c != 0: return c
            return c

        self._rotmats = sorted( [rotmat1, MomentTensor._flip_dc * rotmat1 ], cmp=cmp_mat )
        
    def both_strike_dip_rake(self):
        '''Get both possible (strike,dip,rake) triplets.'''
        results = []
        for rotmat in self._rotmats:
            alpha, beta, gamma = [ r2d*x for x in matrix_to_euler( rotmat ) ]
            results.append( (beta, alpha, -gamma) )
        
        return results

    def angle(self, other):
        a = self._m_eigenvecs
        b = other._m_eigenvecs

        r = []
        for avec in a:
            c = []
            for bvec in b:
                c.append( float(num.dot(avec, bvec.T)))

            r.append(c)

        r = num.matrix(r)
    
        angles = []
        for r1 in self._rotmats:
            for r2 in other._rotmats:
                r = num.dot(r1.T, r2)
                angles.append( math.acos((float(r.trace()) - 1.0) / 2.0)*r2d )

        a = min(angles)
        return min( a, 180. - a )

    def p_axis(self):
        '''Get direction of p axis.'''
        return (self._m_eigenvecs.T)[0]
        
    def t_axis(self):
        '''Get direction of t axis.'''
        return (self._m_eigenvecs.T)[2]
    
    def null_axis(self):
        return self._m_eigenvecs.T[1]
    
    def eigenvals(self):
        return self._m_eigenvals
    
    def both_slip_vectors(self):
        '''Get both possible slip directions.'''
        return  [rotmat*cvec(1.,0.,0.) for rotmat in self._rotmats ]   
    
    def m(self):
        '''Get plain moment tensor as 3x3 matrix.'''
        return self._m.copy()

    def m6(self):
        return to6(self._m)
    
    def m_up_south_east(self):
        return self._to_up_south_east.T * self._m * self._to_up_south_east

    def m6_up_south_east(self):
        return to6(self.m_up_south_east())
    
    def m_plain_double_couple(self):
        '''Get plain double couple with same scalar moment as moment tensor.'''
        rotmat1 = self._rotmats[0]
        m = rotmat1.T * MomentTensor._m_unrot * rotmat1 * self.scalar_moment()
        return m
    
    def moment_magnitude(self):
        '''Get moment magnitude of moment tensor.'''
        return moment_to_magnitude(self.scalar_moment())
        
    def scalar_moment(self):
        '''Get the scalar moment of the moment tensor.'''
        return num.linalg.norm(self._m_eigenvals)/math.sqrt(2.)
    
    def __str__(self):
        mexp = pow(10,math.ceil(num.log10(num.max(num.abs(self._m)))))
        m = self._m/mexp
        s =  'Scalar Moment: M0 = %g (Mw = %3.1f)\n' 
        s += 'Moment Tensor: Mnn = %6.3f,  Mee = %6.3f, Mdd = %6.3f,\n'
        s += '               Mne = %6.3f,  Mnd = %6.3f, Med = %6.3f    [ x %g ]\n'
        s = s % (self.scalar_moment(), self.moment_magnitude(), m[0,0],m[1,1],m[2,2],m[0,1],m[0,2],m[1,2], mexp)
        
        s += self.str_fault_planes()
        return s
    
    def str_fault_planes(self):
        s = ''
        for i,sdr in enumerate(self.both_strike_dip_rake()):
            s += 'Fault plane %i: strike = %3.0f, dip = %3.0f, slip-rake = %4.0f\n' % \
                 (i+1, sdr[0], sdr[1], sdr[2])
            
        return s

def other_plane( strike, dip, rake ):
    mt = MomentTensor( strike=strike, dip=dip, rake=rake )
    both_sdr = mt.both_strike_dip_rake()
    w = [ sum( [ abs(x-y) for x,y in zip(both_sdr[i], (strike, dip, rake)) ] ) for i in (0,1) ]
    if w[0]<w[1]:
        return both_sdr[1]
    else:
        return both_sdr[0]



