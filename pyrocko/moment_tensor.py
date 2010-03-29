
import sys
import random
import math
import numpy as num

dynecm = 1e-7

def symmat6(*vals):
    return num.matrix([[vals[0], vals[3], vals[4]],
                       [vals[3], vals[1], vals[5]],
                       [vals[4], vals[5], vals[2]]], dtype=num.float)

def moment_to_magnitude( moment ):
    return num.log10(moment*1.0e7)/1.5 - 10.7

def magnitude_to_moment( magnitude ):
    return 10.0**(1.5*(magnitude+10.7))*1.0e-7
    
def euler_to_matrix( alpha, beta, gamma ):
    '''Given the euler angles alpha,beta,gamma, create rotation matrix
        
Given coordinate system (x,y,z) and rotated system (xs,ys,zs)
the line of nodes is the intersection between the x-y and the xs-ys
planes.
    alpha is the angle between the z-axis and the zs-axis.
    beta is the angle between the x-axis and the line of nodes.
    gamma is the angle between the line of nodes and the xs-axis.

Usage for moment tensors:
    m_unrot = numpy.matrix([[0,0,-1],[0,0,0],[-1,0,0]])
    euler_to_matrix(dip,strike,-rake, rotmat)
    m = rotmat.T * m_unrot * rotmat'''
    
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
    
    alpha (dip)   : [0, pi/2)
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

    def __init__(self, m=None, m_up_south_east=None, strike=0., dip=0., rake=0., scalar_moment=1. ):
        '''Create moment tensor object based on 3x3 moment tensor matrix or orientation of 
           fault plane and scalar moment.''' 
        
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
                
    def p_axis(self):
        '''Get direction of p axis.'''
        return (self._m_eigenvecs.T)[ argmax(self._m_eigenvals) ]
        
    def t_axis(self):
        '''Get direction of t axis.'''
        return (self._m_eigenvecs.T)[ argmin(self._m_eigenvals) ]
            
    def both_slip_vectors(self):
        '''Get both possible slip directions.'''
        return  [rotmat*cvec(1.,0.,0.) for rotmat in self._rotmats ]   
    
    def m(self):
        '''Get plain moment tensor as 3x3 matrix.'''
        return self._m.copy()
    
    def m_up_south_east(self):
        return self._to_up_south_east.T * self._m * self._to_up_south_east
    
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



import unittest
class MomentTensorTestCase( unittest.TestCase ):
    
    def testMagnitudeMoment(self):
        for i in range(1,10):
            mag = float(i)
            assert abs(mag - moment_to_magnitude(magnitude_to_moment(mag))) < 1e-6, \
                'Magnitude to moment to magnitude test failed.'
            
    def testAnyAngles(self):
        '''Check some arbitrary angles.'''
        for i in range(100):
            (s1,d1,r1) = [ r2d*random.random()*10.-5. for j in range(3) ]
            m0 = 1.+random.random()*1.0e20
            self.forwardBackward( s1, d1, r1, m0 )

    def testProblematicAngles(self):
        '''Checks angles close to fractions of pi, which are especially problematic.'''
        for i in range(100):
            # angles close to fractions of pi are especially problematic
            (s1,d1,r1) = [ r2d*random.randint(-16,16)*math.pi/8.+1e-8*random.random()-0.5e-8 for j in range(3) ]
            m0 = 1.+random.random()*1.0e20
            self.forwardBackward( s1, d1, r1, m0 )
                
    def testNonPlainDoubleCouples(self):
        '''Convert random MTs to plain double couples and compare angles.'''
        for i in range(100):
            ms = [ random.random()*1.0e20-0.5e20 for j in range(6) ]
            m = num.matrix([[ ms[0],ms[3],ms[4]],[ms[3],ms[1],ms[5]],[ms[4],ms[5],ms[2]]], dtype=num.float )
            
            m1 = MomentTensor( m=m )
            m_plain = m1.m_plain_double_couple()
            m2 = MomentTensor( m=m_plain )
            
            self.assertAnglesSame( m1, m2 )           
                
    def forwardBackward(self, strike, dip, rake, scalar_moment ):
        m1 = MomentTensor( strike=strike, dip=dip, rake=rake, scalar_moment=scalar_moment )
        m2 = MomentTensor( m=m1.m() )
        self.assertAnglesSame(m1,m2)
        
    def assertAnglesSame(self, m1, m2):
        assert num.all( num.abs(num.array(m1.both_strike_dip_rake()) - 
                                num.array(m2.both_strike_dip_rake())) < 1e-7*100 ), \
            "angles don't match after forward-backward calculation:\nfirst:\n"+str(m1)+ "\nsecond:\n"+str(m2)

    def assertSame(self, a,b, eps, errstr):
        assert num.all(num.abs(num.array(a)-num.array(b)) < eps), errstr

    def testChile(self):
        m_use =  symmat6( 1.040, -0.030, -1.010, 0.227, -1.510, -0.120 )*1e29*dynecm
        mt = MomentTensor(m_up_south_east=m_use)
        sdr = mt.both_strike_dip_rake()
        self.assertSame( sdr[0], (174.,73.,83.), 1., 'chile fail 1')
        self.assertSame( sdr[1], (18.,18.,112.), 1., 'chile fail 2')
        

if __name__ == "__main__":
    unittest.main()
    
    
    