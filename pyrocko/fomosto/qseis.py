import numpy as num
import logging, os, shutil, sys, glob, math, copy, signal, errno

from tempfile import mkdtemp
from subprocess import Popen, PIPE
from os.path import join as pjoin

from guts import *
from guts_array import *
from pyrocko import trace, util, cake
from pyrocko import gf
from pyrocko.parimap import parimap

Timing = gf.meta.Timing

from pyrocko.moment_tensor import MomentTensor, symmat6

from cStringIO import StringIO

logger = logging.getLogger('fomosto.qseis')

# how to call the programs
program_bins = {
    'qseis': 'qseis',
}

qseis_components = 'r t z v'.split()
qseis_greenf_names = ('ex', 'ss', 'ds', 'cl', 'fz', 'fh')

def nextpow2(i):
    return 2**int(math.ceil(math.log(i)/math.log(2.)))

def str_float_vals(vals):
    return ' '.join( [ '%e' % val for val in vals ] )

def str_int_vals(vals):
    return ' '.join( [ '%i' % val for val in vals ] )

def str_str_vals(vals):
    return ' '.join( [ "'%s'" % val for val in vals ] )

def scl(cs):
    if not cs:
        return '\n#'

    return '\n'+' '.join('(%e,%e)' % (c.real, c.imag) for c in cs)

def cake_model_to_config(mod):
    k = 1000.
    srows = []
    for i, row in enumerate(mod.to_scanlines()):
        depth, vp, vs, rho, qp, qs = row
        row = [ depth/k, vp/k, vs/k, rho/k, qp, qs ]
        srows.append( '%i %s' % (i+1, str_float_vals(row)) )
    
    return '\n'.join(srows), len(srows)

class QSeisSourceMech(Object):
    pass

class QSeisSourceMechMT(QSeisSourceMech):
    mnn = Float.T(default=1.0)
    mee = Float.T(default=1.0)
    mdd = Float.T(default=1.0)
    mne = Float.T(default=0.0)
    mnd = Float.T(default=0.0)
    med = Float.T(default=0.0)

    def string_for_config(self):
        return '1 %(mnn)15e %(mee)15e %(mdd)15e ' \
               '%(mne)15e %(med)15e %(mnd)15e ' % self.__dict__

class QSeisSourceMechSDR(QSeisSourceMech):
    m_iso = Float.T(default=0.0)
    m_clvd = Float.T(default=0.0)
    m_dc = Float.T(default=1.0e9)
    strike = Float.T(default=0.0)
    dip = Float.T(default=90.0)
    rake = Float.T(default=0.0)

    def string_for_config(self):
        return '2 %(m_iso)15e %(m_clvd)15e %(m_dc)15e ' \
               '%(strike)15e %(dip)15e %(rake)15e ' % self.__dict__ 

class QSeisPropagationFilter(Object):
    min_depth = Float.T(default=0.0)
    max_depth = Float.T(default=0.0)
    filtered_phase = Int.T(default=0)

    def string_for_config(self):
        return '%(min_depth)15e %(max_depth)15e ' \
               '%(filtered_phase)i' % self.__dict__

class QSeisPoleZeroFilter(Object):
    constant = Float.T(default=1.0)
    poles = List.T(Complex.T())
    zeros = List.T(Complex.T())

    def string_for_config(self):
        
        return '%15e\n%i%s\n%i%s' % (self.constant, 
                len(self.zeros), scl(self.zeros),
                len(self.poles), scl(self.poles))

class QSeisConfig(Object):

    time_region = Tuple.T(2, Timing.T(), default=(
        Timing('-10'), Timing('+890')))

    cut = Tuple.T(2, Timing.T(), optional=True)

    sw_algorithm = Int.T(default=0)
    slowness_window = Tuple.T(4, Float.T(default=0.0))
    wavenumber_sampling = Float.T(default=2.5)
    aliasing_suppression_factor = Float.T(default=0.1)
    
    filter_surface_effects = Int.T(default=0)
    filter_shallow_paths = Int.T(default=0)
    filter_shallow_paths_depth = Float.T(default=0.0)
    propagation_filters = List.T(QSeisPropagationFilter.T())
    receiver_filter = QSeisPoleZeroFilter.T(optional=True)

    sw_flat_earth_transform = Int.T(default=0)

    gradient_resolution_vp = Float.T(default=0.0)
    gradient_resolution_vs = Float.T(default=0.0)
    gradient_resolution_density = Float.T(default=0.0)

    wavelet_duration_samples = Float.T(default=0.0)
    wavelet_type = Int.T(default=2)
    user_wavelet_samples = List.T(Float.T())

    def items(self):
        return dict( self.T.inamevals(self) )

class QSeisConfigFull(QSeisConfig):

    time_start = Float.T(default=0.0)
    time_reduction_velocity = Float.T(default=0.0)
    time_window = Float.T(default=900.0)

    source_depth = Float.T(default=10.0)
    source_mech = QSeisSourceMech.T(default=QSeisSourceMechMT.D())

    receiver_depth = Float.T(default=0.0)
    receiver_distances = List.T(Float.T())
    nsamples = Int.T(default=256)

    gf_sw_source_types = Tuple.T(6, Int.T(),
            default = (1,1,1,1,0,0))

    gf_filenames = Tuple.T(6, String.T(),
            default = qseis_greenf_names)

    seismogram_filename = String.T(default='seis')

    receiver_azimuths = List.T(Float.T())

    earthmodel_1d = gf.meta.Earthmodel1D.T(optional=True)

    @staticmethod
    def example():
        conf = QSeisConfigFull()
        conf.receiver_distances = [ 2000. ]
        conf.receiver_azimuths = [ 0. ]
        conf.time_start = -10.0
        conf.time_reduction_velocity = 15.0
        conf.earthmodel_1d = cake.load_model().extract(depth_max='cmb')
        conf.sw_flat_earth_transform = 1
        return conf
    
    def get_output_filenames(self, rundir):
        return [ pjoin(rundir, self.seismogram_filename+'.t'+c) for c in qseis_components ]

    def string_for_config(self):

        def aggregate(l):
            return len(l), '\n'.join( [''] + [ x.string_for_config() for x in l ] )
        
        assert len(self.receiver_distances) > 0
        assert len(self.receiver_distances) == len(self.receiver_azimuths)
        assert self.earthmodel_1d is not None

        d = self.__dict__.copy()
    
        # fixing these switches here to reduce the amount of wrapper code
        d['sw_distance_unit'] = 1       # always give distances in [km]      
        d['sw_t_reduce'] = 1            # time reduction always as velocity [km/s]
        d['sw_equidistant'] = 0         # always give all distances and azimuths
        d['sw_irregular_azimuths'] = 1

        d['n_distances'] = len(self.receiver_distances)
        d['str_distances'] = str_float_vals(self.receiver_distances)
        d['str_azimuths'] = str_float_vals(self.receiver_azimuths)

        model_str, nlines = cake_model_to_config(self.earthmodel_1d)
        d['n_model_lines'] = nlines
        d['model_lines'] = model_str

        d['str_slowness_window'] = str_float_vals(self.slowness_window)
        d['n_depth_ranges'], d['str_depth_ranges'] = aggregate(self.propagation_filters)

        if self.wavelet_type == 0: # user wavelet
            d['str_w_samples'] = '\n' \
                        + '%i\n' % len(self.user_wavelet_samples) \
                        + str_float_vals(self.user_wavelet_samples)
        else:
            d['str_w_samples'] = ''

        if self.receiver_filter:
            d['str_receiver_filter'] = self.receiver_filter.string_for_config()
        else:
            d['str_receiver_filter'] = '1.0\n0\n#\n0'

        d['str_gf_sw_source_types'] = str_int_vals(self.gf_sw_source_types)
        d['str_gf_filenames'] = str_str_vals(self.gf_filenames)

        d['str_source'] = self.source_mech.string_for_config()

        template = '''# autogenerated QSEIS input by qseis.py
#
# This is the input file of FORTRAN77 program "qseis06" for calculation of
# synthetic seismograms based on a layered halfspace earth model.
#
# by
# Rongjiang  Wang <wang@gfz-potsdam.de>
# GeoForschungsZentrum Potsdam
# Telegrafenberg, D-14473 Potsdam, Germany
#
# Last modified: Potsdam, Nov., 2006
#
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# If not specified, SI Unit System is used overall!
#
# Coordinate systems:
# cylindrical (z,r,t) with z = downward,
#                          r = from source outward,
#                          t = azmuth angle from north to east;
# cartesian (x,y,z) with   x = north,
#                          y = east,
#                          z = downward;
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
#
#	SOURCE PARAMETERS
#	=================
# 1. source depth [km]
#------------------------------------------------------------------------------
 %(source_depth)e                    |dble: source_depth;
#------------------------------------------------------------------------------
#
#	RECEIVER PARAMETERS
#	===================
# 1. receiver depth [km]
# 2. switch for distance sampling role (1/0 = equidistant/irregular); switch
#    for unit used (1/0 = km/deg)
# 3. number of distance samples
# 4. if equidistant, then start and end trace distance (> 0); else distance
#    list (please order the receiver distances from small to large)
# 5. (reduced) time begin [sec] & length of time window [sec], number of time
#    samples (<= 2*nfmax in qsglobal.h)
# 6. switch for unit of the following time reduction parameter: 1 = velocity
#    [km/sec], 0 = slowness [sec/deg]; time reduction parameter
#------------------------------------------------------------------------------
 %(receiver_depth)e                         |dble: receiver_depth;
 %(sw_equidistant)i  %(sw_distance_unit)i   |int: sw_equidistant, sw_d_unit;
 %(n_distances)i                            |int: no_distances;
 %(str_distances)s                          |dble: d_1,d_n; or d_1,d_2, ...(no comments in between!);
 %(time_start)e %(time_window)e %(nsamples)i  |dble: t_start,t_window; int: no_t_samples;
 %(sw_t_reduce)i %(time_reduction_velocity)e  |int: sw_t_reduce; dble: t_reduce;
#------------------------------------------------------------------------------
#
#	WAVENUMBER INTEGRATION PARAMETERS
#	=================================
# 1. select slowness integration algorithm (0 = suggested for full wave-field
#    modelling; 1 or 2 = suggested when using a slowness window with narrow
#    taper range - a technique for suppressing space-domain aliasing);
# 2. 4 parameters for low and high slowness (Note 1) cut-offs [s/km] with
#    tapering: 0 < slw1 < slw2 defining cosine taper at the lower end, and 0 <
#    slw3 < slw4 defining the cosine taper at the higher end. default values
#    will be used in case of inconsistent input of the cut-offs (possibly with
#    much more computational effort);
# 3. parameter for sampling rate of the wavenumber integration (1 = sampled
#    with the spatial Nyquist frequency, 2 = sampled with twice higher than
#    the Nyquist, and so on: the larger this parameter, the smaller the space-
#    domain aliasing effect, but also the more computation effort); 
# 4. the factor for suppressing time domain aliasing (> 0 and <= 1) (Note 2).
#------------------------------------------------------------------------------
 %(sw_algorithm)i                    |int: sw_algorithm;
 %(str_slowness_window)s             |dble: slw(1-4);
 %(wavenumber_sampling)e             |dble: sample_rate;
 %(aliasing_suppression_factor)e     |dble: supp_factor;
#------------------------------------------------------------------------------
#
#	        OPTIONS FOR PARTIAL SOLUTIONS
#       (only applied to the source-site structure)
#	    ===========================================
#
# 1. switch for filtering free surface effects (0 = with free surface, i.e.,
#    do not select this filter; 1 = without free surface; 2 = without free
#    surface but with correction on amplitude and wave form. Note switch 2
#    can only be used for receivers at the surface)
# 2. switch for filtering waves with a shallow penetration depth (concerning
#    their whole trace from source to receiver), penetration depth limit [km]
#
#    if this option is selected, waves whose travel path never exceeds the
#    given depth limit will be filtered ("seismic nuting"). the condition for
#    selecting this filter is that the given shallow path depth limit should
#    be larger than both source and receiver depth.
#
# 3. number of depth ranges where the following selected up/down-sp2oing P or
#    SV waves should be filtered
# 4. the 1. depth range: upper and lower depth [km], switch for filtering P
#    or SV wave in this depth range:
#
#    switch no:              1      2        3       4         other
#    filtered phase:         P(up)  P(down)  SV(up)  SV(down)  Error
#
# 5. the 2. ...
#
#    The partial solution options are useful tools to increase the numerical
#    significance of desired wave phases. Especially when the desired phases
#    are smaller than the undesired phases, these options should be selected
#    and carefully combined.
#------------------------------------------------------------------------------
 %(filter_surface_effects)i                  |int: isurf;
 %(filter_shallow_paths)i %(filter_shallow_paths_depth)e  |int: sw_path_filter; dble:shallow_depth_limit;
 %(n_depth_ranges)i %(str_depth_ranges)s
#------------------------------------------------------------------------------
#
#	SOURCE TIME FUNCTION (WAVELET) PARAMETERS (Note 3)
#	==================================================
# 1. wavelet duration [unit = time sample rather than sec!], that is about
#    equal to the half-amplitude cut-off period of the wavelet (> 0. if <= 0,
#    then default value = 2 time samples will be used), and switch for the
#    wavelet form (0 = user's own wavelet; 1 = default wavelet: normalized
#    square half-sinusoid for simulating a physical delta impulse; 2 = tapered
#    Heaviside wavelet, i.e. integral of wavelet 1)
# 2. IF user's own wavelet is selected, then number of the wavelet time samples
#    (<= 1024), and followed by
# 3. equidistant wavelet time samples
# 4  ...(continue) (! no comment lines allowed between the time sample list!)
#    IF default, delete line 2, 3, 4 ... or comment them out!
#------------------------------------------------------------------------------
 %(wavelet_duration_samples)e %(wavelet_type)i%(str_w_samples)s
#------------------------------------------------------------------------------
#
#	 FILTER PARAMETERS OF RECEIVERS (SEISMOMETERS OR HYDROPHONES)
#	 ============================================================
# 1. constant coefficient (normalization factor)
# 2. number of roots (<= nrootmax in qsglobal.h)
# 3. list of the root positions in the complex format (Re,Im). If no roots,
#    comment out this line
# 4. number of poles (<= npolemax in qsglobal.h)
# 5. list of the pole positions in the complex format (Re,Im). If no poles,
#    comment out this line
#------------------------------------------------------------------------------
 %(str_receiver_filter)s
#------------------------------------------------------------------------------
#
#	OUTPUT FILES FOR GREEN'S FUNCTIONS (Note 4)
#	===========================================
# 1. selections of source types (yes/no = 1/0)
# 2. file names of Green's functions (please give the names without extensions,
#    which will be appended by the program automatically: *.tz, *.tr, *.tt
#    and *.tv are for the vertical, radial, tangential, and volume change (for
#    hydrophones) components, respectively)
#------------------------------------------------------------------------------
#  explosion   strike-slip dip-slip   clvd       single_f_v  single_f_h
#------------------------------------------------------------------------------
 %(str_gf_sw_source_types)s
 %(str_gf_filenames)s
#------------------------------------------------------------------------------
#	OUTPUT FILES FOR AN ARBITRARY POINT DISLOCATION SOURCE
#               (for applications to earthquakes)
#	======================================================
# 1. selection (0 = not selected; 1 or 2 = selected), if (selection = 1), then
#    the 6 moment tensor elements [N*m]: Mxx, Myy, Mzz, Mxy, Myz, Mzx (x is
#    northward, y is eastward and z is downard); else if (selection = 2), then
#    Mis [N*m] = isotropic moment part = (MT+MN+MP)/3, Mcl = CLVD moment part
#    = (2/3)(MT+MP-2*MN), Mdc = double-couple moment part = MT-MN, Strike [deg],
#    Dip [deg] and Rake [deg].
#
#    Note: to use this option, the Green's functions above should be computed
#          (selection = 1) if they do not exist already. 
#
#                 north(x)
#                  /
#                 /\ strike
#                *----------------------->  east(y)
#                |\                       \ 
#                |-\                       \ 
#                |  \     fault plane       \ 
#                |90 \                       \ 
#                |-dip\                       \ 
#                |     \                       \ 
#                |      \                       \
#           downward(z)  \-----------------------\\
#
# 2. switch for azimuth distribution of the stations (0 = uniform azimuth,
#    else = irregular azimuth angles)
# 3. list of the azimuth angles [deg] for all stations given above (if the
#    uniform azimuth is selected, then only one azimuth angle is required)
#
#------------------------------------------------------------------------------
#     Mis        Mcl        Mdc        Strike     Dip        Rake      File
#------------------------------------------------------------------------------
#  2   0.00       1.00       6.0E+19    120.0      30.0       25.0      'seis'
#------------------------------------------------------------------------------
#     Mxx        Myy        Mzz        Mxy        Myz        Mzx       File
#------------------------------------------------------------------------------
%(str_source)s '%(seismogram_filename)s'
%(sw_irregular_azimuths)i
%(str_azimuths)s
#------------------------------------------------------------------------------
#
#	GLOBAL MODEL PARAMETERS (Note 5)
#	================================
# 1. switch for flat-earth-transform
# 2. gradient resolution [%%] of vp, vs, and ro (density), if <= 0, then default
#    values (depending on wave length at cut-off frequency) will be used
#------------------------------------------------------------------------------
 %(sw_flat_earth_transform)i     |int: sw_flat_earth_transform;
 %(gradient_resolution_vp)e %(gradient_resolution_vs)e %(gradient_resolution_density)e   |dble: vp_res, vs_res, ro_res;
#------------------------------------------------------------------------------
#
#	                LAYERED EARTH MODEL
#       (SHALLOW SOURCE + UNIFORM DEEP SOURCE/RECEIVER STRUCTURE)
#	=========================================================
# 1. number of data lines of the layered model (source site) 
#------------------------------------------------------------------------------
 %(n_model_lines)i                   |int: no_model_lines;
#------------------------------------------------------------------------------
#
#	MULTILAYERED MODEL PARAMETERS (source site)
#	===========================================
# no  depth[km]  vp[km/s]  vs[km/s]  ro[g/cm^3] qp      qs
#------------------------------------------------------------------------------
%(model_lines)s
#------------------------------------------------------------------------------
#
#	          LAYERED EARTH MODEL
#       (ONLY THE SHALLOW RECEIVER STRUCTURE)
#       =====================================
# 1. number of data lines of the layered model
#
#    Note: if the number = 0, then the receiver site is the same as the
#          source site, else different receiver-site structure is considered.
#          please be sure that the lowest interface of the receiver-site
#          structure given given below can be found within the source-site
#          structure, too.
#
#------------------------------------------------------------------------------
 0                               |int: no_model_lines;
#------------------------------------------------------------------------------
#
#	MULTILAYERED MODEL PARAMETERS (shallow receiver-site structure)
#	===============================================================
# no  depth[km]    vp[km/s]    vs[km/s]   ro[g/cm^3]   qp      qs
#------------------------------------------------------------------------------
#  1     0.000      2.900       1.676      2.600       92.00     41.00
#  2     2.000      2.900       1.676      2.600       92.00     41.00
#  3     2.000      5.400       3.121      2.600       92.00     41.00
#  4     7.000      5.400       3.121      2.600       92.00     41.00
#  5     7.000      6.160       3.561      2.600      576.00    256.00
#  6    17.000      6.160       3.561      2.600      576.00    256.00
#  7    17.000      6.630       3.832      2.900      576.00    256.00
#  8    35.000      6.630       3.832      2.900      576.00    256.00
#  9    35.000      8.0400      4.4700     3.3198    1340.00    600.00
#---------------------------------end of all inputs----------------------------


Note 1:

The slowness is defined by inverse value of apparent wave velocity = sin(i)/v
with i = incident angle and v = true wave velocity.

Note 2:

The suppression of the time domain aliasing is achieved by using the complex
frequency technique. The suppression factor should be a value between 0 and 1.
If this factor is set to 0.1, for example, the aliasing phase at the reduced
time begin is suppressed to 10%%.

Note 3:

The default basic wavelet function (option 1) is (2/tau)*sin^2(pi*t/tau),
for 0 < t < tau, simulating physical delta impuls. Its half-amplitude cut-off
frequency is 1/tau. To avoid high-frequency noise, tau should not be smaller
than 4-5 time samples.

Note 4:

  Double-Couple   m11/ m22/ m33/ m12/ m23/ m31  Azimuth_Factor_(tz,tr,tv)/(tt)
  ============================================================================
  explosion       1.0/ 1.0/ 1.0/ -- / -- / --       1.0         /   0.0
  strike-slip     -- / -- / -- / 1.0/ -- / --       sin(2*azi)  /   cos(2*azi)
                  1.0/-1.0/ -- / -- / -- / --       cos(2*azi)  /  -sin(2*azi)
  dip-slip        -- / -- / -- / -- / -- / 1.0      cos(azi)    /   sin(azi)
                  -- / -- / -- / -- / 1.0/ --       sin(azi)    /  -cos(azi)
  clvd           -0.5/-0.5/ 1.0/ -- / -- / --       1.0         /   0.0
  ============================================================================
  Single-Force    fx / fy / fz                  Azimuth_Factor_(tz,tr,tv)/(tt)
  ============================================================================
  fz              -- / -- / 1.0                        1.0      /   0.0
  fx              1.0/ -- / --                         cos(azi) /   sin(azi)
  fy              -- / 1.0/ --                         sin(azi) /  -cos(azi)
  ============================================================================

Note 5:

Layers with a constant gradient will be discretized with a number of homogeneous
sublayers. The gradient resolutions are then used to determine the maximum
allowed thickness of the sublayers. If the resolutions of Vp, Vs and Rho
(density) require different thicknesses, the smallest is first chosen. If this
is even smaller than 1%% of the characteristic wavelength, then the latter is
taken finally for the sublayer thickness.
'''
        return template % d

        
class QSeisError(gf.store.StoreError):
    pass


class Interrupted(gf.store.StoreError):
    def __str__(self):
        return 'Interrupted.'

        
class QSeisRunner:
    
    def __init__(self, tmp=None, keep_tmp=False):
            
        self.tempdir = mkdtemp(prefix='qseisrun-', dir=tmp)
        self.keep_tmp = keep_tmp
        self.program = program_bins['qseis']
        self.config = None

    
    def run(self, config):
        self.config = config
        
        input_fn = pjoin(self.tempdir, 'input')
                
        f = open(input_fn, 'w')
        input_str = config.string_for_config()
        
        logger.debug('===== begin qseis input =====\n'
                     '%s===== end qseis input =====' % input_str)
        
        f.write( input_str )
        f.close()
        program = self.program
        
        old_wd = os.getcwd()

        os.chdir(self.tempdir)

        interrupted = []
        
        def signal_handler(signum, frame):
            os.kill(proc.pid, signal.SIGTERM)
            interrupted.append(True)

        original = signal.signal(signal.SIGINT, signal_handler)
        try:
            try:
                proc = Popen(program, stdin=PIPE, stdout=PIPE, stderr=PIPE)

            except OSError:
                os.chdir(old_wd)
                raise QSeisError('could not start qseis: "%s"' % program)
        
            (output_str, error_str) = proc.communicate('input\n')

        finally:
            signal.signal(signal.SIGINT, original)

        if interrupted:
            raise KeyboardInterrupt()
       
        logger.debug('===== begin qseis output =====\n'
                     '%s===== end qseis output =====' % output_str)

        errmess = []
        if proc.returncode != 0:
            errmess.append('qseis had a non-zero exit state: %i' % proc.returncode)
        if error_str:
            errmess.append('qseis emitted something via stderr')
        if output_str.lower().find('error') != -1:
            errmess.append("the string 'error' appeared in qseis output")

        if errmess:
            self.keep_tmp = True

            os.chdir(old_wd)
            raise QSeisError('''
===== begin qseis input =====
%s===== end qseis input =====
===== begin qseis output =====
%s===== end qseis output =====
===== begin qseis error =====
%s===== end qseis error =====
%s
qseis has been invoked as "%s"
in the directory %s'''.lstrip() %
            (input_str, output_str, error_str, '\n'.join(errmess), program, self.tempdir))
            
        
        self.qseis_output = output_str
        self.qseis_error = error_str
        
        os.chdir(old_wd)
        
    def get_traces(self):
       
        fns = self.config.get_output_filenames(self.tempdir)
        traces = []
        distances = self.config.receiver_distances
        azimuths = self.config.receiver_azimuths
        for comp, fn in zip(qseis_components, fns):
            data = num.loadtxt(fn, skiprows=1, dtype=num.float)
            nsamples, ntraces = data.shape
            ntraces -= 1
            vred = self.config.time_reduction_velocity
            deltat = (data[-1,0] - data[0,0])/(nsamples-1)
            for itrace, distance, azimuth in zip(xrange(ntraces), distances, azimuths):
                tmin = self.config.time_start
                if vred != 0.0:
                    tmin += distance / vred

                tmin += deltat

                tr = trace.Trace( '', '%04i' % itrace, '', comp,
                        tmin=tmin, deltat=deltat, ydata=data[:,itrace+1],
                        meta=dict(
                            distance = distance*km,
                            azimuth = azimuth ))
                
                traces.append(tr)
        
        return traces

    def __del__(self):
        if self.tempdir:
            if not self.keep_tmp:
                shutil.rmtree(self.tempdir)
                self.tempdir = None
            else:
                logger.warn('not removing temporary directory: %s' % self.tempdir)

class QSeisGFBuilder(gf.builder.Builder):
    def __init__(self, store_dir, shared, block_size=None, tmp=None ):
        self.gfmapping = [
            (MomentTensor( m=symmat6(1,0,0,1,0,0) ), {'r': (0, +1), 't': (3, +1), 'z': (5, +1) }),
            (MomentTensor( m=symmat6(0,0,0,0,1,1) ), {'r': (1, +1), 't': (4, +1), 'z': (6, +1) }),
            (MomentTensor( m=symmat6(0,0,1,0,0,0) ), {'r': (2, +1),               'z': (7, +1) }),
            (MomentTensor( m=symmat6(0,1,0,0,0,0) ), {'r': (8, +1),               'z': (9, +1) })] 

        self.store = gf.store.Store(store_dir, 'w')

        if block_size is None:
            block_size = (1,1,100)

        if len(self.store.config.ns) == 2:
            block_size = block_size[1:]

        gf.builder.Builder.__init__(self, self.store.config, block_size=block_size)
        baseconf = self.store.get_extra('qseis')

        conf = QSeisConfigFull(**baseconf.items())
        conf.earthmodel_1d = self.store.config.earthmodel_1d
        
        deltat = 1.0/self.gf_set.sample_rate

        if 'time_window_min' not in shared:
            d = self.store.make_timing_params(conf.time_region[0], conf.time_region[1])
            shared['time_window_min'] = d['tlenmax_vred']
            shared['time_start'] = d['tmin_vred']
            shared['time_reduction_velocity'] = d['vred'] / km

        time_window_min = shared['time_window_min']
        conf.time_start = shared['time_start']

        conf.time_reduction_velocity = shared['time_reduction_velocity']

        conf.nsamples = nextpow2(int(round(time_window_min / deltat)) + 1)
        conf.time_window = (conf.nsamples-1)*deltat

        self.qseis_config = conf

        self.tmp = tmp
        if self.tmp is not None:
            util.ensuredir(self.tmp)
        
    def work_block(self, index):
        if len(self.store.config.ns) == 2:
            (sz, firstx), (sz, lastx), (ns, nx) = \
                    self.get_block_extents(index)

            rz = self.store.config.receiver_depth
        else:
            (rz, sz, firstx), (rz, sz, lastx), (nr, ns, nx) = \
                    self.get_block_extents(index)

        gf_filename = '%%s_%gkm_%gkm' % (sz/km, rz/km) 

        conf = copy.deepcopy(self.qseis_config)

        logger.info('Starting block %i / %i' % 
                (index+1, self.nblocks))

        
        conf.source_depth = float(sz/km)
        conf.receiver_depth = float(rz/km)

        runner = QSeisRunner(tmp=self.tmp)

        
        dx = self.gf_set.distance_delta

        distances = num.linspace(firstx, 
                firstx + (nx-1)*dx, nx).tolist()

        if distances[-1] < self.gf_set.distance_max:
            # add global max distance, because qseis does some adjustments with
            # this value
            distances.append(self.gf_set.distance_max)

        conf.gf_sw_source_types = (1,1,1,1,0,0)

        conf.receiver_distances = [ d/km for d in distances ]
        conf.receiver_azimuths = [ 0.0 ] * len(distances)
        
        ii = 1
        for mt, gfmap in self.gfmapping[:[3,4][self.gf_set.ncomponents==10]]:
            m = mt.m()

            f = float

            conf.source_mech = QSeisSourceMechMT(
                mnn = f(m[0,0]), mee = f(m[1,1]), mdd = f(m[2,2]),
                mne = f(m[0,1]), mnd = f(m[0,2]), med = f(m[1,2]))

            runner.run(conf)

            rawtraces = runner.get_traces()

            interrupted = []
            def signal_handler(signum, frame):
                interrupted.append(True)

            original = signal.signal(signal.SIGINT, signal_handler)
            self.store.lock()
            duplicate_inserts = 0
            try:
                for itr, tr in enumerate(rawtraces):
                    if tr.channel not in gfmap:
                        continue
                    
                    x = tr.meta['distance']
                    if x > firstx + (nx-1)*dx:
                        continue

                    ig, factor = gfmap[tr.channel]

                    if len(self.store.config.ns) == 2:
                        args = (sz,x,ig)
                    else:
                        args = (rz,sz,x,ig)

                    if conf.cut:
                        tmin = self.store.t(conf.cut[0], args[:-1])
                        tmax = self.store.t(conf.cut[1], args[:-1])
                        if None in (tmin, tmax):
                            continue

                        tr.chop(tmin, tmax)

                    gf_tr = gf.store.GFTrace.from_trace(tr)
                    gf_tr.data *= factor

                    try:
                        self.store.put(args, gf_tr)
                    except gf.store.DuplicateInsert, e:
                        duplicate_inserts += 1

            finally:
                if duplicate_inserts:
                    logger.warn('%i insertions skipped (duplicates)' %
                            duplicate_inserts)

                self.store.unlock()
                signal.signal(signal.SIGINT, original)

            if interrupted:
                raise KeyboardInterrupt()

            conf.gf_sw_source_types = (0,0,0,0,0,0)
   
        logger.info('Done with block %i / %i' % 
                (index+1, self.nblocks))


km = 1000.

def init(store_dir):
    qseis = QSeisConfig()
    qseis.time_region = (
            gf.meta.Timing('begin-50'),
            gf.meta.Timing('end+100'))

    qseis.cut = (
            gf.meta.Timing('begin-50'),
            gf.meta.Timing('end+100'))

    qseis.wavelet_duration_samples = 0.001
    qseis.sw_flat_earth_transform = 1

    config = gf.meta.ConfigTypeA(
            id = 'my_qseis_gf_store',
            ncomponents = 10,
            sample_rate = 0.2,
            receiver_depth = 0*km,
            source_depth_min = 10*km,
            source_depth_max = 20*km,
            source_depth_delta = 10*km,
            distance_min = 100*km,
            distance_max = 1000*km,
            distance_delta = 10*km,
            earthmodel_1d = cake.load_model().extract(depth_max='cmb'),
            modelling_code_id = 'qseis',
            tabulated_phases = [
                    gf.meta.TPDef(
                        id = 'begin',
                        definition = 'p,P,p\\,P\\,Pv_(cmb)p'),
                    gf.meta.TPDef(
                        id = 'end',
                        definition = '2.5'),
                    gf.meta.TPDef(
                        id = 'P',
                        definition = '!P'),
                    gf.meta.TPDef(
                        id = 'S',
                        definition = '!S'),
                    gf.meta.TPDef(
                        id = 'p',
                        definition = '!p'),
                    gf.meta.TPDef(
                        id = 's',
                        definition = '!s')


            ])

    config.validate()
    return gf.store.Store.create_editables(store_dir, config=config, extra={'qseis': qseis})

def __work_block(args):
    try:
        store_dir, iblock, shared = args
        builder = QSeisGFBuilder(store_dir, shared)
        builder.work_block(iblock)
    except KeyboardInterrupt:
        raise Interrupted()
    except IOError, e:
        if e.errno == errno.EINTR:
            raise Interrupted()
        else:
            raise

    return store_dir, iblock

def build(store_dir, force=False, nworkers=None, continue_=False):

    done = set()
    status_fn = pjoin(store_dir, '.status')
    if not continue_:
        gf.store.Store.create_dependants(store_dir, force)
        with open(status_fn, 'w') as status:
            pass
    else:
        try:
            with open(status_fn, 'r') as status:
                for line in status:
                    done.add(int(line))
        except IOError:
            raise gf.StoreError('nothing to continue')

    shared = {}
    builder = QSeisGFBuilder(store_dir, shared)
    iblocks = [ x for x in builder.all_block_indices() if x not in done ]
    del builder
     
    original = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        for x in parimap(__work_block, [ (store_dir, iblock, shared) for iblock in iblocks ], 
                nprocs=nworkers):

            store_dir, iblock = x
            with open(status_fn, 'a') as status:
                status.write('%i\n' % iblock)

    finally:
        signal.signal(signal.SIGINT, original)

    os.remove(status_fn)

if __name__ == '__main__':

    conf = QSeisConfigFull.example()

    print conf.string_for_config()

    runner = QSeisRunner()
    runner.run(conf)

    traces = runner.get_traces()

    trace.snuffle(traces)


    

