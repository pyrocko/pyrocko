# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
from __future__ import absolute_import, division

import numpy as num
import logging
import os
import shutil
import math
import copy
import signal

from tempfile import mkdtemp
from subprocess import Popen, PIPE
import os.path as op
from scipy.integrate import cumtrapz

from pyrocko.moment_tensor import MomentTensor, symmat6
from pyrocko.guts import Float, Int, Tuple, List, Bool, Object, String
from pyrocko import trace, util, cake, gf

km = 1e3

guts_prefix = 'pf'

Timing = gf.meta.Timing


logger = logging.getLogger('pyrocko.fomosto.qseis2d')

# how to call the programs
program_bins = {
    'qseis2d.qseisS2014': 'fomosto_qseisS2014',
    'qseis2d.qseisR2014': 'fomosto_qseisR2014',
}

qseis2d_components = {
    1: 'z r t'.split(),
    2: 'e n u'.split(),
}

# defaults
default_gf_directory = 'qseis2d_green'
default_fk_basefilename = 'green'
default_source_depth = 10.0
default_time_region = (Timing('-10'), Timing('+890'))
default_slowness_window = (0.0, 0.0, 0.2, 0.25)


def have_backend():
    for cmd in [[exe] for exe in program_bins.values()]:
        try:
            p = Popen(cmd, stdout=PIPE, stderr=PIPE, stdin=PIPE)
            (stdout, stderr) = p.communicate()

        except OSError:
            return False

    return True


def nextpow2(i):
    return 2 ** int(math.ceil(math.log(i) / math.log(2.)))


def str_float_vals(vals):
    return ' '.join('%e' % val for val in vals)


def str_int_vals(vals):
    return ' '.join('%i' % val for val in vals)


def str_str_vals(vals):
    return ' '.join("'%s'" % val for val in vals)


def scl(cs):
    if not cs:
        return '\n#'

    return '\n' + ' '.join('(%e,%e)' % (c.real, c.imag) for c in cs)


def cake_model_to_config(mod):
    k = 1000.
    srows = []
    for i, row in enumerate(mod.to_scanlines()):
        depth, vp, vs, rho, qp, qs = row
        row = [depth / k, vp / k, vs / k, rho / k, qp, qs]
        srows.append('%i %s' % (i + 1, str_float_vals(row)))

    return '\n'.join(srows), len(srows)


class QSeis2dSource(Object):
    lat = Float.T(default=10.)
    lon = Float.T(default=0.)
    depth = Float.T(default=default_source_depth)

    def string_for_config(self):
        return '%(lat)e %(lon)15e ' % self.__dict__


class QSeisRSourceMech(Object):
    pass


class QSeisRSourceMechMT(QSeisRSourceMech):
    mnn = Float.T(default=1.0)
    mee = Float.T(default=1.0)
    mdd = Float.T(default=1.0)
    mne = Float.T(default=0.0)
    mnd = Float.T(default=0.0)
    med = Float.T(default=0.0)

    def string_for_config(self):
        return '%(mnn)e %(mee)15e %(mdd)15e ' \
               '%(mne)15e %(med)15e %(mnd)15e ' % self.__dict__


class QSeisSPropagationFilter(Object):
    min_depth = Float.T(default=0.0)
    max_depth = Float.T(default=0.0)
    filtered_phase = Int.T(default=0)

    def string_for_config(self):
        return '%(min_depth)15e %(max_depth)15e ' \
               '%(filtered_phase)i' % self.__dict__


class QSeisRBandpassFilter(Object):
    order = Int.T(default=-1)
    lower_cutoff = Float.T(default=0.1)  # [Hz]
    upper_cutoff = Float.T(default=10.0)

    def string_for_config(self):
        return '%(order) %(lower_cutoff)5e %(upper_cutoff)5e' % self.__dict__


class QSeisSConfig(Object):

    qseiss_version = String.T(default='2014')

    calc_slowness_window = Int.T(default=1)
    slowness_window = Tuple.T(4, optional=True)
    wavenumber_sampling = Float.T(default=2.5)
    aliasing_suppression_factor = Float.T(default=0.01)

    filter_shallow_paths = Int.T(default=0)
    filter_shallow_paths_depth = Float.T(default=0.0)
    propagation_filters = List.T(QSeisSPropagationFilter.T())

    sw_flat_earth_transform = Int.T(default=0)

    gradient_resolution_vp = Float.T(default=0.0)
    gradient_resolution_vs = Float.T(default=0.0)
    gradient_resolution_density = Float.T(default=0.0)

    def items(self):
        return dict(self.T.inamevals(self))


class QSeisSConfigFull(QSeisSConfig):

    time_window = Float.T(default=900.0)

    source_depth = Float.T(default=10.0)

    receiver_basement_depth = Float.T(default=35.0)  # [km]
    receiver_min_distance = Float.T(default=1000.0)  # [km]
    receiver_max_distance = Float.T(default=10000.0)  # [km]
    nsamples = Int.T(default=256)

    info_path = String.T(default='green.info')
    fk_path = String.T(default='green.fk')

    earthmodel_1d = gf.meta.Earthmodel1D.T(optional=True)

    @staticmethod
    def example():
        conf = QSeisSConfigFull()
        conf.source_depth = 15.
        conf.receiver_basement_depth = 35.
        conf.receiver_max_distance = 2000.
        conf.earthmodel_1d = cake.load_model().extract(depth_max='cmb')
        conf.sw_flat_earth_transform = 1
        return conf

    def string_for_config(self):

        def aggregate(xx):
            return len(xx), '\n'.join(
                [''] + [x.string_for_config() for x in xx])

        assert self.earthmodel_1d is not None
        assert self.slowness_window is not None or self.calc_slowness_window,\
            "'slowness window' undefined and 'calc_slowness_window'=0"

        d = self.__dict__.copy()

        model_str, nlines = cake_model_to_config(self.earthmodel_1d)
        d['n_model_lines'] = nlines
        d['model_lines'] = model_str

        if not self.slowness_window:
            d['str_slowness_window'] = str_float_vals(default_slowness_window)
        else:
            d['str_slowness_window'] = str_float_vals(self.slowness_window)

        d['n_depth_ranges'], d['str_depth_ranges'] = \
            aggregate(self.propagation_filters)

        template = '''# autogenerated QSEISS input by qseis2d.py
# This is the input file of FORTRAN77 program "QseisS" for calculation of
# f-k spectra of upgoing seismic waves at the reveiver-site basement.
#
# by
# Rongjiang  Wang <wang@gfz-potsdam.de>
# GeoForschungsZentrum Potsdam
# Telegrafenberg, D-14473 Potsdam, Germany
#
# Modified from qseis2006, Potsdam, Dec. 2014
#
#	SOURCE DEPTH
#	============
# 1. source depth [km]
#------------------------------------------------------------------------------
 %(source_depth)e                   |dble;
#------------------------------------------------------------------------------
#
#	RECEIVER-SITE PARAMETERS
#	========================
# 1. receiver-site basement depth [km]
# 2. max. epicental distance [km]
#------------------------------------------------------------------------------
 %(receiver_basement_depth)e              |dble;
 %(receiver_max_distance)e                |dble;
#------------------------------------------------------------------------------
#	TIME SAMPLING PARAMETERS
#	========================
# 1. length of time window [sec]
# 2. number of time samples (<= 2*nfmax in qsglobal.h)
#------------------------------------------------------------------------------
 %(time_window)e      |dble: t_window;
 %(nsamples)i         |int: no_t_samples;
#------------------------------------------------------------------------------
#	SLOWNESS WINDOW PARAMETERS
#	==========================
# 1. the low and high slowness cut-offs [s/km] with tapering: 0 < slw1 < slw2
#    defining the bounds of cosine taper at the lower end, and 0 < slw3 < slw4
#    defining the bounds of cosine taper at the higher end.
# 2. slowness sampling rate (1.0 = sampling with the Nyquist slowness, 2.0 =
#    sampling with twice higher than the Nyquist slowness, and so on: the
#    larger this parameter, the smaller the space-domain aliasing effect, but
#    also the more computation effort);
# 3. the factor for suppressing time domain aliasing (> 0 and <= 1).
#------------------------------------------------------------------------------
 %(str_slowness_window)s             |dble: slw(1-4);
 %(wavenumber_sampling)e             |dble: sample_rate;
 %(aliasing_suppression_factor)e     |dble: supp_factor;
#------------------------------------------------------------------------------
#	OPTIONS FOR PARTIAL SOLUTIONS
#	=============================
# 1. switch for filtering waves with a shallow penetration depth (concerning
#    their whole trace from source to receiver), penetration depth limit [km]
#    (Note: if this option is selected, waves whose travel path never exceeds
#    the given depth limit will be filtered ("seismic nuting"). the condition
#    for selecting this filter is that the given shallow path depth limit
#    should be larger than both source and receiver depth.)
# 2. number of depth ranges where the following selected up/down-sp2oing P or
#    SV waves should be filtered
# 3. the 1. depth range: upper and lower depth [km], switch for filtering P
#    or SV wave in this depth range:
#    switch no:              1      2        3       4         other
#    filtered phase:         P(up)  P(down)  SV(up)  SV(down)  Error
# 4. the 2. ...
#    (Note: the partial solution options are useful tools to increase the
#    numerical resolution of desired wave phases. especially when the desired
#    phases are much smaller than the undesired phases, these options should
#    be selected and carefully combined.)
#------------------------------------------------------------------------------
 %(filter_shallow_paths)i %(filter_shallow_paths_depth)e
 %(n_depth_ranges)i  %(str_depth_ranges)s               |int, dble, dble, int;
#------------------------------------------------------------------------------
#	OUTPUT FILE FOR F-K SPECTRA of GREEN'S FUNCTIONS
#	================================================
# 1. info-file of Green's functions (ascii including f-k sampling parameters)
# 2. file name of Green's functions (binary files including explosion, strike
#    -slip, dip-slip and clvd sources)
#------------------------------------------------------------------------------
 '%(info_path)s'
 '%(fk_path)s'
#------------------------------------------------------------------------------
#	GLOBAL MODEL PARAMETERS
#	=======================
# 1. switch for flat-earth-transform
# 2. gradient resolution [percent] of vp, vs, and rho (density), if <= 0, then
#    default values (depending on wave length at cut-off frequency) will be
#    used
#------------------------------------------------------------------------------
 %(sw_flat_earth_transform)i     |int: sw_flat_earth_transform;
 %(gradient_resolution_vp)e %(gradient_resolution_vs)e %(gradient_resolution_density)e
#------------------------------------------------------------------------------
#	SOURCE-SITE LAYERED EARTH MODEL
#	===============================
# 1. number of data lines of the layered model
#------------------------------------------------------------------------------
 %(n_model_lines)i                   |int: no_model_lines;
#------------------------------------------------------------------------------
# no  depth[km]  vp[km/s]  vs[km/s] rho[g/cm^3] qp      qs
#------------------------------------------------------------------------------
%(model_lines)s             ;
#-----------------------END OF INPUT PARAMETERS--------------------------------

Glossary

SLOWNESS: The slowness is the inverse of apparent wave velocity = sin(i)/v with
i = incident angle and v = true wave velocity.

SUPPRESSION OF TIME-DOMAIN ALIASING: The suppression of the time domain aliasing
is achieved by using the complex frequency technique. The suppression factor
should be a value between 0 and 1. If this factor is set to 0.1, for example, the
aliasing phase at the reduced time begin is suppressed to 10 percent.

MODEL PARAMETER GRADIENT RESOLUTION: Layers with a constant gradient will be
discretized with a number of homogeneous sublayers. The gradient resolutions are
then used to determine the maximum allowed thickness of the sublayers. If the
resolutions of Vp, Vs and Rho (density) require different thicknesses, the
smallest is first chosen. If this is even smaller than 1 percent of the
characteristic wavelength, then the latter is taken finally for the sublayer
thickness.
'''  # noqa
        return (template % d).encode('ascii')


class QSeisRReceiver(Object):
    lat = Float.T(default=10.0)
    lon = Float.T(default=0.0)
    depth = Float.T(default=0.0)
    tstart = Float.T(default=0.0)
    distance = Float.T(default=0.0)

    def string_for_config(self):
        return '%(lat)e %(lon)15e %(depth)15e' % self.__dict__


class QSeisRConfig(Object):

    qseisr_version = String.T(default='2014')

    receiver_filter = QSeisRBandpassFilter.T(optional=True)

    wavelet_duration = Float.T(default=0.001)     # [s]
    wavelet_type = Int.T(default=1)
    user_wavelet_samples = List.T(Float.T())

    def items(self):
        return dict(self.T.inamevals(self))


class QSeisRConfigFull(QSeisRConfig):

    source = QSeis2dSource(depth=default_source_depth)  # [lat, lon(deg)]
    receiver = QSeisRReceiver()  # [lat, lon(deg)]

    source_mech = QSeisRSourceMech.T(
        optional=True,
        default=QSeisRSourceMechMT.D())

    time_reduction = Float.T(default=0.0)

    info_path = String.T(default='green.info')
    fk_path = String.T(default='green.fk')

    output_format = Int.T(default=1)  # 1/2 components in [Z, R, T]/[E, N, U]
    output_filename = String.T(default='seis.dat')

    earthmodel_receiver_1d = gf.meta.Earthmodel1D.T(optional=True)

    @staticmethod
    def example():
        conf = QSeisRConfigFull()
        conf.source = QSeis2dSource(lat=-80.5, lon=90.1)
        conf.receiver_location = QSeisRReceiver(lat=13.4, lon=240.5, depth=0.0)
        conf.time_reduction = 10.0
        conf.earthmodel_receiver_1d = cake.load_model().extract(
            depth_max='moho')
        return conf

    @property
    def components(self):
        return qseis2d_components[self.output_format]

    def get_output_filename(self, rundir):
        return op.join(rundir, self.output_filename)

    def string_for_config(self):

        def aggregate(xx):
            return len(xx), '\n'.join(
                [''] + [x.string_for_config() for x in xx])

        assert self.earthmodel_receiver_1d is not None

        d = self.__dict__.copy()

#        Actually not existing anymore in code
#        d['sw_surface'] = 0  # 0-free-surface, 1-no fre surface

        d['str_source_location'] = self.source.string_for_config()

        d['str_receiver'] = self.receiver.string_for_config()

        d['str_output_filename'] = "'%s'" % self.output_filename

        model_str, nlines = cake_model_to_config(self.earthmodel_receiver_1d)

        d['n_model_receiver_lines'] = nlines
        d['model_receiver_lines'] = model_str

        if self.wavelet_type == 0:  # user wavelet
            d['str_w_samples'] = '\n' \
                + '%i\n' % len(self.user_wavelet_samples) \
                + str_float_vals(self.user_wavelet_samples)
        else:
            d['str_w_samples'] = ''

        if self.receiver_filter:
            d['str_receiver_filter'] = self.receiver_filter.string_for_config()
        else:
            d['str_receiver_filter'] = '-1  0.0  200.'

        if self.source_mech:
            d['str_source'] = '%s' % (self.source_mech.string_for_config())
        else:
            d['str_source'] = '0'

        template = '''# autogenerated QSEISR input by qseis2d.py
# This is the input file of FORTRAN77 program "QseisR" for calculation of
# synthetic seismograms using the given f-k spectra of incident seismic
# waves at the receiver-site basement, where the f-k spectra should be
# prepared by the "QseisS" code.
#
# by
# Rongjiang  Wang <wang@gfz-potsdam.de>
# GeoForschungsZentrum Potsdam
# Telegrafenberg, D-14473 Potsdam, Germany
#
# Modified from qseis2006, Potsdam, Dec. 2014
#
#------------------------------------------------------------------------------
#	SOURCE PARAMETERS
#	=================
# 1. epicenter (lat[deg], lon[deg])
# 2. moment tensor in N*m: Mxx, Myy, Mzz, Mxy, Myz, Mzx
#    Note: x is northward, y is eastward and z is downard
#          conversion from CMT system:
#          Mxx = Mtt, Myy= Mpp, Mzz = Mrr, Mxy = -Mtp, Myz = -Mrp, Mzx = Mrt
# 3. file of f-k spectra of incident waves
# 4. source duration [s], and selection of source time functions, i.e.,
#    source wavelet (0 = user's own wavelet; 1 = default wavelet: normalized
#    square half-sinusoid)
# 5. if user's own wavelet is selected, then number of the wavelet time samples
#    (<= 1024), followed by
# 6. the equidistant wavelet time series (no comment lines between the time
#    series)
#------------------------------------------------------------------------------
 %(str_source_location)s                        |dble(2);
 %(str_source)s                                |dble(6);
 '%(fk_path)s'                                |char;
 %(wavelet_duration)e %(wavelet_type)i %(str_w_samples)s  |dble, int, dbls;
#------------------------------------------------------------------------------
#	RECEIVER PARAMETERS
#	===================
# 1. station location  (lat[deg], lon[deg], depth[km])
#    Note: the epicentral distance should not exceed the max. distance used
#          for generating the f-k spectra
# 2. time reduction[s]
# 3. order of bandpass filter(if <= 0, then no filter will be used), lower
#    and upper cutoff frequences[Hz]
# 4. selection of output format (1 = Down/Radial/Azimuthal, 2 = East/North/Up)
# 5. output file of velocity seismograms
# 6. number of datalines representing the layered receiver-site structure, and
#    selection of surface condition (0 = free surface, 1 = without free
#    surface, i.e., upper halfspace is replaced by medium of the 1. layer)
# 7. ... layered structure model
#------------------------------------------------------------------------------
 %(str_receiver)s                     |dble(3);
 %(time_reduction)e                                   |dble;
 %(str_receiver_filter)s                                       |int, dble(2);
 %(output_format)i                                            |int;
 %(str_output_filename)s                                       |char;
#------------------------------------------------------------------------------
 %(n_model_receiver_lines)i                    |int: no_model_lines;
#------------------------------------------------------------------------------
#	MULTILAYERED MODEL PARAMETERS (shallow receiver-site structure)
#	===============================================================
# no  depth[km]    vp[km/s]    vs[km/s]   ro[g/cm^3]   qp      qs
#------------------------------------------------------------------------------
%(model_receiver_lines)s
#-----------------------END OF INPUT PARAMETERS--------------------------------
#
#-Requirements to use QSEIS2d: ------------------------------------------------
# (1) Teleseismic body waves with penetration depth much larger than the
#     receiver-site basement depth
# (2) The last layer parameters of the receiver-site structure should be
#     identical with that of the source-site model at the depth which is
#     defined as the common basement depth
# (3) The cutoff frequency should be high enough for separating different
#     wave types.
'''  # noqa
        return (template % d).encode('ascii')


class QSeis2dConfig(Object):
    '''
    Combined config object for QSeisS and QSeisR.

    This config object should contain all settings which cannot be derived from
    the backend-independant Pyrocko GF Store config.
    '''

    qseis_s_config = QSeisSConfig.T(default=QSeisSConfig.D())
    qseis_r_config = QSeisRConfig.T(default=QSeisRConfig.D())
    qseis2d_version = String.T(default='2014')

    time_region = Tuple.T(2, Timing.T(), default=default_time_region)
    cut = Tuple.T(2, Timing.T(), optional=True)
    fade = Tuple.T(4, Timing.T(), optional=True)
    relevel_with_fade_in = Bool.T(default=False)

    gf_directory = String.T(default='qseis2d_green')


class QSeis2dError(gf.store.StoreError):
    pass


class Interrupted(gf.store.StoreError):
    def __str__(self):
        return 'Interrupted.'


class QSeisSRunner(object):
    '''
    Takes QSeis2dConfigFull or QSeisSConfigFull objects, runs the program.
    '''
    def __init__(self, tmp, keep_tmp=False):
        self.tempdir = mkdtemp(prefix='qseisSrun-', dir=tmp)
        self.keep_tmp = keep_tmp
        self.config = None

    def run(self, config):

        if isinstance(config, QSeis2dConfig):
            config = QSeisSConfig(**config.qseis_s_config)

        self.config = config

        input_fn = op.join(self.tempdir, 'input')

        with open(input_fn, 'wb') as f:
            input_str = config.string_for_config()
            logger.debug('===== begin qseisS input =====\n'
                         '%s===== end qseisS input =====' % input_str.decode())
            f.write(input_str)

        program = program_bins['qseis2d.qseisS%s' % config.qseiss_version]

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
                raise QSeis2dError('could not start qseisS: "%s"' % program)

            (output_str, error_str) = proc.communicate(b'input\n')

        finally:
            signal.signal(signal.SIGINT, original)

        if interrupted:
            raise KeyboardInterrupt()

        logger.debug('===== begin qseisS output =====\n'
                     '%s===== end qseisS output =====' % output_str.decode())

        errmess = []
        if proc.returncode != 0:
            errmess.append(
                'qseisS had a non-zero exit state: %i' % proc.returncode)

        if error_str:
            errmess.append('qseisS emitted something via stderr')

        if output_str.lower().find(b'error') != -1:
            errmess.append("the string 'error' appeared in qseisS output")

        if errmess:
            self.keep_tmp = True

            os.chdir(old_wd)
            raise QSeis2dError('''
===== begin qseisS input =====
%s===== end qseisS input =====
===== begin qseisS output =====
%s===== end qseisS output =====
===== begin qseisS error =====
%s===== end qseisS error =====
%s
qseisS has been invoked as "%s"
in the directory %s'''.lstrip() % (
                input_str.decode(),
                output_str.decode(),
                error_str.decode(),
                '\n'.join(errmess), program,
                self.tempdir))

        self.qseiss_output = output_str
        self.qseiss_error = error_str

        os.chdir(old_wd)

    def __del__(self):
        if self.tempdir:
            if not self.keep_tmp:
                shutil.rmtree(self.tempdir)
                self.tempdir = None
            else:
                logger.warn(
                    'not removing temporary directory: %s' % self.tempdir)


class QSeisRRunner(object):
    '''
    Takes QSeis2dConfig or QSeisRConfigFull objects, runs the program and
    reads the output.
    '''
    def __init__(self, tmp=None, keep_tmp=False):
        self.tempdir = mkdtemp(prefix='qseisRrun-', dir=tmp)
        self.keep_tmp = keep_tmp
        self.config = None

    def run(self, config):
        if isinstance(config, QSeis2dConfig):
            config = QSeisRConfigFull(**config.qseis_r_config)

        self.config = config

        input_fn = op.join(self.tempdir, 'input')

        with open(input_fn, 'wb') as f:
            input_str = config.string_for_config()
            old_wd = os.getcwd()
            os.chdir(self.tempdir)
            logger.debug('===== begin qseisR input =====\n'
                         '%s===== end qseisR input =====' % input_str.decode())
            f.write(input_str)

        program = program_bins['qseis2d.qseisR%s' % config.qseisr_version]

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
                raise QSeis2dError('could not start qseisR: "%s"' % program)

            (output_str, error_str) = proc.communicate(b'input\n')

        finally:
            signal.signal(signal.SIGINT, original)

        if interrupted:
            raise KeyboardInterrupt()

        logger.debug('===== begin qseisR output =====\n'
                     '%s===== end qseisR output =====' % output_str.decode())

        errmess = []
        if proc.returncode != 0:
            errmess.append(
                'qseisR had a non-zero exit state: %i' % proc.returncode)

        if error_str:
            errmess.append('qseisR emitted something via stderr')

        if output_str.lower().find(b'error') != -1:
            errmess.append("the string 'error' appeared in qseisR output")

        if errmess:
            self.keep_tmp = True

            os.chdir(old_wd)
            raise QSeis2dError('''
===== begin qseisR input =====
%s===== end qseisR input =====
===== begin qseisR output =====
%s===== end qseisR output =====
===== begin qseisR error =====
%s===== end qseisR error =====
%s
qseisR has been invoked as "%s"
in the directory %s'''.lstrip() % (
                input_str.decode(),
                output_str.decode(),
                error_str.decode(),
                '\n'.join(errmess), program,
                self.tempdir))

        self.qseisr_output = output_str
        self.qseisr_error = error_str

        os.chdir(old_wd)

    def get_traces(self):

        fn = self.config.get_output_filename(self.tempdir)
        data = num.loadtxt(fn, skiprows=1, dtype=float)
        nsamples, ntraces = data.shape
        deltat = (data[-1, 0] - data[0, 0]) / (nsamples - 1)
        toffset = data[0, 0]

        tred = self.config.time_reduction
        rec = self.config.receiver
        tmin = rec.tstart + toffset + deltat + tred

        traces = []
        for itrace, comp in enumerate(self.config.components):
            # qseis2d gives velocity-integrate to displacement
            # integration removes one sample, add it again in front
            displ = cumtrapz(num.concatenate(
                (num.zeros(1), data[:, itrace + 1])), dx=deltat)

            tr = trace.Trace(
                '', '%04i' % itrace, '', comp,
                tmin=tmin, deltat=deltat, ydata=displ,
                meta=dict(distance=rec.distance,
                          azimuth=0.0))

            traces.append(tr)

        return traces

    def __del__(self):
        if self.tempdir:
            if not self.keep_tmp:
                shutil.rmtree(self.tempdir)
                self.tempdir = None
            else:
                logger.warn(
                    'not removing temporary directory: %s' % self.tempdir)


class QSeis2dGFBuilder(gf.builder.Builder):
    nsteps = 2

    def __init__(self, store_dir, step, shared, block_size=None, tmp=None,
                 force=False):

        self.store = gf.store.Store(store_dir, 'w')

        storeconf = self.store.config

        if step == 0:
            block_size = (1, 1, storeconf.ndistances)
        else:
            if block_size is None:
                block_size = (1, 1, 1)  # QSeisR does only allow one receiver

        if len(storeconf.ns) == 2:
            block_size = block_size[1:]

        gf.builder.Builder.__init__(
            self, storeconf, step, block_size=block_size)

        baseconf = self.store.get_extra('qseis2d')

        conf_s = QSeisSConfigFull(**baseconf.qseis_s_config.items())
        conf_r = QSeisRConfigFull(**baseconf.qseis_r_config.items())

        conf_s.earthmodel_1d = storeconf.earthmodel_1d
        if storeconf.earthmodel_receiver_1d is not None:
            conf_r.earthmodel_receiver_1d = \
                storeconf.earthmodel_receiver_1d

        else:
            conf_r.earthmodel_receiver_1d = \
                storeconf.earthmodel_1d.extract(
                    depth_max='moho')
            # depth_max=conf_s.receiver_basement_depth*km)

        deltat = 1.0 / self.gf_config.sample_rate

        if 'time_window_min' not in shared:
            d = self.store.make_timing_params(
                baseconf.time_region[0], baseconf.time_region[1],
                force=force)

            shared['time_window_min'] = float(
                    num.ceil(d['tlenmax'] / self.gf_config.sample_rate) *
                    self.gf_config.sample_rate)
            shared['time_reduction'] = d['tmin_vred']

        time_window_min = shared['time_window_min']

        conf_s.nsamples = nextpow2(int(round(time_window_min / deltat)) + 1)
        conf_s.time_window = (conf_s.nsamples - 1) * deltat
        conf_r.time_reduction = shared['time_reduction']

        if step == 0:
            if 'slowness_window' not in shared:
                if conf_s.calc_slowness_window:
                    phases = [
                        storeconf.tabulated_phases[i].phases
                        for i in range(len(
                            storeconf.tabulated_phases))]

                    all_phases = []
                    map(all_phases.extend, phases)

                    mean_source_depth = num.mean((
                        storeconf.source_depth_min,
                        storeconf.source_depth_max))

                    arrivals = conf_s.earthmodel_1d.arrivals(
                        phases=all_phases,
                        distances=num.linspace(
                            conf_s.receiver_min_distance,
                            conf_s.receiver_max_distance,
                            100) * cake.m2d,
                        zstart=mean_source_depth)

                    ps = num.array(
                        [arrivals[i].p for i in range(len(arrivals))])

                    slownesses = ps / (cake.r2d * cake.d2m / km)

                    shared['slowness_window'] = (0.,
                                                 0.,
                                                 1.1 * float(slownesses.max()),
                                                 1.3 * float(slownesses.max()))

                else:
                    shared['slowness_window'] = conf_s.slowness_window

            conf_s.slowness_window = shared['slowness_window']

        self.qseis_s_config = conf_s
        self.qseis_r_config = conf_r
        self.qseis_baseconf = baseconf

        self.tmp = tmp
        if self.tmp is not None:
            util.ensuredir(self.tmp)

        util.ensuredir(baseconf.gf_directory)

    def cleanup(self):
        self.store.close()

    def work_block(self, iblock):
        if len(self.store.config.ns) == 2:
            (sz, firstx), (sz, lastx), (ns, nx) = \
                self.get_block_extents(iblock)

            rz = self.store.config.receiver_depth
        else:
            (rz, sz, firstx), (rz, sz, lastx), (nr, ns, nx) = \
                self.get_block_extents(iblock)

        source_depth = float(sz / km)
        conf_s = copy.deepcopy(self.qseis_s_config)
        conf_r = copy.deepcopy(self.qseis_r_config)

        gf_directory = op.abspath(self.qseis_baseconf.gf_directory)

        fk_path = op.join(gf_directory, 'green_%.3fkm.fk' % source_depth)
        info_path = op.join(gf_directory, 'green_%.3fkm.info' % source_depth)

        conf_s.fk_path = fk_path
        conf_s.info_path = info_path

        conf_r.fk_path = fk_path
        conf_r.info_path = info_path

        if self.step == 0 and os.path.isfile(fk_path):
            logger.info('Skipping step %i / %i, block %i / %i'
                        '(GF already exists)' %
                        (self.step + 1, self.nsteps, iblock + 1, self.nblocks))
            return

        logger.info(
            'Starting step %i / %i, block %i / %i' %
            (self.step + 1, self.nsteps, iblock + 1, self.nblocks))

        dx = self.gf_config.distance_delta
        conf_r.wavelet_duration = 0.001 * self.gf_config.sample_rate

        if self.step == 0:
            conf_s.source_depth = source_depth
            runner = QSeisSRunner(tmp=self.tmp)
            runner.run(conf_s)

        else:
            conf_r.receiver = QSeisRReceiver(lat=90 - firstx * cake.m2d,
                                             lon=180.,
                                             tstart=0.0,
                                             distance=firstx)
            conf_r.source = QSeis2dSource(lat=90 - 0.001 * dx * cake.m2d,
                                          lon=0.0,
                                          depth=source_depth)

            runner = QSeisRRunner(tmp=self.tmp)

            mmt1 = (MomentTensor(m=symmat6(1, 0, 0, 1, 0, 0)),
                    {'r': (0, 1), 't': (3, 1), 'z': (5, 1)})
            mmt2 = (MomentTensor(m=symmat6(0, 0, 0, 0, 1, 1)),
                    {'r': (1, 1), 't': (4, 1), 'z': (6, 1)})
            mmt3 = (MomentTensor(m=symmat6(0, 0, 1, 0, 0, 0)),
                    {'r': (2, 1), 'z': (7, 1)})
            mmt4 = (MomentTensor(m=symmat6(0, 1, 0, 0, 0, 0)),
                    {'r': (8, 1), 'z': (9, 1)})

            gfmapping = [mmt1, mmt2, mmt3, mmt4]

            for mt, gfmap in gfmapping:
                if mt:
                    m = mt.m()
                    f = float
                    conf_r.source_mech = QSeisRSourceMechMT(
                        mnn=f(m[0, 0]), mee=f(m[1, 1]), mdd=f(m[2, 2]),
                        mne=f(m[0, 1]), mnd=f(m[0, 2]), med=f(m[1, 2]))
                else:
                    conf_r.source_mech = None

                if conf_r.source_mech is not None:
                    runner.run(conf_r)

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
                        if x > firstx + (nx - 1) * dx:
                            continue

                        ig, factor = gfmap[tr.channel]

                        if len(self.store.config.ns) == 2:
                            args = (sz, x, ig)
                        else:
                            args = (rz, sz, x, ig)

                        if self.qseis_baseconf.cut:
                            tmin = self.store.t(
                                self.qseis_baseconf.cut[0], args[:-1])
                            tmax = self.store.t(
                                self.qseis_baseconf.cut[1], args[:-1])

                            if None in (tmin, tmax):
                                continue

                            tr.chop(tmin, tmax)

                        tmin = tr.tmin
                        tmax = tr.tmax

                        if self.qseis_baseconf.fade:
                            ta, tb, tc, td = [
                                self.store.t(v, args[:-1])
                                for v in self.qseis_baseconf.fade]

                            if None in (ta, tb, tc, td):
                                continue

                            if not (ta <= tb and tb <= tc and tc <= td):
                                raise QSeis2dError(
                                    'invalid fade configuration')

                            t = tr.get_xdata()
                            fin = num.interp(t, [ta, tb], [0., 1.])
                            fout = num.interp(t, [tc, td], [1., 0.])
                            anti_fin = 1. - fin
                            anti_fout = 1. - fout

                            y = tr.ydata

                            sum_anti_fin = num.sum(anti_fin)
                            sum_anti_fout = num.sum(anti_fout)

                            if sum_anti_fin != 0.0:
                                yin = num.sum(anti_fin * y) / sum_anti_fin
                            else:
                                yin = 0.0

                            if sum_anti_fout != 0.0:
                                yout = num.sum(anti_fout * y) / sum_anti_fout
                            else:
                                yout = 0.0

                            y2 = anti_fin * yin + \
                                fin * fout * y + \
                                anti_fout * yout

                            if self.qseis_baseconf.relevel_with_fade_in:
                                y2 -= yin

                            tr.set_ydata(y2)

                        gf_tr = gf.store.GFTrace.from_trace(tr)
                        gf_tr.data *= factor

                        try:
                            self.store.put(args, gf_tr)
                        except gf.store.DuplicateInsert:
                            duplicate_inserts += 1

                finally:
                    if duplicate_inserts:
                        logger.warn('%i insertions skipped (duplicates)' %
                                    duplicate_inserts)

                    self.store.unlock()
                    signal.signal(signal.SIGINT, original)

                if interrupted:
                    raise KeyboardInterrupt()

            logger.info(
                'Done with step %i / %i, block %i / %i' %
                (self.step + 1, self.nsteps, iblock + 1, self.nblocks))


def init(store_dir, variant):
    if variant is None:
        variant = '2014'

    if variant not in ('2014'):
        raise gf.store.StoreError('unsupported variant: %s' % variant)

    modelling_code_id = 'qseis2d.%s' % variant

    qseis2d = QSeis2dConfig()

    qseis2d.time_region = (
        gf.meta.Timing('begin-50'),
        gf.meta.Timing('end+100'))

    qseis2d.cut = (
        gf.meta.Timing('begin-50'),
        gf.meta.Timing('end+100'))

    qseis2d.qseis_s_config.sw_flat_earth_transform = 1

    store_id = os.path.basename(os.path.realpath(store_dir))

    config = gf.meta.ConfigTypeA(

        id=store_id,
        ncomponents=10,
        sample_rate=0.2,
        receiver_depth=0 * km,
        source_depth_min=10 * km,
        source_depth_max=20 * km,
        source_depth_delta=10 * km,
        distance_min=100 * km,
        distance_max=1000 * km,
        distance_delta=10 * km,
        earthmodel_1d=cake.load_model().extract(depth_max='cmb'),
        modelling_code_id=modelling_code_id,
        tabulated_phases=[
            gf.meta.TPDef(
                id='begin',
                definition='p,P,p\\,P\\,Pv_(cmb)p'),
            gf.meta.TPDef(
                id='end',
                definition='2.5'),
            gf.meta.TPDef(
                id='P',
                definition='!P'),
            gf.meta.TPDef(
                id='S',
                definition='!S'),
            gf.meta.TPDef(
                id='p',
                definition='!p'),
            gf.meta.TPDef(
                id='s',
                definition='!s')])

    config.validate()
    return gf.store.Store.create_editables(
        store_dir, config=config, extra={'qseis2d': qseis2d})


def build(store_dir, force=False, nworkers=None, continue_=False, step=None,
          iblock=None):

    return QSeis2dGFBuilder.build(
        store_dir, force=force, nworkers=nworkers, continue_=continue_,
        step=step, iblock=iblock)
