import math

# some awsome colors
tango_colors = {
'butter1': (252, 233,  79),
'butter2': (237, 212,   0),
'butter3': (196, 160,   0),
'chameleon1': (138, 226,  52),
'chameleon2': (115, 210,  22),
'chameleon3': ( 78, 154,   6),
'orange1': (252, 175,  62),
'orange2': (245, 121,   0),
'orange3': (206,  92,   0),
'skyblue1': (114, 159, 207),
'skyblue2': ( 52, 101, 164),
'skyblue3': ( 32,  74, 135),
'plum1': (173, 127, 168),
'plum2': (117,  80, 123),
'plum3': ( 92,  53, 102),
'chocolate1': (233, 185, 110),
'chocolate2': (193, 125,  17),
'chocolate3': (143,  89,   2),
'scarletred1': (239,  41,  41),
'scarletred2': (204,   0,   0),
'scarletred3': (164,   0,   0),
'aluminium1': (238, 238, 236),
'aluminium2': (211, 215, 207),
'aluminium3': (186, 189, 182),
'aluminium4': (136, 138, 133),
'aluminium5': ( 85,  87,  83),
'aluminium6': ( 46,  52,  54)
}

graph_colors = [ tango_colors[_x] for _x in ('scarletred2', 'skyblue3', 'chameleon3', 'orange2', 'plum2', 'chocolate2', 'butter2') ]

        
def color(x=None):
    if x is None:
        return tuple( [ random.randint(0,255) for _x in 'rgb' ] )
        
    if isinstance(x,int):
        if 0 <= x < len(graph_colors):
            return  graph_colors[x]
        else:
            return (0,0,0)
   
    elif isinstance(x,str):
        if x in tango_colors:
            return tango_colors[x]
        
    elif isinstance(x, tuple):
        return x
        
    assert False, "Don't know what to do with this color definition: %s" % x

def nice_value(x):
    '''Round x to nice value.'''
    
    exp = 1.0
    sign = 1
    if x<0.0:
        x = -x
        sign = -1
    while x >= 1.0:
        x /= 10.0
        exp *= 10.0
    while x < 0.1:
        x *= 10.0
        exp /= 10.0
    
    if x >= 0.75:
        return sign * 1.0 * exp
    if x >= 0.375:
        return sign * 0.5 * exp
    if x >= 0.225:
        return sign * 0.25 * exp
    if x >= 0.15:
        return sign * 0.2 * exp
    
    return sign * 0.1 * exp


class AutoScaler:
    '''Tunable 1D autoscaling based on data range.
    
    Instances of this class may be used to determine nice minima, maxima and
    increments for ax annotations, as well as suitable common exponents for
    notation.

    The autoscaling process is guided by the following public attributes
    (default values are given in parantheses):

      approx_ticks (7.0):

        Approximate number of increment steps (tickmarks) to generate.

      mode ('auto'):

        Mode of operation: one of 'auto', 'min-max', '0-max', 'min-0',
        'symmetric' or 'off'.

          'auto':      Look at data range and choose one of the choices below.
          'min-max':   Output range is selected to include data range.
          '0-max':     Output range shall start at zero and end at data max.
          'min-0':     Output range shall start at data min and end at zero. 
          'symmetric': Output range shall by symmetric by zero.
          'off':       Similar to 'min-max', but snap and space are disabled, 
                       such that the output range always exactly matches the 
                       data range.
      
      exp (None):
      
        If defined, override automatically determined exponent for notation by
        the given value.
        
      snap (False):
      
        If set to True, snap output range to multiples of increment. This
        parameter has no effect, if mode is set to 'off'.
        
      inc (None):
      
        If defined, override automatically determined tick increment by the
        given value.
      
      space (0.0):
      
        Add some padding to the range. The value given, is the fraction by which
        the output range is increased on each side. If mode is '0-max' or 'min-0',
        the end at zero is kept fixed at zero. This parameter has no effect if 
        mode is set to 'off'.
      
      exp_factor (3):
        
        Exponent of notation is chosen to be a multiple of this value.
        
      no_exp_interval ((-3,5)):
      
        Range of exponent, for which no exponential notation is allowed.'''
            
    def __init__(self, approx_ticks=7.0, 
                       mode='auto',
                       exp=None,
                       snap=False,
                       inc=None,
                       space=0.0,
                       exp_factor=3,
                       no_exp_interval=(-3,5)):
        
        '''Create new AutoScaler instance.
        
        The parameters are described in the AutoScaler documentation.
        '''
                       
        self.approx_ticks = approx_ticks
        self.mode = mode
        self.exp = exp
        self.snap = snap
        self.inc = inc
        self.space = space
        self.exp_factor = exp_factor
        self.no_exp_interval = no_exp_interval
        
    def make_scale(self, data_range, override_mode=None):
        
        '''Get nice minimum, maximum and increment for given data range.
        
        Returns (minimum,maximum,increment) or (maximum,minimum,-increment),
        depending on whether data_range is (data_min, data_max) or (data_max,
        data_min). If override_mode is defined, the mode attribute is
        temporarily overridden by the given value.
        '''
        
        data_min = min(data_range)
        data_max = max(data_range)
        
        is_reverse = (data_range[0] > data_range[1])
        
        a = self.mode
        if self.mode == 'auto':
            a = self.guess_autoscale_mode( data_min, data_max )
        
        if override_mode is not None:
            a = override_mode
        
        mi, ma = 0, 0
        if a == 'off':
            mi, ma = data_min, data_max
        elif a == '0-max':
            mi = 0.0
            if data_max > 0.0:
                ma = data_max
            else:
                ma = 1.0
        elif a == 'min-0':
            ma = 0.0
            if data_min < 0.0:
                mi = data_min
            else:
                mi = -1.0
        elif a == 'min-max':
            mi, ma = data_min, data_max
        elif a == 'symmetric':
            m = max(abs(data_min),abs(data_max))
            mi = -m
            ma =  m
        
        nmi = mi
        if (mi != 0. or a == 'min-max') and a != 'off':
            nmi = mi - self.space*(ma-mi)
            
        nma = ma
        if (ma != 0. or a == 'min-max') and a != 'off':
            nma = ma + self.space*(ma-mi)
             
        mi, ma = nmi, nma
        
        if mi == ma and a != 'off':
            mi -= 1.0
            ma += 1.0
        
        # make nice tick increment
        if self.inc is not None:
            inc = self.inc
        else:
            if self.approx_ticks > 0.:
                inc = nice_value( (ma-mi)/self.approx_ticks )
            else:
                inc = nice_value( (ma-mi)*10. )
        
        if inc == 0.0:
            inc = 1.0
            
        # snap min and max to ticks if this is wanted
        if self.snap and a != 'off':
            ma = inc * math.ceil(ma/inc)
            mi = inc * math.floor(mi/inc) 
        
        if is_reverse:
            return ma, mi, -inc
        else:
            return mi, ma, inc
        
    def make_exp(self, x):
        '''Get nice exponent for notation of x.
        
        For ax annotations, give tick increment as x.'''
        
        if self.exp is not None: return self.exp
        x = abs(x)
        if x == 0.0: return 0
        if 10**self.no_exp_interval[0] <= x <= 10**self.no_exp_interval[1]: return 0
        return math.floor(math.log10(x)/self.exp_factor)*self.exp_factor
    
    def guess_autoscale_mode(self, data_min, data_max):
        '''Guess mode of operation, based on data range.
        
        Used to map 'auto' mode to '0-max', 'min-0', 'min-max' or 'symmetric'.
        '''
        a = 'min-max'
        if data_min >= 0.0:
            if data_min < data_max/2.:
                a = '0-max'
            else: 
                a = 'min-max'
        if data_max <= 0.0:
            if data_max > data_min/2.:
                a = 'min-0'
            else:
                a = 'min-max'
        if data_min < 0.0 and data_max > 0.0:
            if abs((abs(data_max)-abs(data_min))/(abs(data_max)+abs(data_min))) < 0.5:
                a = 'symmetric'
            else:
                a = 'min-max'
        return a
