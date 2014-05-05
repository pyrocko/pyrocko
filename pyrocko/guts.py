'''Lightweight declarative YAML and XML data binding for Python.'''

import yaml
try:
    from yaml import CSafeLoader as SafeLoader, CSafeDumper as SafeDumper
except:
    from yaml import SafeLoader, SafeDumper

import datetime, calendar, re, sys, time, math, types
from itertools import izip
from cStringIO import StringIO

g_iprop = 0

g_deferred = {}
g_deferred_content = {}

g_tagname_to_class = {}
g_xmltagname_to_class = {}

guts_types = ['Object', 'SObject', 'String', 'Unicode', 'Int', 'Float', 'Complex', 'Bool', 
        'Timestamp', 'DateTimestamp', 'StringPattern', 'UnicodePattern', 'StringChoice', 'List', 'Tuple', 'Union', 'Choice', 'Any']

us_to_cc_regex = re.compile(r'([a-z])_([a-z])')
def us_to_cc(s):
    return us_to_cc_regex.sub(lambda pat: pat.group(1)+pat.group(2).upper(), s)

cc_to_us_regex1 = re.compile(r'([a-z])([A-Z]+)([a-z]|$)')
cc_to_us_regex2 = re.compile(r'([A-Z])([A-Z][a-z])')
def cc_to_us(s):
    return cc_to_us_regex2.sub('\\1_\\2', cc_to_us_regex1.sub('\\1_\\2\\3', s)).lower()

re_frac = re.compile(r'\.[1-9]FRAC')
frac_formats = dict([  ('.%sFRAC' % x, '%.'+x+'f') for x in '123456789' ] )

def make_xmltagname_from_name(name):
    return us_to_cc(name)

def make_name_from_xmltagname(xmltagname):
    return cc_to_us(xmltagname)

def make_content_name(name):
    if name.endswith('_list'):
        return name[:-5]
    elif name.endswith('s'):
        return name[:-1]
    else:
        return name

def expand_stream_args(mode):
    def wrap(f):
        '''Decorator to enhance functions taking stream objects.

        Wraps a function f(..., stream, ...) so that it can also be called as
        f(..., filename='myfilename', ...) or as f(..., string='mydata', ...).
        '''

        def g(*args, **kwargs):
            stream = kwargs.pop('stream', None)
            filename = kwargs.pop('filename', None)
            string = kwargs.pop('string', None)

            assert sum( x is not None for x in (stream, filename, string) ) <= 1

            if stream is not None:
                kwargs['stream'] = stream
                return f(*args, **kwargs)

            elif filename is not None:
                stream = open(filename, mode)
                kwargs['stream'] = stream
                retval = f(*args, **kwargs)
                if isinstance(retval, types.GeneratorType):
                    def wrap_generator(gen):
                        try:
                            for x in gen:
                                yield x
                        
                        except GeneratorExit:
                            pass

                        stream.close()

                    return wrap_generator(retval)

                else:
                    stream.close()
                    return retval

            elif string is not None:
                assert mode == 'r', 'Keyword argument string=... cannot be used in dumper function.'
                kwargs['stream'] = StringIO(string)
                return f(*args, **kwargs)
            
            else:
                assert mode == 'w', 'Use keyword argument stream=... or filename=... in loader function.'
                sout = StringIO()
                f(stream=sout, *args, **kwargs)
                return sout.getvalue()


        return g
    return wrap

class FractionalSecondsMissing(Exception):
    '''Exception raised by :py:func:`str_to_time` when the given string lacks
    fractional seconds.'''
    pass

class FractionalSecondsWrongNumberOfDigits(Exception):
    '''Exception raised by :py:func:`str_to_time` when the given string has an incorrect number of digits in the fractional seconds part.'''
    pass

def _endswith_n(s, endings):
    for ix, x in enumerate(endings):
        if s.endswith(x):
            return ix
    return -1

def str_to_time(s, format='%Y-%m-%d %H:%M:%S.OPTFRAC'):
    '''Convert string representing UTC time to floating point system time.
    
    :param s: string representing UTC time
    :param format: time string format
    :returns: system time stamp as floating point value
    
    Uses the semantics of :py:func:`time.strptime` but allows for fractional seconds.
    If the format ends with ``'.FRAC'``, anything after a dot is interpreted as
    fractional seconds. If the format ends with ``'.OPTFRAC'``, the fractional part,
    including the dot is made optional. The latter has the consequence, that the time 
    strings and the format may not contain any other dots. If the format ends
    with ``'.xFRAC'`` where x is 1, 2, or 3, it is ensured, that exactly that
    number of digits are present in the fractional seconds.
    '''
        
    fracsec = 0.
    fixed_endings = '.FRAC', '.1FRAC', '.2FRAC', '.3FRAC'
    
    iend = _endswith_n(format, fixed_endings)
    if iend != -1:
        dotpos = s.rfind('.')
        if dotpos == -1:
            raise FractionalSecondsMissing('string=%s, format=%s' % (s,format))
        
        if iend > 0 and iend != (len(s)-dotpos-1):
            raise FractionalSecondsWrongNumberOfDigits('string=%s, format=%s' % (s,format))
        
        format = format[:-len(fixed_endings[iend])]
        fracsec = float(s[dotpos:])
        s = s[:dotpos]
        
    elif format.endswith('.OPTFRAC'):
        dotpos = s.rfind('.')
        format = format[:-8]
        if dotpos != -1 and len(s[dotpos:]) > 1:
            fracsec = float(s[dotpos:])
        
        if dotpos != -1:
            s = s[:dotpos]
      
    return calendar.timegm(time.strptime(s, format)) + fracsec


def time_to_str(t, format='%Y-%m-%d %H:%M:%S.3FRAC'):
    '''Get string representation for floating point system time.
    
    :param t: floating point system time
    :param format: time string format
    :returns: string representing UTC time
    
    Uses the semantics of :py:func:`time.strftime` but additionally allows 
    for fractional seconds. If *format* contains ``'.xFRAC'``, where ``x`` is a digit between 1 and 9, 
    this is replaced with the fractional part of *t* with ``x`` digits precision.
    '''
    
    if isinstance(format, int):
        format = '%Y-%m-%d %H:%M:%S.'+str(format)+'FRAC'
    
    ts = float(math.floor(t))
    tfrac = t-ts
    
    m = re_frac.search(format)
    if m:
        sfrac = (frac_formats[m.group(0)] % tfrac)
        if sfrac[0] == '1':
            ts += 1.
                        
        format, nsub = re_frac.subn(sfrac[1:], format, 1)
   
    return time.strftime(format, time.gmtime(ts))

class Defer:
    def __init__(self, classname, *args, **kwargs):
        global g_iprop
        if kwargs.get('position', None) is None:
            kwargs['position'] = g_iprop

        g_iprop += 1

        self.classname = classname
        self.args = args
        self.kwargs = kwargs

class TBase(object):

    strict = False
    multivalued = False

    @classmethod
    def init_propertystuff(cls):
        cls.properties = []
        cls.xmltagname_to_name = {}
        cls.xmltagname_to_name_multivalued = {}
        cls.xmltagname_to_class = {}
        cls.content_property = None

    def __init__(self, default=None, optional=False, xmlstyle='element', xmltagname=None, help=None, position=None):

        global g_iprop
        if position is not None:
            self.position = position
        else:
            self.position = g_iprop

        g_iprop += 1
        self._default = default
        self.optional = optional
        self.name = None
        self._xmltagname = xmltagname
        self.parent = None
        self.xmlstyle = xmlstyle
        self.help = help

    def default(self):
        if isinstance(self._default, DefaultMaker):
            return self._default.make()
        else:
            return self._default

    def has_default(self):
        return self._default is not None

    def xname(self):
        if self.name is not None:
            return self.name
        elif self.parent is not None:
            return 'element of %s' % self.parent.xname()
        else:
            return '?'

    def get_xmltagname(self):
        if self._xmltagname is not None:
            return self._xmltagname
        elif self.name:
            return make_xmltagname_from_name(self.name)
        elif self.xmltagname:
            return self.xmltagname
        else:
            assert False

    @classmethod
    def get_property(cls, name):
        for prop in cls.properties:
            if prop.name == name:
                return prop

        raise ValueError()

    @classmethod
    def remove_property(cls, name):

        prop = cls.get_property(name)
        
        if not prop.multivalued:
            del cls.xmltagname_to_class[prop.effective_xmltagname]
            del cls.xmltagname_to_name[prop.effective_xmltagname]
        else:
            del cls.xmltagname_to_class[prop.content_t.effective_xmltagname]
            del cls.xmltagname_to_name_multivalued[prop.content_t.effective_xmltagname]

        if cls.content_property == prop:
            cls.content_property = None

        cls.properties.remove(prop)

        return prop

    @classmethod
    def add_property(cls, name, prop):

        prop.instance = prop
        prop.name = name 

        if isinstance(prop, Choice.T):
            for tc in prop.choices:
                tc.effective_xmltagname = tc.get_xmltagname()
                cls.xmltagname_to_class[tc.effective_xmltagname] = tc.cls
                cls.xmltagname_to_name[tc.effective_xmltagname] = prop.name
        elif not prop.multivalued:
            prop.effective_xmltagname = prop.get_xmltagname()
            cls.xmltagname_to_class[prop.effective_xmltagname] = prop.cls
            cls.xmltagname_to_name[prop.effective_xmltagname] = prop.name
        else:
            prop.content_t.name = make_content_name(prop.name)
            prop.content_t.effective_xmltagname = prop.content_t.get_xmltagname()
            cls.xmltagname_to_class[prop.content_t.effective_xmltagname] = prop.content_t.cls
            cls.xmltagname_to_name_multivalued[prop.content_t.effective_xmltagname] = prop.name

        cls.properties.append(prop)

        cls.properties.sort(key=lambda x: x.position)

        if prop.xmlstyle == 'content':
            cls.content_property = prop

    @classmethod
    def ivals(cls, val):
        for prop in cls.properties:
            yield getattr(val, prop.name)

    @classmethod
    def ipropvals(cls, val):
        for prop in cls.properties:
            yield prop, getattr(val, prop.name)

    @classmethod
    def inamevals(cls, val):
        for prop in cls.properties:
            yield prop.name, getattr(val, prop.name)

    @classmethod
    def ipropvals_to_save(cls, val, xmlmode=False):

        for prop in cls.properties:
            v = getattr(val, prop.name)
            if v is not None and (not (prop.optional or (prop.multivalued and not v)) or prop.default() != v):
                if xmlmode:
                    yield prop, prop.to_save_xml(v) 
                else:
                    yield prop, prop.to_save(v)

    @classmethod
    def inamevals_to_save(cls, val, xmlmode=False):
        for prop, v in cls.ipropvals_to_save(val, xmlmode):
            yield prop.name, v

    @classmethod
    def translate_from_xml(cls, list_of_pairs, strict):
        d = {}
        for k,v in list_of_pairs:
            if k in cls.xmltagname_to_name_multivalued:
                k2 = cls.xmltagname_to_name_multivalued[k]
                if k2 not in d:
                    d[k2] = []

                d[k2].append(v)
            elif k in cls.xmltagname_to_name:
                k2 = cls.xmltagname_to_name[k]
                if k2 in d:
                    raise ArgumentError('Unexpectedly found more than one child element "%s" within "%s".' % (k, cls.tagname))
                d[k2] = v
            elif k is None:
                if cls.content_property:
                    k2 = cls.content_property.name
                    d[k2] = v
            else:
                if strict: 
                    raise ArgumentError('Unexpected child element "%s" found within "%s".' % (k, cls.tagname))

        return d

    def validate(self, val, regularize=False, depth=-1):
        if self.optional and val is None:
            return val

        is_derived = isinstance(val, self.cls)
        is_exact = type(val) == self.cls
        not_ok = not self.strict and not is_derived or self.strict and not is_exact
        
        if not_ok:
            if regularize:
                try:
                    val = self.regularize_extra(val)
                except (RegularizationError, ValueError):
                    raise ValidationError('%s: could not convert "%s" to type %s' % (self.xname(), val, self.cls.__name__))
            else:
                raise ValidationError('%s: "%s" (type: %s) is not of type %s' % (self.xname(), val, type(val), self.cls.__name__))

        validator = self
        if type(val) != self.cls and isinstance(val, self.cls):
            validator = val.T.instance

        validator.validate_extra(val)

        if depth != 0:
            val = validator.validate_children(val, regularize, depth)

        return val

    def regularize_extra(self, val):
        return self.cls(val)

    def validate_extra(self, val):
        pass

    def validate_children(self, val, regularize, depth):
        for prop, propval in self.ipropvals(val):
            newpropval = prop.validate(propval, regularize, depth-1)
            if regularize and (newpropval is not propval):
                setattr(val, prop.name, newpropval)

        return val

    def to_save(self, val):
        return val

    def to_save_xml(self, val):
        return self.to_save(val)

    def extend_xmlelements(self, elems, v):
        if self.multivalued:
            for x in v:
                elems.append((self.content_t.effective_xmltagname, x))
        else:
            elems.append((self.effective_xmltagname, v))

    def deferred(self):
        return []

    def classname_for_help(self):
        if self.dummy_cls in guts_plain_dummy_types:
            return '``%s``' % self.cls.__name__
        else:
            return ':py:class:`%s`' % self.tagname

    @classmethod
    def props_help_string(cls):
        baseprops = []
        for base in cls.dummy_cls.__bases__:
            if hasattr(base, 'T'):
                baseprops.extend(base.T.properties)
            
        l = []
        l.append('')
        for prop in cls.properties:
            if prop in baseprops:
                continue

            descr = [ prop.classname_for_help() ]
            if prop.optional:
                descr.append('*optional*')

            d = prop.default()
            if d is not None:
                descr.append('*default:* ``%s``' % repr(d))

            if prop.help is not None:
                descr.append(prop.help)
            
            l.append('    .. py:attribute:: %s' % prop.name)
            l.append('')
            l.append('      %s' % ', '.join(descr))
            l.append('')

        return '\n'.join(l)

    @classmethod
    def class_help_string(cls):
        return cls.dummy_cls.__doc_template__

    @classmethod
    def class_signature(cls):
        l = []
        for prop in cls.properties:
            d = prop.default()
            if d is not None:
                arg = repr(d)

            elif prop.optional:
                arg = 'None'

            else:
                arg = '...'

            l.append('%s=%s' % (prop.name, arg))

        return '(%s)' % ', '.join(l)

    @classmethod
    def help(cls):
        return cls.props_help_string()

class ObjectMetaClass(type):
    def __new__(meta, classname, bases, class_dict):
        cls = type.__new__(meta, classname, bases, class_dict)
        if classname != 'Object':
            t_class_attr_name = '_%s__T' % classname
            if not hasattr(cls, t_class_attr_name):
                if hasattr(cls, 'T'):
                    class T(cls.T):
                        pass
                else:
                    class T(TBase):
                        pass

                setattr(cls, t_class_attr_name, T)

            T = getattr(cls, t_class_attr_name)

            if cls.dummy_for is not None:
                T.cls = cls.dummy_for
            else:
                T.cls = cls

            T.dummy_cls = cls

            if hasattr(cls, 'xmltagname'):
                T.xmltagname = cls.xmltagname
            else:
                T.xmltagname = classname

            mod = sys.modules[cls.__module__]
            if hasattr(mod, 'guts_prefix'):
                if mod.guts_prefix:
                    T.tagname = mod.guts_prefix + '.' + classname
                else:
                    T.tagname = classname
            else:
                if cls.__module__ != '__main__':
                    T.tagname = cls.__module__ + '.' + classname
                else:
                    T.tagname = classname

            T.classname = classname

            T.init_propertystuff()

            for k in dir(cls):
                prop = getattr(cls, k)

                if k.endswith('__'):
                    k = k[:-2]

                if isinstance(prop, TBase):
                    if prop.deferred():
                        for defer in prop.deferred():
                            g_deferred_content.setdefault(defer.classname[:-2], []).append((prop,defer))
                            g_deferred.setdefault(defer.classname[:-2], []).append((T,k,prop))

                    else:
                        T.add_property(k, prop)

                elif isinstance(prop, Defer):
                    g_deferred.setdefault(prop.classname[:-2], []).append((T,k,prop))

            if classname in g_deferred_content:
                for prop, defer in g_deferred_content[classname]:
                    prop.process_deferred(defer, T(*defer.args, **defer.kwargs))

                del g_deferred_content[classname]

            if classname in g_deferred:
                for (T_, k_, prop_) in g_deferred.get(classname, []):
                    if isinstance(prop_, Defer):
                        prop_ =  T(*prop_.args, **prop_.kwargs)

                    if not prop_.deferred():
                        T_.add_property(k_, prop_)
                
                del g_deferred[classname]


            g_tagname_to_class[T.tagname] = cls
            if hasattr(cls, 'xmltagname'):
                g_xmltagname_to_class[T.xmltagname] = cls

            cls.T = T
            T.instance = T()

            cls.__doc_template__ = cls.__doc__
            cls.__doc__ = T.class_help_string()

            if cls.__doc__ is None:
                cls.__doc__ = 'Undocumented.'

            cls.__doc__ += '\n' + T.props_help_string()

        return cls

class ValidationError(Exception):
    pass

class RegularizationError(Exception):
    pass

class ArgumentError(Exception):
    pass

class DefaultMaker:
    def __init__(self, cls, args, kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
    
    def make(self):
        return self.cls(*self.args, **self.kwargs)

class Object(object):
    __metaclass__ = ObjectMetaClass
    dummy_for = None

    def __init__(self, **kwargs):
        for prop in self.T.properties:
            k = prop.name
            if k in kwargs:
                setattr(self, k, kwargs.pop(k))
            else:
                if not prop.optional and not prop.has_default():
                    raise ArgumentError('Missing argument to %s: %s' % (self.T.tagname, prop.name))
                else:
                    setattr(self, k, prop.default())
        
        if kwargs:
            raise ArgumentError('Invalid argument to %s: %s' % (self.T.tagname, ', '.join(kwargs.keys())))

    @classmethod
    def D(cls, *args, **kwargs):
        return DefaultMaker(cls, args, kwargs)

    def validate(self, regularize=False, depth=-1):
        self.T.instance.validate(self, regularize, depth)

    def regularize(self, depth=-1):
        self.validate(regularize=True, depth=depth)

    def dump(self, stream=None, filename=None, header=False):
        return dump(self, stream=stream, filename=filename, header=header)

    def dump_xml(self, stream=None, filename=None, header=False):
        return dump_xml(self, stream=stream, filename=filename, header=header)

    @classmethod
    def load(cls, stream=None, filename=None, string=None):
        return load(stream=stream, filename=filename, string=string)

    @classmethod
    def load_xml(cls, stream=None, filename=None, string=None):
        return load_xml(stream=stream, filename=filename, string=string)

    def __str__(self):
        return self.dump()


class SObject(Object):
    
    class __T(TBase):
        def regularize_extra(self, val):
            if isinstance(val, basestring):
                return self.cls(val)

            return val

        def to_save(self, val):
            return str(val)

        def to_save_xml(self, val):
            return str(val)


class Any(Object):

    class __T(TBase):
        def validate(self, val, regularize=False, depth=-1):
            if isinstance(val, Object):
                val.validate(regularize, depth)

            return val

class Int(Object):
    dummy_for = int

    class __T(TBase):
        strict = True

        def to_save_xml(self, value):
            return repr(value)

class Float(Object):
    dummy_for = float

    class __T(TBase):
        strict = True

        def to_save_xml(self, value):
            return repr(value)

class Complex(Object):
    dummy_for = complex
    
    class __T(TBase):
        strict = True

        def regularize_extra(self, val):

            if isinstance(val, list) or isinstance(val, tuple):
                assert len(val) == 2
                val = complex(*val)

            elif not isinstance(val, complex):
                val = complex(val)

            return val

        def to_save(self, value):
            return repr(value)

        def to_save_xml(self, value):
            return repr(value)

class Bool(Object):
    dummy_for = bool

    class __T(TBase):
        strict = True

        def regularize_extra(self, val):
            if isinstance(val, basestring):
                if val.lower().strip() in ('0', 'false'):
                    return False

            return bool(val)

class String(Object):
    dummy_for = str

class Unicode(Object):
    dummy_for = unicode

guts_plain_dummy_types = (String, Unicode, Int, Float, Complex, Bool)

class List(Object):
    dummy_for = list

    class __T(TBase):
        multivalued = True

        def __init__(self, content_t=Any.T(), *args, **kwargs):
            TBase.__init__(self, *args, **kwargs)
            assert isinstance(content_t, TBase) or isinstance(content_t, Defer)
            self.content_t = content_t
            self.content_t.parent = self

        def default(self):
            if self._default is not None:
                return self._default
            if self.optional:
                return None
            else:
                return []

        def has_default(self):
            return True

        def validate(self, val, regularize, depth):
            return TBase.validate(self, val, regularize, depth+1)

        def validate_children(self, val, regularize, depth):
            for i, ele in enumerate(val):
                newele = self.content_t.validate(ele, regularize, depth-1)
                if regularize and newele is not ele:
                    val[i] = newele

            return val

        def to_save(self, val):
            return [ self.content_t.to_save(v) for v in val ]

        def to_save_xml(self, val):
            return [ self.content_t.to_save_xml(v) for v in val ]

        def deferred(self):
            if isinstance( self.content_t, Defer):
                return [ self.content_t ]

            return []

        def process_deferred(self, defer, t_inst):
            if defer is self.content_t:
                self.content_t = t_inst

        def classname_for_help(self):
            return '``list`` of %s objects' % self.content_t.classname_for_help()

def make_typed_list_class(t):
    class O(List):
        class __T(List.T):
            def __init__(self, *args, **kwargs):
                List.T.__init__(self, content_t=t.T(), *args, **kwargs)

    return O

class Tuple(Object):
    dummy_for = tuple

    class __T(TBase):
        multivalued = True

        def __init__(self, n=None, content_t=Any.T(), *args, **kwargs):
            TBase.__init__(self, *args, **kwargs)
            assert isinstance(content_t, TBase)
            self.content_t = content_t
            self.content_t.parent = self
            self.n = n

        def default(self):
            if self._default is not None:
                return self._default
            elif self.optional:
                return None
            else:
                if self.n is not None:
                    return tuple( self.content_t.default() for x in xrange(self.n) )
                else:
                    return tuple()


        def has_default(self):
            return True

        def validate(self, val, regularize, depth):
            return TBase.validate(self, val, regularize, depth+1)

        def validate_extra(self, val):
            if self.n is not None and len(val) != self.n:
                raise ValidationError('%s should have length %i' % (self.xname(), self.n))

            return val

        def validate_children(self, val, regularize, depth):
            if not regularize:
                for ele in val:
                    self.content_t.validate(ele, regularize, depth-1)

                return val
            else:
                newval = []
                isnew = False
                for ele in val:
                    newele = self.content_t.validate(ele, regularize, depth-1)
                    newval.append(newele)
                    if newele is not ele:
                        isnew = True
                
                if isnew:
                    return tuple(newval)
                else:
                    return val

        def to_save(self, val):
            return tuple( self.content_t.to_save(v) for v in val )

        def to_save_xml(self, val):
            return [ self.content_t.to_save_xml(v) for v in val ]

        def classname_for_help(self):
            if self.n is not None:
                return '``tuple`` of %i %s objects' % (self.n, self.content_t.classname_for_help())
            else:
                return '``tuple`` of %s objects' % (self.content_t.classname_for_help())

class Timestamp(Object):
    dummy_for = float

    class __T(TBase):

        def regularize_extra(self, val):
            if isinstance(val, datetime.datetime):
                tt = val.utctimetuple()
                val = calendar.timegm(tt) + val.microsecond * 1e-6  

            elif isinstance(val, datetime.date):
                tt = val.timetuple()
                val = float(calendar.timegm(tt))

            elif isinstance(val, str) or isinstance(val, unicode):
                val = val.strip()
                val = re.sub(r'(Z|\+00(:?00)?)$', '', val)
                if val[10] == 'T':
                    val = val.replace('T', ' ', 1)
                val = str_to_time(val)
            
            elif isinstance(val, int):
                val = float(val)

            else:
                raise ValidationError('%s: cannot convert "%s" to float' % (self.xname(), val))

            return val
        
        def to_save(self, val):
            return datetime.datetime.utcfromtimestamp(val)

        def to_save_xml(self, val):
            return datetime.datetime.utcfromtimestamp(val).isoformat()


class DateTimestamp(Object):
    dummy_for = float

    class __T(TBase):

        def regularize_extra(self, val):
            if isinstance(val, datetime.datetime):
                tt = val.utctimetuple()
                val = calendar.timegm(tt) + val.microsecond * 1e-6  

            elif isinstance(val, str) or isinstance(val, unicode):
                val = str_to_time(val, format='%Y-%m-%d')
            
            if not isinstance(val, float):
                val = float(val)

            return val
        
        def to_save(self, val):
            return time_to_str(val, format='%Y-%m-%d')

        def to_save_xml(self, val):
            return time_to_str(val, format='%Y-%m-%d')

class StringPattern(String):

    '''Any ``str`` matching pattern ``%(pattern)s``.'''

    dummy_for = str
    pattern = '.*'

    class __T(TBase):
        def __init__(self, pattern=None, *args, **kwargs):
            TBase.__init__(self, *args, **kwargs)

            if pattern is not None:
                self.pattern = pattern
            else:
                self.pattern = self.dummy_cls.pattern

        def validate_extra(self, val):
            pat = self.pattern
            if not re.search(pat, val):
                raise ValidationError('%s: "%s" does not match pattern %s' % (self.xname(), val, repr(pat)))

        @classmethod
        def class_help_string(cls):
            dcls = cls.dummy_cls
            doc = dcls.__doc_template__ or StringPattern.__doc_template__
            return doc % { 'pattern': repr(dcls.pattern) }


class UnicodePattern(Unicode):

    '''Any ``unicode`` matching pattern ``%(pattern)s``.'''

    dummy_for = unicode
    pattern = '.*'

    class __T(TBase):
        def __init__(self, pattern=None, *args, **kwargs):
            TBase.__init__(self, *args, **kwargs)

            if pattern is not None:
                self.pattern = pattern
            else:
                self.pattern = self.dummy_cls.pattern

        def validate_extra(self, val):
            pat = self.pattern
            if not re.search(pat, val, flags=re.UNICODE):
                raise ValidationError('%s: "%s" does not match pattern %s' % (self.xname(), val, repr(pat)))

        @classmethod
        def class_help_string(cls):
            dcls = cls.dummy_cls
            doc = dcls.__doc_template__ or UnicodePattern.__doc_template__
            return doc % { 'pattern': repr(dcls.pattern) }


class StringChoice(String):

    '''Any ``str`` out of ``%(choices)s``.'''

    dummy_for = str
    choices = []

    class __T(TBase):
        def __init__(self, choices=None, *args, **kwargs):
            TBase.__init__(self, *args, **kwargs)

            if choices is not None:
                self.choices = choices
            else:
                self.choices = self.dummy_cls.choices

        def validate_extra(self, val):
            if val not in self.choices:
                raise ValidationError(
                        '%s: "%s" is not a valid choice out of %s' % 
                        (self.xname(), val, repr(self.choices)))

        @classmethod
        def class_help_string(cls):
            dcls = cls.dummy_cls
            doc = dcls.__doc_template__ or StringChoice.__doc_template__
            return doc % { 'choices': repr(dcls.choices) }
                

# this will not always work...
class Union(Object):
    members = []
    dummy_for = str

    class __T(TBase):
        def __init__(self, members=None, *args, **kwargs):
            TBase.__init__(self, *args, **kwargs)
            if members is not None:
                self.members = members
            else:
                self.members = self.dummy_cls.members

        def validate(self, val, regularize=False, depth=-1):
            assert self.members
            for member in self.members:
                try:
                    return member.validate(val, regularize, depth=depth)
                except ValidationError, e:
                    pass

            raise e


class Choice(Object):
    choices = []

    class __T(TBase):
        def __init__(self, choices=None, *args, **kwargs):
            TBase.__init__(self, *args, **kwargs)
            if choices is not None:
                self.choices = choices
            else:
                self.choices = self.dummy_cls.choices

            self.cls_to_xmltagname = dict((t.cls, t.get_xmltagname()) for t in self.choices)

        def validate(self, val, regularize=False, depth=-1):
            if self.optional and val is None:
                return val

            t = None
            for  tc in self.choices:
                is_derived = isinstance(val, tc.cls)
                is_exact = type(val) == tc.cls
                if not (not tc.strict and not is_derived or tc.strict and not is_exact):
                    t = tc
                    break
            
            if t is None: 
                if regularize:
                    ok = False
                    for tc in self.choices:
                        try:
                            val = tc.regularize_extra(val)
                            ok = True
                            t = tc
                            break
                        except (RegularizationError, ValueError), e:
                            pass

                    if not ok: 
                        raise ValidationError('%s: could not convert "%s" to any type out of (%s)' % (self.xname(), val, ','.join(x.cls.__name__ for x in self.choices)))
                else:
                    raise ValidationError('%s: "%s" (type: %s) is not of any type out of (%s)' % (self.xname(), val, type(val), ','.join(x.cls.__name__ for x in self.choices)))

            validator = t
            if type(val) != t.cls and isinstance(val, t.cls):
                validator = val.T.instance

            validator.validate_extra(val)

            if depth != 0:
                val = validator.validate_children(val, regularize, depth)

            return val
        
        def extend_xmlelements(self, elems, v):
            elems.append((self.cls_to_xmltagname[type(v)], v))


def _dump(object, stream, header=False, _dump_function=yaml.dump):

    if header:
        stream.write('%YAML 1.1\n')
        if isinstance(header, basestring):
            banner = '\n'.join( '# ' + x for x in header.splitlines() )
            stream.write(banner)
            stream.write('\n')

    _dump_function(object, stream=stream, explicit_start=True, Dumper=SafeDumper)

def _dump_all(object, stream, header=True):
    _dump(object, stream=stream, header=header, _dump_function=yaml.dump_all)

def _load(stream):
    return yaml.load(stream=stream, Loader=SafeLoader)

def _load_all(stream):
    return list(yaml.load_all(stream=stream, Loader=SafeLoader))

def _iload_all(stream):
    return yaml.load_all(stream=stream, Loader=SafeLoader)

def multi_representer(dumper, data):
    node = dumper.represent_mapping('!'+data.T.tagname, 
            data.T.inamevals_to_save(data), flow_style=False)

    return node

def multi_constructor(loader, tag_suffix, node):
    tagname = str(tag_suffix)
    cls = g_tagname_to_class[tagname]
    kwargs = dict(loader.construct_mapping(node, deep=True).iteritems())
    o = cls(**kwargs)
    o.validate(regularize=True, depth=1)
    return o

yaml.add_multi_representer(Object, multi_representer, Dumper=SafeDumper)
yaml.add_multi_constructor('!', multi_constructor, Loader=SafeLoader)


class Constructor(object):
    def __init__(self, add_namespace_maps=False, strict=False):
        self.stack = []
        self.queue = []
        self.namespaces = {}
        self.namespaces_rev = {}
        self.add_namespace_maps = add_namespace_maps
        self.strict = strict

    def start_element(self, name, attrs):
        name = name.split()[-1]
        if self.stack and self.stack[-1][1] is not None:
            cls = self.stack[-1][1].T.xmltagname_to_class.get(name, None)
            if cls is not None and (not issubclass(cls, Object) or issubclass(cls, SObject)):
                cls = None
        else:
            cls = g_xmltagname_to_class.get(name, None)

        self.stack.append((name, cls, attrs, [], []))

    def end_element(self, name):
        name = name.split()[-1]
        name, cls, attrs, content2, content1 = self.stack.pop()

        if cls is not None:
            content2.extend( x for x in attrs.iteritems() )
            content2.append( (None, ''.join(content1)) )
            o = cls(**cls.T.translate_from_xml(content2, self.strict))
            o.validate(regularize=True, depth=1)
            if self.add_namespace_maps:
                o.namespace_map = dict(self.namespaces)

            if self.stack and not all(x[1] is None for x in self.stack):
                self.stack[-1][-2].append((name,o))
            else:
                self.queue.append(o)
        else:
            content = [ ''.join(content1) ]
            if self.stack:
                for c in content:
                    self.stack[-1][-2].append((name,c))


    def characters(self, char_content):
        if self.stack:
            self.stack[-1][-1].append(char_content)

    def start_namespace(self, ns, uri):
        assert ns not in self.namespaces
        assert uri not in self.namespaces_rev

        self.namespaces[ns] = uri
        self.namespaces_rev[uri] = ns

    def end_namespace(self, ns):
        del self.namespaces_rev[self.namespaces[ns]]
        del self.namespaces[ns]
    
    def get_queued_elements(self):
        queue = self.queue
        self.queue = []
        return queue 

def _iload_all_xml(stream, bufsize=100000, add_namespace_maps=False, strict=False):
    from xml.parsers.expat import ParserCreate

    parser = ParserCreate(namespace_separator=' ')

    handler = Constructor(add_namespace_maps=add_namespace_maps, strict=strict) 

    parser.StartElementHandler = handler.start_element
    parser.EndElementHandler = handler.end_element
    parser.CharacterDataHandler = handler.characters
    parser.StartNamespaceDeclHandler = handler.start_namespace
    parser.EndNamespaceDeclHandler = handler.end_namespace

    while True:
        data = stream.read(bufsize)
        parser.Parse(data, bool(not data))
        for element in handler.get_queued_elements():
            yield element

        if not data:
            break

def _load_all_xml(*args, **kwargs):
    return list(_iload_all_xml(*args, **kwargs))

def _load_xml(*args, **kwargs):
    g = _iload_all_xml(*args, **kwargs)
    return g.next()


def _dump_all_xml(objects, stream, root_element_name='root', header=True):

    _dump_xml_header(stream, header)

    beg = '<%s>\n' % root_element_name
    end = '</%s>\n' % root_element_name

    stream.write(beg)

    for object in objects:
        _dump_xml(object, stream=stream)
        
    stream.write(end)

def _dump_xml_header(stream, banner=None):

    stream.write('<?xml version="1.0" encoding="UTF-8" ?>\n')
    if isinstance(banner, basestring):
        stream.write('<!-- ')
        stream.write(banner)
        stream.write(' -->\n')


def _dump_xml(obj, stream, depth=0, xmltagname=None, header=False):
    from xml.sax.saxutils import escape, quoteattr
        
    if depth == 0 and header:
        _dump_xml_header(stream, header)

    indent = ' '*depth*2
    if xmltagname is None:
        xmltagname = obj.T.xmltagname

    if isinstance(obj, Object):
        obj.validate(depth=1)
        attrs = []
        allattrs = True
        elems = []
        conent = []
        for prop, v in obj.T.ipropvals_to_save(obj, xmlmode=True):
            if prop.xmlstyle == 'attribute':
                assert not prop.multivalued 
                assert not isinstance(v, Object)
                attrs.append((prop.effective_xmltagname, v))

            elif prop.xmlstyle == 'content':
                assert not prop.multivalued
                assert not isinstance(v, Object)
                elems.append((None, v))

            else:
                prop.extend_xmlelements(elems, v)


        attr_str = ''
        if attrs:
            attr_str = ' ' + ' '.join( '%s=%s' % (k, quoteattr(v)) for (k,v) in attrs )
            
        if not elems:
            stream.write('%s<%s%s />\n' % (indent, xmltagname, attr_str))
        else:
            oneline = len(elems) == 1 and elems[0][0] is None
            stream.write(u'%s<%s%s>%s' % (indent, xmltagname, attr_str, ('\n','')[oneline]))
            for (k,v) in elems:
                if k is None:
                    stream.write('%s' % escape(unicode(v), {'\0': '&#00;'}).encode('utf8'))
                else:
                    _dump_xml(v, stream=stream, depth=depth+1, xmltagname=k)

            stream.write('%s</%s>\n' % ((indent,'')[oneline], xmltagname))
    else:
        stream.write('%s<%s>%s</%s>\n' % (indent, xmltagname, escape(unicode(obj), {'\0': '&#00;'}).encode('utf8'), xmltagname))

def walk(x, typ=None, path=[]):
    if typ is None or isinstance(x, typ):
        yield path, x
    if isinstance(x, Object):
        for (prop, val) in x.T.ipropvals(x):
            if prop.multivalued:
                for iele, ele in enumerate(val):
                    for y in walk(ele, typ, path=path+[ '%s[%i]' % (prop.name, iele)]):
                        yield y
            else:
                for y in walk(val, typ, path=path+[prop.name]):
                    yield y


@expand_stream_args('w')
def dump(*args, **kwargs):
    return _dump(*args, **kwargs)

@expand_stream_args('r')
def load(*args, **kwargs):
    return _load(*args, **kwargs)

def load_string(s, *args, **kwargs):
    return load(string=s, *args, **kwargs)

@expand_stream_args('w')
def dump_all(*args, **kwargs):
    return _dump_all(*args, **kwargs)

@expand_stream_args('r')
def load_all(*args, **kwargs):
    return _load_all(*args, **kwargs)

@expand_stream_args('r')
def iload_all(*args, **kwargs):
    return _iload_all(*args, **kwargs)


@expand_stream_args('w')
def dump_xml(*args, **kwargs):
    return _dump_xml(*args, **kwargs)

@expand_stream_args('r')
def load_xml(*args, **kwargs):
    return _load_xml(*args, **kwargs)

def load_xml_string(s, *args, **kwargs):
    return load_xml(string=s, *args, **kwargs)

@expand_stream_args('w')
def dump_all_xml(*args, **kwargs):
    return _dump_all_xml(*args, **kwargs)

@expand_stream_args('r')
def load_all_xml(*args, **kwargs):
    return _load_all_xml(*args, **kwargs)

@expand_stream_args('r')
def iload_all_xml(*args, **kwargs):
    return _iload_all_xml(*args, **kwargs)


__all__ = guts_types + [ 'guts_types', 'TBase', 'ValidationError', 
        'ArgumentError', 'Defer', 
        'dump', 'load',
        'dump_all', 'load_all', 'iload_all',
        'dump_xml', 'load_xml', 
        'dump_all_xml', 'load_all_xml', 'iload_all_xml',
        'load_string',
        'load_xml_string',
        'make_typed_list_class', 'walk'
        ] 

