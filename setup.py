import numpy
from distutils.core import setup, Extension
                      
setup (name = 'pyrocko',
       include_dirs = [ numpy.get_include() ],
       version = '0.1',
       description = 'Seismological Processing Unit',
       packages = [ 'pyrocko' ],
       ext_modules = [ Extension('pyrocko/mseed_ext',
                                 include_dirs = ['./libmseed'],
                                 library_dirs = ['./libmseed'],
                                 libraries = ['mseed'],
                                 sources = ['pyrocko/mseed_ext.c']) ])
       
       