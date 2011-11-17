import numpy
from distutils.core import setup, Extension

packname = 'pyrocko' 
subpacknames = [ 'pyrocko.snufflings' ]

setup( name = packname,
    version = '0.2',
    description = 'Seismological Processing Unit',
    packages = [ packname ] + subpacknames,
    ext_modules = [ 
        
        Extension( packname+'/mseed_ext',
            include_dirs = [ numpy.get_include(), './libmseed' ],
            library_dirs = ['./libmseed'],
            libraries = ['mseed'],
            sources = ['pyrocko/mseed_ext.c']),
                
        Extension( packname+'/evalresp_ext',
            include_dirs = [ numpy.get_include(), './evalresp-3.3.0' ],
            library_dirs = ['./evalresp-3.3.0/.libs'],
            libraries = ['evresp'],
            sources = [ packname+'/evalresp_ext.c']),
            
        Extension( packname+'/gse_ext',
            include_dirs = [ numpy.get_include() ],
            sources = [ packname+'/gse_ext.c' ]),

        Extension( packname+'/autopick_ext',
            include_dirs = [ numpy.get_include() ],
            sources = [ packname+'/autopick_ext.c' ]),

    ],
                
    scripts = [ 'apps/snuffler', 'apps/hamster', 'apps/cake' ]
)
