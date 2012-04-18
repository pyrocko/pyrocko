import numpy
from distutils.core import setup, Extension
import os, time, sys
from os.path import join as pjoin

class NotInAGitRepos(Exception):
    pass

def git_infos():
    '''Query git about sha1 of last commit and check if there are local modifications.'''

    from subprocess import Popen, PIPE
    import re
    def q(c):
        return Popen(c, stdout=PIPE).communicate()[0]
    
    if not os.path.exists('.git'):
        raise NotInAGitRepos()

    sha1 = q(['git', 'log', '--pretty=oneline', '-n1']).split()[0]
    sha1 = re.sub('[^0-9a-f]', '', sha1)
    sstatus = q(['git', 'status'])
    local_modifications = bool(re.search(r'^#\s+modified:', sstatus, flags=re.M))
    return sha1, local_modifications

def make_info_module(packname, version):
    '''Put version and revision information into file pyrocko/info.py.'''

    sha1, local_modifications = None, None
    combi = '%s-%s' % (packname, version)
    try:
        sha1 , local_modifications = git_infos()
        combi += '-%s' % sha1
        if local_modifications:
            combi += '-modified'

    except (OSError, NotInAGitRepos):
        pass
   
    datestr = time.strftime('%Y-%m-%d_%H:%M:%S')
    combi += '-%s' % datestr

    s = '''# This module is automatically created from setup.py
git_sha1 = %s
local_modifications = %s
version = %s
long_version = %s
installed_date = %s
''' % tuple( [ repr(x) for x in (sha1, local_modifications, version, combi, datestr) ] )
   
    try:
        f = open(pjoin('pyrocko', 'info.py'), 'w')
        f.write(s)
        f.close()
    except:
        pass

packname = 'pyrocko' 
version = '0.2'

subpacknames = [ 'pyrocko.snufflings' ]
if sys.version_info >= (2,5):
    subpacknames.append( 'pyrocko.need_python_2_5' )

make_info_module(packname, version)

setup( name = packname,
    version = version,
    description = 'Seismological Processing Unit',
    author = 'Sebastian Heimann',
    author_email = 'sebastian.heimann@zmaw.de',
    url = 'http://emolch.github.com/pyrocko/',
    packages = [ packname ] + subpacknames,
    ext_package = packname,
    ext_modules = [ 
        
        Extension( 'mseed_ext',
            include_dirs = [ numpy.get_include(), './libmseed' ],
            library_dirs = ['./libmseed'],
            libraries = ['mseed'],
            sources = [ pjoin(packname, 'mseed_ext.c') ]),
                
        Extension( 'evalresp_ext',
            include_dirs = [ numpy.get_include(), './evalresp-3.3.0' ],
            library_dirs = ['./evalresp-3.3.0/.libs'],
            libraries = ['evresp'],
            sources = [ pjoin(packname, 'evalresp_ext.c') ]),
            
        Extension( 'gse_ext',
            include_dirs = [ numpy.get_include() ],
            sources = [ pjoin(packname, 'gse_ext.c') ]),

        Extension( 'autopick_ext',
            include_dirs = [ numpy.get_include() ],
            sources = [ pjoin(packname, 'autopick_ext.c') ]),

    ],
                
    scripts = [ 'apps/snuffler', 'apps/hamster', 'apps/cake' ],
    package_data = { packname: ['data/*.png', 'data/*.html'] },
)

