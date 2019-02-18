from __future__ import absolute_import, division, print_function
import sys
import os
import os.path as op
import time
import shutil
import tempfile
try:
    import numpy
except ImportError:
    # numpy might not be available when running install_prerequisites
    class numpy():
        def __init__(self):
            pass

        @classmethod
        def get_include(self):
            return None

from distutils.sysconfig import get_python_inc
from setuptools import setup, Extension, Command
from setuptools.command.build_py import build_py
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install

packname = 'pyrocko'
version = '2019.06.06'


class NotInAGitRepos(Exception):
    pass


def git_infos():
    '''Query git about sha1 of last commit and check if there are local \
       modifications.'''

    from subprocess import Popen, PIPE
    import re

    def q(c):
        return Popen(c, stdout=PIPE).communicate()[0]

    if not op.exists('.git'):
        raise NotInAGitRepos()

    sha1 = q(['git', 'log', '--pretty=oneline', '-n1']).split()[0]
    sha1 = re.sub(br'[^0-9a-f]', '', sha1)
    sha1 = str(sha1.decode('ascii'))
    sstatus = q(['git', 'status', '--porcelain', '-uno'])
    local_modifications = bool(sstatus.strip())
    return sha1, local_modifications


def bash_completions_dir():
    from subprocess import Popen, PIPE

    def q(c):
        return Popen(c, stdout=PIPE).communicate()[0]

    try:
        d = q(['pkg-config', 'bash-completion', '--variable=completionsdir'])
        return d.strip().decode('utf-8')
    except Exception:
        return None


def make_info_module(packname, version):
    '''Put version and revision information into file src/info.py.'''

    sha1, local_modifications = None, None
    combi = '%s-%s' % (packname, version)
    try:
        sha1, local_modifications = git_infos()
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
long_version = %s  # noqa
installed_date = %s
''' % tuple([repr(x) for x in (
        sha1, local_modifications, version, combi, datestr)])

    try:
        f = open(op.join('src', 'info.py'), 'w')
        f.write(s)
        f.close()
    except Exception:
        pass


def make_prerequisites():
    from subprocess import check_call
    try:
        check_call(['sh', 'prerequisites/prerequisites.sh'])
    except Exception:
        sys.exit('error: failed to build the included prerequisites with '
                 '"sh prerequisites/prerequisites.sh"')


def get_readme_paths():
    paths = []

    for (path, dirnames, filenames) in os.walk(
            op.join(op.dirname(__file__), 'src')):
        paths.extend(
            [op.join(path.split('/', 1)[1], fn) for fn in filenames if
             fn == 'README.md'])
    return paths


def get_build_include(lib_name):
    return op.join(op.dirname(op.abspath(__file__)), lib_name)


def find_pyrocko_installs():
    found = []
    seen = set()
    orig_sys_path = sys.path
    for p in sys.path:

        ap = op.abspath(p)
        if ap == op.abspath('.'):
            continue

        if ap in seen:
            continue

        seen.add(ap)

        sys.path = [p]

        try:
            import pyrocko
            dpath = op.dirname(op.abspath(pyrocko.__file__))
            x = (pyrocko.installed_date, p, dpath,
                 pyrocko.long_version)
            found.append(x)
            del sys.modules['pyrocko']
            del sys.modules['pyrocko.info']
        except (ImportError, AttributeError):
            pass

    sys.path = orig_sys_path
    return found


def print_installs(found, file):
    print(
        '\nsys.path configuration is: \n  %s\n' % '\n  '.join(sys.path),
        file=file)

    dates = sorted([xx[0] for xx in found])
    i = 1

    for (installed_date, p, installed_path, long_version) in found:
        oldnew = ''
        if len(dates) >= 2:
            if installed_date == dates[0]:
                oldnew = ' (oldest)'

            if installed_date == dates[-1]:
                oldnew = ' (newest)'

        print('''Pyrocko installation #%i:
  date installed: %s%s
  version: %s
  path: %s
''' % (i, installed_date, oldnew, long_version, installed_path), file=file)
        i += 1


def check_multiple_install():
    found = find_pyrocko_installs()
    e = sys.stderr

    dates = sorted([xx[0] for xx in found])

    if len(found) > 1:
        print_installs(found, e)

    if len(found) > 1:
        print(
            '''Installation #1 is used with default sys.path configuration.

WARNING: Multiple installations of Pyrocko are present on this system.''',
            file=e)
        if found[0][0] != dates[-1]:
            print('WARNING: Not using newest installed version.', file=e)


def check_pyrocko_install_compat():
    found = find_pyrocko_installs()
    if len(found) == 0:
        return

    expected_submodules = ['gui', 'dataset', 'client',
                           'streaming', 'io', 'model',
                           'modelling']

    installed_date, p, install_path, long_version = found[0]

    installed_submodules = [d for d in os.listdir(install_path)
                            if op.isdir(op.join(install_path, d))]

    if not all([ed in installed_submodules for ed in expected_submodules]):

        print_installs(found, sys.stdout)

        print('''\n
###############################################################################

WARNING: Found an old, incompatible, Pyrocko installation!

Please purge the old installation and the 'build' directory before installing
this new version:

    sudo rm -rf '%s' build

###############################################################################
''' % install_path)

        sys.exit(1)


class CheckInstalls(Command):
    description = '''check for multiple installations of Pyrocko'''
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        check_multiple_install()


class Uninstall(Command):
    description = 'delete installations of Pyrocko known to the invoked ' \
                  'Python interpreter'''

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        found = find_pyrocko_installs()
        print_installs(found, sys.stdout)

        if found:
            print('''
Use the following commands to remove the Pyrocko installation(s) known to the
currently running Python interpreter:

  sudo rm -rf build''')

            for _, _, install_path, _ in found:
                print('  sudo rm -rf "%s"' % install_path)

            print()

        else:
            print('''
No Pyrocko installations found with the currently running Python interpreter.
''')


class CustomInstallCommand(install):

    def symlink_interpreter(self):
        if hasattr(self, 'install_scripts') and sys.executable:
            target = os.path.join(self.install_scripts, 'pyrocko-python')
            if os.path.exists(target):
                os.unlink(target)

            os.symlink(sys.executable, target)

    def run(self):
        check_pyrocko_install_compat()
        install.run(self)
        self.symlink_interpreter()
        check_multiple_install()
        bd_dir = bash_completions_dir()
        if bd_dir:
            try:
                shutil.copy('extras/pyrocko', bd_dir)
                print('Installing pyrocko bash_completion to "%s"' % bd_dir)
            except Exception:
                print(
                    'Could not install pyrocko bash_completion to "%s" '
                    '(continuing without)'
                    % bd_dir)


class InstallPrerequisits(Command):
    description = '''install prerequisites with system package manager'''
    user_options = [
        ('force-yes', None, 'Do not ask for confirmation to install')]

    def initialize_options(self):
        self.force_yes = False

    def finalize_options(self):
        pass

    def run(self):

        from subprocess import Popen, PIPE, STDOUT
        import platform

        distribution = platform.linux_distribution()[0].lower().rstrip()

        if distribution == 'ubuntu':
            distribution = 'debian'

        if distribution.startswith('centos'):
            distribution = 'centos'

        fn = 'prerequisites/prerequisites_%s_python%i.sh' % (
                distribution, sys.version_info.major)

        if not self.force_yes:
            try:
                input = raw_input
            except NameError:
                pass

            confirm = input('Execute: %s \n\
proceed? [y/n]' % open(fn, 'r').read())
            if not confirm.lower() == 'y':
                sys.exit(0)

        p = Popen(['sh', fn], stdin=PIPE, stdout=PIPE, stderr=STDOUT,
                  shell=False)

        while p.poll() is None:
            print(p.stdout.readline().decode(
                'ascii', errors='replace').rstrip())

        print(p.stdout.read().decode('ascii', errors='replace'))


class CustomBuildPyCommand(build_py):

    def make_compat_modules(self):
        mapping = [
            ('pyrocko', 'css', ['pyrocko.io.css']),
            ('pyrocko', 'datacube', ['pyrocko.io.datacube']),
            ('pyrocko.fdsn', '__init__', []),
            ('pyrocko.fdsn', 'enhanced_sacpz', ['pyrocko.io.enhanced_sacpz']),
            ('pyrocko.fdsn', 'station', ['pyrocko.io.stationxml']),
            ('pyrocko', 'gcf', ['pyrocko.io.gcf']),
            ('pyrocko', 'gse1', ['pyrocko.io.gse1']),
            ('pyrocko', 'gse2_io_wrap', ['pyrocko.io.gse2']),
            ('pyrocko', 'ims', ['pyrocko.io.ims']),
            ('pyrocko', 'io_common', ['pyrocko.io.io_common']),
            ('pyrocko', 'kan', ['pyrocko.io.kan']),
            ('pyrocko', 'mseed', ['pyrocko.io.mseed']),
            ('pyrocko', 'rdseed', ['pyrocko.io.rdseed']),
            ('pyrocko.fdsn', 'resp', ['pyrocko.io.resp']),
            ('pyrocko', 'sacio', ['pyrocko.io.sac']),
            ('pyrocko', 'segy', ['pyrocko.io.segy']),
            ('pyrocko', 'seisan_response', ['pyrocko.io.seisan_response']),
            ('pyrocko', 'seisan_waveform', ['pyrocko.io.seisan_waveform']),
            ('pyrocko', 'suds', ['pyrocko.io.suds']),
            ('pyrocko', 'yaff', ['pyrocko.io.yaff']),
            ('pyrocko', 'eventdata', ['pyrocko.io.eventdata']),
            ('pyrocko.fdsn', 'ws', ['pyrocko.client.fdsn']),
            ('pyrocko', 'catalog', ['pyrocko.client.catalog']),
            ('pyrocko', 'iris_ws', ['pyrocko.client.iris']),
            ('pyrocko', 'crust2x2', ['pyrocko.dataset.crust2x2']),
            ('pyrocko', 'crustdb', ['pyrocko.dataset.crustdb']),
            ('pyrocko', 'geonames', ['pyrocko.dataset.geonames']),
            ('pyrocko', 'tectonics', ['pyrocko.dataset.tectonics']),
            ('pyrocko', 'topo', ['pyrocko.dataset.topo']),
            ('pyrocko', 'automap', ['pyrocko.plot.automap']),
            ('pyrocko', 'beachball', ['pyrocko.plot.beachball']),
            ('pyrocko', 'cake_plot', ['pyrocko.plot.cake_plot']),
            ('pyrocko', 'gmtpy', ['pyrocko.plot.gmtpy']),
            ('pyrocko', 'hudson', ['pyrocko.plot.hudson']),
            ('pyrocko', 'response_plot', ['pyrocko.plot.response']),
            ('pyrocko', 'snuffling', ['pyrocko.gui.snuffler.snuffling']),
            ('pyrocko', 'pile_viewer', ['pyrocko.gui.snuffler.pile_viewer']),
            ('pyrocko', 'marker', ['pyrocko.gui.snuffler.marker']),
            ('pyrocko', 'snuffler', ['pyrocko.gui.snuffler.snuffler']),
            ('pyrocko', 'gui_util', ['pyrocko.gui.util']),
            ('pyrocko.gui', 'snuffling', ['pyrocko.gui.snuffler.snuffling']),
            ('pyrocko.gui', 'pile_viewer', [
                'pyrocko.gui.snuffler.pile_viewer']),
            ('pyrocko.gui', 'marker', ['pyrocko.gui.snuffler.marker']),
        ]

        for (package, compat_module, import_modules) in mapping:
            module_code = '''
import sys
import pyrocko
if pyrocko.grumpy:
    sys.stderr.write('using renamed pyrocko module: %s.%s\\n')
    sys.stderr.write('           -> should now use: %s\\n\\n')

''' % (package, compat_module, ', '.join(import_modules)) + ''.join(
                ['from %s import *\n' % module for module in import_modules])

            outfile = self.get_module_outfile(
                self.build_lib, package.split('.'), compat_module)

            dir = os.path.dirname(outfile)
            self.mkpath(dir)
            with open(outfile, 'w') as f:
                f.write(module_code)

    def run(self):
        make_info_module(packname, version)
        self.make_compat_modules()
        build_py.run(self)


class CustomBuildExtCommand(build_ext):
    def run(self):
        make_prerequisites()
        build_ext.run(self)


class CustomBuildAppCommand(build_ext):
    def run(self):
        self.make_app()

    def make_app(self):
        import glob
        import os
        import shutil
        from setuptools import setup

        APP = ['apps/snuffler']
        DATA_FILES = []
        OPTIONS = {
            'argv_emulation': True,
            'iconfile': 'src/data/snuffler.icns',
            'packages': 'pyrocko',
            'excludes': [
                'PyQt4.QtDesigner',
                'PyQt4.QtScript',
                'PyQt4.QtScriptTools',
                'PyQt4.QtTest',
                'PyQt4.QtCLucene',
                'PyQt4.QtDeclarative',
                'PyQt4.QtHelp',
                'PyQt4.QtSql',
                'PyQt4.QtTest',
                'PyQt4.QtXml',
                'PyQt4.QtXmlPatterns',
                'PyQt4.QtMultimedia',
                'PyQt4.phonon',
                'matplotlib.tests',
                'matplotlib.testing'],
            'plist': 'src/data/Info.plist'}

        setup(
            app=APP,
            data_files=DATA_FILES,
            options={'py2app': OPTIONS},
            setup_requires=['py2app'],
        )

        # Manually delete files which refuse to be ignored using 'excludes':
        want_delete = glob.glob(
            'dist/snuffler.app/Contents/Frameworks/libvtk*')

        map(os.remove, want_delete)

        want_delete_dir = glob.glob(
            'dist/Snuffler.app/Contents/Resources/lib/python2.7/'
            'matplotlib/test*')
        map(shutil.rmtree, want_delete_dir)


def _check_for_openmp():
    """Check  whether the default compiler supports OpenMP.
    This routine is adapted from pynbody // yt.
    Thanks to Nathan Goldbaum and Andrew Pontzen.
    """
    import distutils.sysconfig
    import subprocess

    tmpdir = tempfile.mkdtemp(prefix='pyrocko')
    compiler = os.environ.get(
      'CC', distutils.sysconfig.get_config_var('CC')).split()[0]

    # Attempt to compile a test script.
    # See http://openmp.org/wp/openmp-compilers/
    tmpfile = op.join(tmpdir, 'check_openmp.c')
    with open(tmpfile, 'w') as f:
        f.write('''
#include <omp.h>
#include <stdio.h>
int main() {
    #pragma omp parallel
    printf("Hello from thread %d", omp_get_thread_num());
}
''')

    try:
        with open(os.devnull, 'w') as fnull:
            exit_code = subprocess.call([compiler, '-fopenmp', '-o%s'
                                         % op.join(tmpdir, 'check_openmp'),
                                        tmpfile],
                                        stdout=fnull, stderr=fnull)
    except OSError:
        exit_code = 1
    finally:
        shutil.rmtree(tmpdir)

    if exit_code == 0:
        print('Continuing your build using OpenMP...')
        return True

    import multiprocessing
    import platform
    if multiprocessing.cpu_count() > 1:
        print('''WARNING
OpenMP support is not available in your default C compiler, even though
your machine has more than one core available.
Some routines in pyrocko are parallelized using OpenMP and these will
only run on one core with your current configuration.
''')
        if platform.uname()[0] == 'Darwin':
            print('''Since you are running on Mac OS, it's likely that the problem here
is Apple's Clang, which does not support OpenMP at all. The easiest
way to get around this is to download the latest version of gcc from
here: http://hpc.sourceforge.net. After downloading, just point the
CC environment variable to the real gcc and OpenMP support should
get enabled automatically. Something like this -
sudo tar -xzf /path/to/download.tar.gz /
export CC='/usr/local/bin/gcc'
python setup.py clean
python setup.py build
''')
    print('Continuing your build without OpenMP...')
    return False


if _check_for_openmp():
    omp_arg = ['-fopenmp']
    omp_lib = ['-lgomp']
else:
    omp_arg = []
    omp_lib = []


subpacknames = [
    'pyrocko.gf',
    'pyrocko.fomosto',
    'pyrocko.fomosto.report',
    'pyrocko.client',
    'pyrocko.apps',
    'pyrocko.io',
    'pyrocko.model',
    'pyrocko.modelling',
    'pyrocko.plot',
    'pyrocko.gui',
    'pyrocko.gui.snuffler',
    'pyrocko.gui.snuffler',
    'pyrocko.gui.snuffler.snufflings',
    'pyrocko.gui.snuffler.snufflings.map',
    'pyrocko.gui.sparrow',
    'pyrocko.gui.sparrow.elements',
    'pyrocko.dataset',
    'pyrocko.dataset.topo',
    'pyrocko.streaming',
    'pyrocko.scenario',
    'pyrocko.scenario.targets',
    'pyrocko.scenario.sources',
    'pyrocko.obspy_compat',
]

entry_points = {
    'console_scripts':
        ['fomosto = pyrocko.apps.fomosto:main',
         'cake = pyrocko.apps.cake:main',
         'automap = pyrocko.apps.automap:main',
         'hamster = pyrocko.apps.hamster:main',
         'jackseis = pyrocko.apps.jackseis:main',
         'colosseo = pyrocko.apps.colosseo:main',
         'sparrow = pyrocko.apps.sparrow:main',
         'sparrow%i = pyrocko.apps.sparrow:main' % sys.version_info.major],
    'gui_scripts':
        ['snuffler = pyrocko.apps.snuffler:main']
}

setup(
    cmdclass={
        'install': CustomInstallCommand,
        'build_py': CustomBuildPyCommand,
        # 'py2app': CustomBuildAppCommand,
        'build_ext': CustomBuildExtCommand,
        'check_multiple_install': CheckInstalls,
        'install_prerequisites': InstallPrerequisits,
        'uninstall': Uninstall,
    },

    name=packname,
    version=version,
    description='A versatile seismology toolkit for Python.',
    long_description=open('maintenance/readme-pip.rst', 'rb').read().decode('utf8'),
    author='The Pyrocko Developers',
    author_email='info@pyrocko.org',
    url='https://pyrocko.org',
    license='GPLv3',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: C',
        'Programming Language :: Python :: Implementation :: CPython',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Application Frameworks'
    ],
    keywords=[
        'seismology, waveform analysis, earthquake modelling, geophysics,'
        ' geophysical inversion'],
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, <4',
    install_requires=[],

    extras_require={
        'gui_scripts': ['PyQt4'],
    },

    packages=[packname] + subpacknames,
    package_dir={'pyrocko': 'src'},
    ext_package=packname,
    ext_modules=[
        Extension(
            'util_ext',
            extra_compile_args=['-Wextra'],
            sources=[op.join('src', 'ext', 'util_ext.c')]),

        Extension(
            'signal_ext',
            include_dirs=[get_python_inc(), numpy.get_include()],
            extra_compile_args=['-Wextra'],
            sources=[op.join('src', 'ext', 'signal_ext.c')]),

        Extension(
            'mseed_ext',
            include_dirs=[get_python_inc(), numpy.get_include(),
                          get_build_include('libmseed/')],
            library_dirs=[get_build_include('libmseed/')],
            libraries=['mseed'],
            extra_compile_args=['-Wextra'],
            sources=[op.join('src', 'io', 'ext', 'mseed_ext.c')]),

        Extension(
            'evalresp_ext',
            include_dirs=[get_python_inc(), numpy.get_include(),
                          get_build_include('evalresp-3.3.0/include/')],
            library_dirs=[get_build_include('evalresp-3.3.0/lib/')],
            libraries=['evresp'],
            extra_compile_args=[
                '-Wextra',
                '-I%s' % get_build_include('evalresp-3.3.0/include')],
            sources=[op.join('src', 'ext', 'evalresp_ext.c')]),

        Extension(
            'ims_ext',
            include_dirs=[get_python_inc(), numpy.get_include()],
            extra_compile_args=['-Wextra'],
            sources=[op.join('src', 'io', 'ext', 'ims_ext.c')]),

        Extension(
            'datacube_ext',
            include_dirs=[get_python_inc(), numpy.get_include()],
            extra_compile_args=['-Wextra'],
            sources=[op.join('src', 'io', 'ext', 'datacube_ext.c')]),

        Extension(
            'autopick_ext',
            include_dirs=[get_python_inc(), numpy.get_include()],
            extra_compile_args=['-Wextra'],
            sources=[op.join('src', 'ext', 'autopick_ext.c')]),

        Extension(
            'gf.store_ext',
            include_dirs=[get_python_inc(), numpy.get_include()],
            extra_compile_args=['-D_FILE_OFFSET_BITS=64', '-Wextra'] + omp_arg,
            extra_link_args=[] + omp_lib,
            sources=[op.join('src', 'gf', 'ext', 'store_ext.c')]),

        Extension(
            'eikonal_ext',
            include_dirs=[get_python_inc(), numpy.get_include()],
            extra_compile_args=['-Wextra'] + omp_arg,
            extra_link_args=[] + omp_lib,
            sources=[op.join('src', 'ext', 'eikonal_ext.c')]),

        Extension(
            'parstack_ext',
            include_dirs=[get_python_inc(), numpy.get_include()],
            extra_compile_args=['-Wextra'] + omp_arg,
            extra_link_args=[] + omp_lib,
            sources=[op.join('src', 'ext', 'parstack_ext.c')]),

        Extension(
            'ahfullgreen_ext',
            include_dirs=[get_python_inc(), numpy.get_include()],
            extra_compile_args=['-Wextra'],
            sources=[op.join('src', 'ext', 'ahfullgreen_ext.c')]),

        Extension(
            'orthodrome_ext',
            include_dirs=[get_python_inc(), numpy.get_include()],
            extra_compile_args=['-Wextra'],
            sources=[op.join('src', 'ext', 'orthodrome_ext.c')]),

        Extension(
            "avl",
            sources=[op.join('src', 'ext', 'pyavl-1.12', 'avl.c'),
                     op.join('src', 'ext', 'pyavl-1.12', 'avlmodule.c')],
            define_macros=[('HAVE_AVL_VERIFY', None),
                           ('AVL_FOR_PYTHON', None)],
            include_dirs=[get_python_inc()],
            extra_compile_args=['-Wno-parentheses', '-Wno-uninitialized'],
            extra_link_args=[] if sys.platform != 'sunos5' else ['-Wl,-x']),

        Extension(
            'modelling.disloc_ext',
            include_dirs=[get_python_inc(), numpy.get_include()],
            extra_compile_args=['-Wextra'] + omp_arg,
            extra_link_args=[] + omp_lib,
            sources=[op.join('src', 'modelling', 'ext', 'disloc_ext.c')]),

        Extension(
            'modelling.okada_ext',
            include_dirs=[get_python_inc(), numpy.get_include()],
            extra_compile_args=['-Wextra'] + omp_arg,
            extra_link_args=[] + omp_lib,
            sources=[op.join('src', 'modelling', 'ext', 'okada_ext.c')])
    ],

    scripts=[
        'src/apps/gmtpy-epstopdf',
    ],

    entry_points=entry_points,
    package_data={
        packname: ['data/*.png',
                   'data/*.html',
                   'data/earthmodels/*.nd',
                   'data/colortables/*.cpt',
                   'data/tectonics/*.txt',
                   'data/fomosto_report/gfreport.*',
                   'gui/snufflings/map/*ml',
                   'gui/snufflings/map/*.js',
                   ] + get_readme_paths()}
)
