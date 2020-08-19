#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import platform
import sys
import os

if '--help' in sys.argv[1:] or '-h' in sys.argv[1:]:
    sys.exit('''usage: install_prerequisites.py [--yes] [--help]

Try to install Pyrocko's prerequisites through your system's native package
manager. This script simply calls a shell script (see under `prerequisites/`)
appropriate for your system and Python version.

Options:

    --yes     Do not ask any questions (batch mode).
    --help    Show this help message and exit

''')

force_yes = '--yes' in sys.argv[1:]


distribution = ''
try:
    distribution = platform.linux_distribution()[0].lower().rstrip()
except Exception:
    pass

if not distribution:
    try:
        if platform.uname()[2].find('arch') != -1:
            distribution = 'arch'
    except Exception:
        pass

if not distribution:
    sys.exit(
        'Cannot determine platform for automatic prerequisite installation.')

if distribution == 'ubuntu':
    distribution = 'debian'

if distribution.startswith('centos'):
    distribution = 'centos'

fn = 'prerequisites/prerequisites_%s_python%i.sh' % (
        distribution, sys.version_info.major)

if not force_yes:
    try:
        input_func = raw_input
    except NameError:
        input_func = input

    confirm = input_func('Execute: %s \n\
proceed? [y/n]' % open(fn, 'r').read())
    if not confirm.lower() == 'y':
        sys.exit(0)

os.execl('/bin/sh', 'sh', fn)
