#!/usr/bin/env python

import os
import sys
from glob import glob

from nose.core import run_exit

if len(sys.argv) == 2:
    target = sys.argv[1].split('.')
    if len(target) == 3:
        scripts = [sys.argv[1]]
    elif len(target) == 2:
        scripts = glob(os.path.join(*(target + ['test_*.py'])))
    elif len(target) == 1:
        scripts = glob(os.path.join(*(target + ['*', 'test_*.py'])))

else:
    scripts = glob(os.path.join('test', '*', 'test_*.py'))

sys.argv[1:] = scripts
run_exit()
