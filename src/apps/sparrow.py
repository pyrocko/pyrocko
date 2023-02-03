# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function, division

import sys
import pyrocko


def main():
    try:
        from pyrocko.gui.sparrow import cli
        cli.main()

    except pyrocko.DependencyMissingVTK as e:
        sys.exit(str(e))

    except pyrocko.DependencyMissing as e:
        sys.exit(str(e))
