# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import print_function

if __name__ == '__main__':
    import sys
    import pyrocko
    import numpy
    import scipy
    import matplotlib
    from pyrocko.gui.qt_compat import Qt
    if sys.argv[1:] == ['deps']:
        print('pyrocko: %s' % pyrocko.long_version)
        print('numpy: %s' % numpy.__version__)
        print('scipy: %s' % scipy.__version__)
        print('matplotlib: %s' % matplotlib.__version__)
        print('PyQt: %s' % Qt.PYQT_VERSION_STR)
        print('Qt: %s' % Qt.QT_VERSION_STR)
        print('python: %s.%s.%s' % sys.version_info[:3])
    elif sys.argv[1:] == ['short']:
        print(pyrocko.version)
    else:
        print(pyrocko.long_version)
