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
    try:
        import matplotlib
    except ImportError:
        matplotlib = None
    try:
        from pyrocko.gui.qt_compat import Qt
    except ImportError:
        Qt = None

    na = 'not available'
    if sys.argv[1:] == ['deps']:
        print('pyrocko: %s' % pyrocko.long_version)
        print('numpy: %s' % numpy.__version__)
        print('scipy: %s' % scipy.__version__)
        print('matplotlib: %s' % (
            matplotlib.__version__ if matplotlib else na))
        print('PyQt: %s' % (Qt.PYQT_VERSION_STR if Qt else na))
        print('Qt: %s' % (Qt.QT_VERSION_STR if Qt else na))
        print('python: %s.%s.%s' % sys.version_info[:3])
    elif sys.argv[1:] == ['short']:
        print(pyrocko.version)
    else:
        print(pyrocko.long_version)
