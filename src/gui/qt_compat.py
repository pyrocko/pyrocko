import os
import platform
import matplotlib

from pyrocko import config

# Needed by MacOS Big Sur
# https://stackoverflow.com/questions/64833558/apps-not-popping-up-on-macos-big-sur-11-0-1#_=
if platform.uname()[0] == 'Darwin':
    os.environ['QT_MAC_WANTS_LAYER'] = '1'

gui_toolkit = config.effective_gui_toolkit()

qt5_backend_available = 'Qt5Agg' in matplotlib.rcsetup.all_backends

if gui_toolkit == 'auto':
    try:
        if qt5_backend_available:
            import PyQt5
            use_pyqt5 = True
        else:
            use_pyqt5 = False
    except ImportError:
        use_pyqt5 = False

elif gui_toolkit == 'qt4':
    use_pyqt5 = False
else:
    use_pyqt5 = True
    if not qt5_backend_available:
        raise Exception(
            'Qt5 was forced but matplotlib Qt5Agg backend is not availble')


if use_pyqt5:
    if matplotlib.get_backend().find('Qt4') != -1:  # noqa
        matplotlib.use('Qt5Agg')

    import PyQt5 as PyQt
    from PyQt5 import Qt
    from PyQt5 import QtCore as qc
    from PyQt5 import QtGui as qg
    from PyQt5 import QtWidgets as qw
    from PyQt5 import QtNetwork as qn
    from PyQt5 import QtOpenGL as qgl
    from PyQt5 import QtSvg as qsvg
    from PyQt5 import QtPrintSupport as qprint
    QSortFilterProxyModel = qc.QSortFilterProxyModel
    QItemSelectionModel = qc.QItemSelectionModel
    QItemSelection = qc.QItemSelection

    def getSaveFileName(*args, **kwargs):
         return qw.QFileDialog.getSaveFileName(*args, **kwargs)[0]

    class QPixmapCache(qg.QPixmapCache):
        def cached(self, key):
            return self.find(key)

else:
    if matplotlib.get_backend().find('Qt5') != -1:  # noqa
        matplotlib.use('Qt4Agg')

    import PyQt4 as PyQt
    from PyQt4 import Qt
    from PyQt4 import QtCore as qc
    from PyQt4 import QtGui as qg
    qw = qg
    from PyQt4 import QtNetwork as qn
    from PyQt4 import QtOpenGL as qgl
    from PyQt4 import QtSvg as qsvg
    qprint = qg
    QSortFilterProxyModel = qg.QSortFilterProxyModel
    QItemSelectionModel = qg.QItemSelectionModel
    QItemSelection = qg.QItemSelection

    getSaveFileName = qw.QFileDialog.getSaveFileName

    class QPixmapCache(qg.QPixmapCache):
        def cached(self, key):
            pixmap = qg.QPixmap()
            found = self.find(key, pixmap)
            if found:
                return pixmap
            return found


try:
    vers = qc.QVersionNumber.fromString
except AttributeError:
    def vers(s):
        return tuple(s.split('.'))

# Application attribute has to be set for QWebView
if vers(Qt.QT_VERSION_STR) >= vers('5.4.0'):
    Qt.QCoreApplication.setAttribute(qc.Qt.AA_ShareOpenGLContexts, True)
