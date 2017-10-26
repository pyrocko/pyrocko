
import matplotlib

use_pyqt5 = False
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
    QSortFilterProxyModel = qc.QSortFilterProxyModel
    QItemSelectionModel = qc.QItemSelectionModel
    QItemSelection = qc.QItemSelection
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
    QSortFilterProxyModel = qg.QSortFilterProxyModel
    QItemSelectionModel = qg.QItemSelectionModel
    QItemSelection = qg.QItemSelection


try:
    vers = qc.QVersionNumber.fromString
except AttributeError:
    def vers(s):
        return tuple(s.split('.'))

# Application attribute has to be set for QWebView
if vers(Qt.QT_VERSION_STR) >= vers('5.4.0'):
    Qt.QCoreApplication.setAttribute(qc.Qt.AA_ShareOpenGLContexts, True)
