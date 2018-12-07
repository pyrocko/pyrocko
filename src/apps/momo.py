# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function, division

import sys
import signal

from pyrocko import moment_tensor_viewer as mtv

from PyQt4 import QtCore, QtGui


class Momo(QtGui.QApplication):

    def __init__(self, *args):
        QtGui.QApplication.__init__(self, *args)

        viewer = mtv.BeachballView()
        editor = mtv.MomentTensorEditor()

        self.win = QtGui.QMainWindow()
        self.win.setWindowTitle('Momo - Moment Tensor Calculator')
        self.win.setCentralWidget(viewer)

        dockwin = QtGui.QDockWidget('Moment Tensor')
        dockwin.setWidget(editor)
        self.win.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dockwin)
        self.connect(editor,
                     QtCore.SIGNAL("moment_tensor_changed(PyQt_PyObject)"),
                     viewer.set_moment_tensor)
        self.win.show()

        sb = self.win.statusBar()
        sb.clearMessage()
        sb.showMessage('Welcome to Momo!')

        self.connect(self, QtCore.SIGNAL("lastWindowClosed()"), self.myquit)
        signal.signal(signal.SIGINT, self.myquit)

    def myquit(self, *args):
        self.quit()


def main(args):
    app = Momo(args)
    app.exec_()

    sys.exit()


if __name__ == "__main__":
    main(sys.argv)
