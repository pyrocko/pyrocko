#!/usr/bin/env python

'''Moment tensor calculator.'''

# Copyright (c) 2010, Sebastian Heimann <sebastian.heimann@zmaw.de>
#
# This file is part of pyrocko. For licensing information please see the file
# COPYING which is included with pyrocko.

import sys
import signal

from pyrocko import moment_tensor_viewer as mtv

from PyQt4.QtCore import *  # noqa
from PyQt4.QtGui import *  # noqa


class Momo(QApplication):

    def __init__(self, *args):
        apply(QApplication.__init__, (self,) + args)

        viewer = mtv.BeachballView()
        editor = mtv.MomentTensorEditor()

        self.win = QMainWindow()
        self.win.setWindowTitle('Momo - Moment Tensor Calculator')
        self.win.setCentralWidget(viewer)

        dockwin = QDockWidget('Moment Tensor')
        dockwin.setWidget(editor)
        self.win.addDockWidget(Qt.BottomDockWidgetArea, dockwin)
        self.connect(editor, SIGNAL("moment_tensor_changed(PyQt_PyObject)"),
                     viewer.set_moment_tensor)
        self.win.show()

        sb = self.win.statusBar()
        sb.clearMessage()
        sb.showMessage('Welcome to Momo!')

        self.connect(self, SIGNAL("lastWindowClosed()"), self.myquit)
        signal.signal(signal.SIGINT, self.myquit)

    def myquit(self, *args):
        self.quit()


def main(args):

    app = Momo(args)
    app.exec_()

    sys.exit()

if __name__ == "__main__":
    main(sys.argv)
