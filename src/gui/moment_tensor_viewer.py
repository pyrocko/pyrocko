# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
from __future__ import absolute_import

from PyQt4 import QtCore as qc
from PyQt4 import QtGui as qg

from .util import make_QPolygonF, LinValControl
from .pile_viewer import Projection

from pyrocko import beachball, moment_tensor as mtm
from pyrocko import plot


class BeachballView(qg.QWidget):

    def __init__(self, *args):
        qg.QWidget.__init__(self, *args)
        mt = mtm.MomentTensor(m=mtm.symmat6(1., -1., 2., 0., -2., 1.))
        self._mt = mt
        self.set_moment_tensor(mt)

    def set_moment_tensor(self, mt):
        self._mt = mt
        self.update()

    def paintEvent(self, paint_ev):
        '''Called by QT whenever widget needs to be painted.'''

        painter = qg.QPainter(self)
        painter.setRenderHint(qg.QPainter.Antialiasing)
        self.drawit(painter)

    def drawit(self, p):
        '''Draw beachball into painter.'''

        h = self.height()
        w = self.width()

        s = min(h, w)*0.9

        xproj = Projection()
        xproj.set_in_range(-1., 1.)
        xproj.set_out_range((w-s)/2., w-(w-s)/2.)

        yproj = Projection()
        yproj.set_in_range(-1., 1.)
        yproj.set_out_range(h-(h-s)/2., (h-s)/2.)

        # m = mtm.symmat6(*(num.random.random(6)*2.-1.))
        # mtm.MomentTensor(m=m)

        mt = self._mt

        mt_devi = mt.deviatoric()
        eig = mt_devi.eigensystem()

        group_to_color = {
            'P': plot.graph_colors[0],
            'T': plot.graph_colors[1]}

        for (group, patches, patches_lower, patches_upper,
                lines, lines_lower, lines_upper) in beachball.eig2gx(eig):

            color = group_to_color[group]
            brush = qg.QBrush(qg.QColor(*color))
            p.setBrush(brush)

            pen = qg.QPen(qg.QColor(*color))
            pen.setWidth(1)
            p.setPen(pen)

            for poly in patches_lower:
                px, py, pz = poly.T
                points = make_QPolygonF(xproj(px), yproj(py))
                p.drawPolygon(points)

            color = (0, 0, 0)
            pen = qg.QPen(qg.QColor(*color))
            pen.setWidth(2)
            p.setPen(pen)

            for poly in lines_lower:
                px, py, pz = poly.T
                points = make_QPolygonF(xproj(px), yproj(py))
                p.drawPolyline(points)


class MomentTensorEditor(qg.QFrame):

    def __init__(self, *args):
        qg.QFrame.__init__(self, *args)

        self._mt = mtm.MomentTensor(m=mtm.symmat6(1., -1., 2., 0., -2., 1.))

        setupdata = [
            (LinValControl, 'Strike 1', 0., 360., 0., 0),
            (LinValControl, 'Dip 1', 0., 90., 0., 1),
            (LinValControl, 'Slip-Rake 1', -180., 180., 0., 2),
            (LinValControl, 'Strike 2', 0., 360., 0., 3),
            (LinValControl, 'Dip 2', 0., 90., 0., 4),
            (LinValControl, 'Slip-Rake 2', -180., 180., 0., 5)]

        layout = qg.QGridLayout()
        self.setLayout(layout)

        val_controls = []
        for irow, (typ, name, vmin, vmax, vcur, ind) in enumerate(setupdata):
            val_control = typ()
            val_control.setup(name, vmin, vmax, vcur, ind)
            val_controls.append(val_control)
            for icol, widget in enumerate(val_control.widgets()):
                layout.addWidget(widget, irow, icol)
            self.connect(
                val_control, qc.SIGNAL('valchange(PyQt_PyObject,int)'),
                self.valchange)

        self.val_controls = val_controls
        self.adjust_values()

    def adjust_values(self):

        ((strike1, dip1, rake1),
         (strike2, dip2, rake2)) = self._mt.both_strike_dip_rake()

        for val_control, value in zip(
                self.val_controls, [
                    strike1, dip1, rake1, strike2, dip2, rake2]):
            val_control.set_value(value)

    def valchange(self, val, ind):
        strike, dip, rake = [
            val_control.get_value() for val_control in self.val_controls[:3]]

        self._mt = mtm.MomentTensor(
            strike=strike, dip=dip, rake=rake)

        self.adjust_values()

        self.emit(
            qc.SIGNAL('moment_tensor_changed(PyQt_PyObject)'), self._mt)
