from pyrocko.guts import Bool, Float
from pyrocko import cake
from pyrocko.dataset import topo
from pyrocko.gui.qt_compat import qw, qc

from pyrocko.gui.vtk_util import TrimeshPipe
from .base import Element, ElementState
from pyrocko import geometry

guts_prefix = 'sparrow'


class TopoState(ElementState):
    visible = Bool.T(default=True)
    exaggeration = Float.T(default=1.0)


class TopoElement(Element):

    def __init__(self):
        Element.__init__(self)
        self.parent = None
        self.mesh = None
        self._controls = None

        region = (6.0, 8.0, 50., 51)
        self._tile = topo.get('SRTMGL3', region)
        self._visible = False

    def get_name(self):
        return 'Topography'

    def set_parent(self, parent):
        self.parent = parent
        state = parent.state
        upd = self.update
        self._listeners = [upd]
        state.add_listener(upd, 'topo.exaggeration')

        self.parent.add_panel(
            self.get_name(), self._get_controls(), visible=True)
        self.update()

    def update(self, *args):

        tstate = self.parent.state.topo

        if tstate.visible:
            if self.mesh is None:
                t = self._tile
                vertices, faces = geometry.topo_to_mesh(
                    t.y(), t.x(), t.data*tstate.exaggeration, cake.earthradius)

                self.mesh = TrimeshPipe(vertices, faces)

            else:
                t = self._tile
                vertices = geometry.topo_to_vertices(
                    t.y(), t.x(), t.data*tstate.exaggeration, cake.earthradius)

                self.mesh.set_vertices(vertices)

        if not self._visible and tstate.visible:
            self.parent.add_actor(self.mesh.actor)
            self._visible = tstate.visible

        if self._visible and not tstate.visible:
            self.parent.remove_actor(self.mesh.actor)
            self._visible = tstate.visible

        self.parent.update_view()

    def _get_controls(self):
        state = self.parent.state
        if not self._controls:
            from ..state import state_bind_slider

            frame = qw.QFrame()
            layout = qw.QGridLayout()
            frame.setLayout(layout)

            layout.addWidget(qw.QLabel('Exaggeration'), 0, 0)

            slider = qw.QSlider(qc.Qt.Horizontal)
            slider.setSizePolicy(
                qw.QSizePolicy(
                    qw.QSizePolicy.Expanding, qw.QSizePolicy.Minimum))
            slider.setMinimum(0)
            slider.setMaximum(5)
            # slider.setSingleStep(1)
            # slider.setPageStep(1)
            layout.addWidget(slider, 0, 1)

            state_bind_slider(self, state, 'topo.exaggeration', slider)

        self._controls = frame

        return self._controls


__all__ = [
    'TopoElement',
    'TopoState',
]
