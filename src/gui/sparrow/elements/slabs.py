# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

import numpy as num

from pyrocko.guts import Bool, Float, StringChoice
from pyrocko.gui.qt_compat import qw, qc

from pyrocko.dataset import slab2
from pyrocko.gui.vtk_util import TrimeshPipe

from . import base
from .. import common
from ..state import state_bind_combobox, state_bind_slider


guts_prefix = 'sparrow'

km = 1e3


class SlabParameterChoice(StringChoice):
    choices = slab2.get_parameter_names()


parameter_labels = {
    'depth': 'Depth [km]',
    'depth_uncertainty': 'Depth uncertainty [km]',
    'strike': 'Strike [deg]',
    'dip': 'Dip [deg]',
    'thickness': 'Thickness [km]',
}

parameter_display_factors = {
    'depth': 1.0 / km,
    'depth_uncertainty': 1.0 / km,
    'thickness': 1.0 / km,
}


class SlabsState(base.ElementState):
    visible = Bool.T(default=True)
    opacity = Float.T(default=1.0)
    parameter = SlabParameterChoice.T(default='depth')
    cpt = base.CPTState.T(default=base.CPTState.D(cpt_name='viridis'))

    def create(self):
        element = SlabsElement()
        return element


def slabs_to_mesh(slabs):
    vertices = []
    faces = []
    ioffset = 0
    for slab in slabs:
        slab_vertices, slab_faces = slab.get_mesh()
        vertices.append(slab_vertices)
        faces.append(slab_faces + ioffset)
        ioffset += slab_vertices.shape[0]

    return num.concatenate(vertices), num.concatenate(faces)


def slabs_to_values(slabs, parameter):
    return num.concatenate([
        slab.get_face_values(parameter) for slab in slabs]) \
        * parameter_display_factors.get(parameter, 1.0)


class SlabsElement(base.Element):

    def __init__(self):
        base.Element.__init__(self)
        self._parent = None
        self._controls = None
        self._pipe = None
        self._slabs = None
        self._mesh = None
        self._values = {}
        self.cpt_handler = base.CPTHandler()

    def get_name(self):
        return 'Slabs'

    def bind_state(self, state):
        base.Element.bind_state(self, state)
        self.talkie_connect(
            state, ['visible', 'opacity', 'parameter'], self.update)

        self.cpt_handler.bind_state(state.cpt, self.update)

    def unbind_state(self):
        self.cpt_handler.unbind_state()
        base.Element.unbind_state(self)

    def set_parent(self, parent):
        if self._slabs is None:
            try:
                self._slabs = slab2.Slab2().get_slabs()
            except slab2.Slab2Error as e:
                qw.QMessageBox.warning(
                    parent, 'Slab2 Error',
                    'Could not load Slab2 dataset:\n\n%s' % str(e))
                return

        self._parent = parent
        self._parent.add_panel(
            self.get_title_label(),
            self._get_controls(),
            visible=True,
            title_controls=[
                self.get_title_control_remove(),
                self.get_title_control_visible()])

        self.update()

    def unset_parent(self):
        self.unbind_state()
        if self._parent:
            self._remove_pipes()

            if self._controls:
                self._parent.remove_panel(self._controls)
                self._controls = None

            self._parent.update_view()
            self._parent = None

    def _remove_pipes(self):
        if self._pipe is not None:
            self._parent.remove_actor(self._pipe.actor)
            self._pipe = None

        self.cpt_handler.remove_cbar_pipe()

    def _get_mesh_values(self, parameter):
        if self._mesh is None:
            self._mesh = slabs_to_mesh(self._slabs)

        if parameter not in self._values:
            self._values[parameter] = slabs_to_values(self._slabs, parameter)

        return self._mesh, self._values[parameter]

    def update(self, *args):
        state = self._state

        if state.visible:
            (vertices, faces), values = self._get_mesh_values(state.parameter)

            self.cpt_handler._values = values
            self.cpt_handler.update_cpt()
            lut = self.cpt_handler._lookuptable

            if self._pipe is None:
                self._pipe = TrimeshPipe(
                    vertices, faces,
                    values=values,
                    lut=lut,
                    backface_culling=False)

                self._parent.add_actor(self._pipe.actor)
            else:
                self._pipe.set_values(values)
                self._pipe.set_lookuptable(lut)

            self._pipe.set_opacity(state.opacity)
            self.cpt_handler.update_cbar(parameter_labels[state.parameter])

        else:
            self._remove_pipes()

        self._parent.update_view()

    def _get_controls(self):
        if not self._controls:
            state = self._state

            frame = qw.QFrame()
            layout = qw.QGridLayout()
            layout.setAlignment(qc.Qt.AlignTop)
            frame.setLayout(layout)

            iy = 0
            layout.addWidget(qw.QLabel('Display Parameter'), iy, 0)
            cb = common.string_choices_to_combobox(SlabParameterChoice)
            layout.addWidget(cb, iy, 1)
            state_bind_combobox(self, state, 'parameter', cb)

            iy += 1
            layout.addWidget(qw.QLabel('Opacity'), iy, 0)
            slider = qw.QSlider(qc.Qt.Horizontal)
            slider.setSizePolicy(
                qw.QSizePolicy(
                    qw.QSizePolicy.Expanding, qw.QSizePolicy.Fixed))
            slider.setMinimum(0)
            slider.setMaximum(1000)
            layout.addWidget(slider, iy, 1)
            state_bind_slider(self, state, 'opacity', slider, factor=0.001)

            self.cpt_handler.cpt_controls(self._parent, state.cpt, layout)
            self.cpt_handler._update_cpt_combobox()
            self.cpt_handler._update_cptscale_lineedit()

            layout.addWidget(qw.QFrame(), layout.rowCount(), 0, 1, 3)

            self._controls = frame

        return self._controls


__all__ = [
    'SlabsElement',
    'SlabsState',
]
