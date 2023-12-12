# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------


from pyrocko.guts import String, Timestamp, List, str_duration, \
    parse_duration, Duration, Float
from pyrocko.gui.qt_compat import qw, qg
from pyrocko.gui import talkie, util as gui_util
from pyrocko.gui.state import state_bind_lineedit, state_bind

guts_prefix = 'gato'


class ConstraintsState(talkie.TalkieRoot):
    tmin = Timestamp.T(optional=True)
    tmax = Timestamp.T(optional=True)
    tduration = Duration.T(optional=True)
    tposition = Float.T(default=0.0)
    channels = List.T(String.T())
    tcursor = Timestamp.T(optional=True)

    @talkie.computed(['tmin', 'tmax', 'tduration', 'tposition'])
    def tmin_effective(self):
        return gui_util.tmin_effective(
            self.tmin, self.tmax, self.tduration, self.tposition)

    @talkie.computed(['tmin', 'tmax', 'tduration', 'tposition'])
    def tmax_effective(self):
        return gui_util.tmax_effective(
            self.tmin, self.tmax, self.tduration, self.tposition)


class Constrainer(qw.QFrame, talkie.TalkieConnectionOwner):

    def __init__(self, state, *args, **kwargs):
        qw.QFrame.__init__(self, *args, **kwargs)
        talkie.TalkieConnectionOwner.__init__(self)

        self.state = state

        layout = qw.QGridLayout()
        self.setLayout(layout)

        channels_le = qw.QLineEdit()
        channels_le.setPlaceholderText('BH?, SHZ')

        layout.addWidget(channels_le, 0, 0)

        def from_string(s):
            return [s.strip() for s in s.split(',') if s]

        def to_string(channels):
            return ', '.join(channels)

        state_bind_lineedit(
            self, self.state, 'channels', channels_le,
            from_string, to_string)

        layout.addWidget(self.controls_time())

    def state_bind(self, *args, **kwargs):
        state_bind(self, self.state, *args, **kwargs)

    def controls_time(self):
        frame = qw.QFrame(self)
        frame.setSizePolicy(
            qw.QSizePolicy.Minimum, qw.QSizePolicy.Fixed)

        layout = qw.QGridLayout()
        frame.setLayout(layout)

        layout.addWidget(qw.QLabel('Min'), 0, 0)
        le_tmin = qw.QLineEdit()
        layout.addWidget(le_tmin, 0, 1)

        layout.addWidget(qw.QLabel('Max'), 1, 0)
        le_tmax = qw.QLineEdit()
        layout.addWidget(le_tmax, 1, 1)

        label_tcursor = qw.QLabel()

        label_tcursor.setSizePolicy(
            qw.QSizePolicy.Minimum, qw.QSizePolicy.Fixed)

        layout.addWidget(label_tcursor, 2, 1)
        self._label_tcursor = label_tcursor

        self.state_bind(
            ['tmin'], gui_util.lineedit_to_time, le_tmin,
            [le_tmin.editingFinished, le_tmin.returnPressed],
            gui_util.time_to_lineedit,
            attribute='tmin')
        self.state_bind(
            ['tmax'], gui_util.lineedit_to_time, le_tmax,
            [le_tmax.editingFinished, le_tmax.returnPressed],
            gui_util.time_to_lineedit,
            attribute='tmax')

        self.tmin_lineedit = le_tmin
        self.tmax_lineedit = le_tmax

        range_edit = gui_util.RangeEdit()
        # range_edit.rangeEditPressed.connect(self.disable_capture)
        # range_edit.rangeEditReleased.connect(self.enable_capture)
        # range_edit.set_data_provider(self)
        # range_edit.set_data_name('time')

        xblock = [False]

        def range_to_range_edit(state, widget):
            if not xblock[0]:
                widget.blockSignals(True)
                widget.set_focus(state.tduration, state.tposition)
                widget.set_range(state.tmin, state.tmax)
                widget.blockSignals(False)

        def range_edit_to_range(widget, state):
            xblock[0] = True
            self.state.tduration, self.state.tposition = widget.get_focus()
            self.state.tmin, self.state.tmax = widget.get_range()
            xblock[0] = False

        self.state_bind(
            ['tmin', 'tmax', 'tduration', 'tposition'],
            range_edit_to_range,
            range_edit,
            [range_edit.rangeChanged, range_edit.focusChanged],
            range_to_range_edit)

        def handle_tcursor_changed():
            self.state.tcursor = range_edit.get_tcursor()

        range_edit.tcursorChanged.connect(handle_tcursor_changed)

        layout.addWidget(range_edit, 0, 2, 3, 1)

        layout.addWidget(qw.QLabel('Focus'), 0, 3)
        le_focus = qw.QLineEdit()
        layout.addWidget(le_focus, 0, 4)

        def focus_to_lineedit(state, widget):
            if state.tduration is None:
                widget.setText('')
            else:
                widget.setText('%s, %g' % (
                    str_duration(state.tduration),
                    state.tposition))

        def lineedit_to_focus(widget, state):
            s = str(widget.text())
            w = [x.strip() for x in s.split(',')]
            try:
                if len(w) == 0 or not w[0]:
                    state.tduration = None
                    state.tposition = 0.0
                else:
                    state.tduration = parse_duration(w[0])
                    if len(w) > 1:
                        state.tposition = float(w[1])
                    else:
                        state.tposition = 0.0

            except Exception:
                raise ValueError('need two values: <duration>, <position>')

        self.state_bind(
            ['tduration', 'tposition'], lineedit_to_focus, le_focus,
            [le_focus.editingFinished, le_focus.returnPressed],
            focus_to_lineedit)

        label_effective_tmin = qw.QLabel()
        label_effective_tmax = qw.QLabel()

        label_effective_tmin.setSizePolicy(
            qw.QSizePolicy.Minimum, qw.QSizePolicy.Fixed)
        label_effective_tmax.setSizePolicy(
            qw.QSizePolicy.Minimum, qw.QSizePolicy.Fixed)
        label_effective_tmin.setMinimumSize(
            qg.QFontMetrics(label_effective_tmin.font()).width(
                '0000-00-00 00:00:00.000  '), 0)

        layout.addWidget(label_effective_tmin, 1, 4)
        layout.addWidget(label_effective_tmax, 2, 4)

        for var in ['tmin', 'tmax', 'tduration', 'tposition']:
            self.talkie_connect(
                self.state, var, self.update_effective_time_labels)

        self._label_effective_tmin = label_effective_tmin
        self._label_effective_tmax = label_effective_tmax

        self.talkie_connect(
            self.state, 'tcursor', self.update_tcursor)

        return frame

    def update_effective_time_labels(self, *args):
        tmin = self.state.tmin_effective
        tmax = self.state.tmax_effective

        stmin = gui_util.time_or_none_to_str(tmin)
        stmax = gui_util.time_or_none_to_str(tmax)

        self._label_effective_tmin.setText(stmin)
        self._label_effective_tmax.setText(stmax)

    def update_tcursor(self, *args):
        tcursor = self.state.tcursor
        stcursor = gui_util.time_or_none_to_str(tcursor)
        self._label_tcursor.setText(stcursor)
