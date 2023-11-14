# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------


from pyrocko.guts import String, Timestamp, List
from pyrocko.gui.qt_compat import qw
from pyrocko.gui import talkie
from pyrocko.gui.state import state_bind_lineedit

guts_prefix = 'gato'


class ConstraintsState(talkie.TalkieRoot):
    tmin = Timestamp.T(optional=True)
    tmax = Timestamp.T(optional=True)
    channels = List.T(String.T())


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
