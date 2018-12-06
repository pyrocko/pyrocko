from pyrocko.gui.talkie import TalkieRoot


class ElementState(TalkieRoot):
    pass


class Element(object):
    def __init__(self):
        self._listeners = []

    def register_state_listener(self, listener):
        self._listeners.append(listener)  # keep listeners alive
