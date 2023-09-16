# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------


from pyrocko.guts import String
from .gshhg import GSHHGElement, GSHHGState

guts_prefix = 'sparrow'


class CoastlinesState(GSHHGState):

    dataset = String.T(default='coastlines')

    def create(self):
        element = CoastlinesElement()
        return element


class CoastlinesElement(GSHHGElement):

    def __init__(self):
        GSHHGElement.__init__(self)

    def get_name(self):
        return 'Coastlines'

    def bind_state(self, state):
        GSHHGElement.bind_state(self, state)


__all__ = [
    'CoastlinesElement',
    'CoastlinesState']
