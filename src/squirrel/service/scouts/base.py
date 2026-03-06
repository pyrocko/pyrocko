from pyrocko.guts import Object, String, Timestamp, List, Float
from pyrocko.model.codes import CodesNSLCE

from pyrocko.squirrel.mantra import Mantra


class ScoutContext(Object):
    time = Timestamp.T()
    tmin = Timestamp.T()
    tmax = Timestamp.T()
    codes = List.T(CodesNSLCE.T())
    codes_visible = List.T(CodesNSLCE.T())
    frequency_min = Float.T()
    frequency_max = Float.T()


class ScoutResult(Object):
    name = String.T()
    context = ScoutContext.T()
    image_data_base64 = String.T(optional=True)


class Scout(Object):
    name = String.T()
    mantra = Mantra.T()

    def update(self, context):
        pass


__all__ = [
    'Scout',
    'ScoutResult',
    'ScoutContext',
]
