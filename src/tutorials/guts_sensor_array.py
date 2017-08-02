from pyrocko.guts import load, Object, Float, Int, String
from pyrocko.gf import Target

guts_prefix = 'gft'

class SensorArray(Target):

    distance_min = Float.T()
    distance_max = Float.T()
    strike = Float.T()
    sensor_count = Int.T(default=50)

    # this attribute is only used in this example
    name = String.T(optional=True)

    def __init__(self, **kwargs):

        # call the guts initilizer
        Object.__init__(self, **kwargs)
