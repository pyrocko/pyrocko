from pyrocko import autopick_ext
import numpy as num

in_data = num.arange(10, dtype=num.float)
print autopick_ext.recursive_stalta(10, 20, 0.5, 0.5, True, [1.,2.,3.])

