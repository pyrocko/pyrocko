'''
Calculate P-phase arrivals.
'''

from pyrocko import cake
import numpy as num

km = 1000.

# Load builtin 'prem-no-ocean' model (medium resolution)
model = cake.load_model('prem-no-ocean.m')

# Source depth [m].
source_depth = 300. * km

# Distances as a numpy array [deg].
distances = num.linspace(1500, 3000, 16)*km * cake.m2d

# Define the phase to use.
Phase = cake.PhaseDef('P')

# calculate distances and arrivals and print them:
print('distance [km]      time [s]')
for arrival in model.arrivals(distances, phases=Phase, zstart=source_depth):
    print('%13g %13g' % (arrival.x*cake.d2m/km, arrival.t))
