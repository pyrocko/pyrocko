from pyrocko import cake
import numpy as num
from math import pi

'''
Calculate travel times of the P-phase.
In this case you need an earthmodel as taup input file with 
format: 'nd'.
'''

# Load 'nd'-format earth model.
cake.mod = cake.load_model('prem.nd','nd')

# List of source depths [m].
source_depth = 1000.

# Distances have to be a numpy array [deg].
distances = num.linspace(1,20,21)

# Define phase to use.
Phase = cake.PhaseDef('P')

# calculate ditances and arrivals and print them:
for arrival in cake.mod.arrivals(distances, phases=Phase, zstart=source_depth):
    print (arrival.x*(cake.earthradius*pi/180),arrival.t)

