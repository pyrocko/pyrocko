Cake
====

Cake is a module which is able to solve classical seismic ray therory problems for layered earth models (layer cake models).

For various phases this module can calculate:

* arrival times
* ray paths
* reflection and transmission coefficients
* take-off and incidence angles
* geometrical spreading factors

Computations are done for a spherical earth.
Cake can either be invoked from the command line directly or imported into python scripts. Both is demonstrated in the examples below.

Invocation
----------

::  
    
    cake (command) [options]

Commands:

**print**

    print the imported earh model.

**plot-xt**

    plot arrival times over epicentral distance.

**plot-xp**

    plot ray parameter over epizentral distance.
    
**plot-rays**

    plot a fan of phase paths.

**plot**

    plot arrival times over epicentral distance and a fan of phase paths.

For further help on each of these commands type ``--help`` after the individual command.

Command Line Examples
---------------------

Plot P, S and p Phases
^^^^^^^^^^^^^^^^^^^^^^

Ten receiver distances ranging from 100km to 1000km and a source depth of 10km.

::

    cake plot-rays --crust2loc=45,10 --phases=P,p --sdepth=10 --distances=100:1000:10

The option ``--crust2loc`` refers to the `crust2x2 <http://emolch.github.com/pyrocko/crust2x2.html>`_ pyrocko module and expects latitude and longitude of the source location.

Python Script Examples
----------------------

Calculate P-phase arrivals for the whole earth 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following python script calculates arrival times for the P-phase emitted by an event in a depth of 300km.

::

    from pyrocko import cake
    import numpy as num
    from math import pi

    '''
    Calculate P-phase arrivals.
    '''

    # Load 'nd'-format earth model.
    cake.mod = cake.load_model('prem.nd','nd')

    # Source depths [km].
    source_depth = 300.

    # Distances have to be stored as numpy array [deg].
    distances = num.linspace(1,20,21)

    # Define the phase to use.
    Phase = cake.PhaseDef('P')

    # calculate distances and arrivals and print them:
    for arrival in cake.mod.arrivals(distances, phases=Phase, zstart=source_depth):
        print (arrival.x*(cake.earthradius*pi/180),arrival.t)

