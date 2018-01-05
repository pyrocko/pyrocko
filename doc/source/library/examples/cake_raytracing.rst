Traveltime calculation and raytracing with ``cake``
===================================================

Calculate synthetic traveltimes
-------------------------------

Here we will excercise two example how to calculate traveltimes for the phases ``P`` and ``Pg`` for different earth velocity models.

Modules covered in this example:
 * :py:class:`python.cake`

 The first example is minimalistic and will give you a simple travel time table. 

Download :download:`cake_ray_tracing.py </../../examples/cake_arrivals.py>`

.. literalinclude :: /../../examples/cake_arrivals.py
    :language: python


The second code snippet includes some lines to plot a simple traveltime figure.

Download :download:`cake_ray_tracing.py </../../examples/cake_first_arrivals.py>`

.. literalinclude :: /../../examples/cake_first_arrivals.py
    :language: python


Travel time table interpolation
-------------------------------

This example demonstrates how to interpolate and query travel time tables.

Classes covered in this example:
 * :py:class:`pyrocko.spit.SPTree` (interpolation of travel time tables)
 * :py:class:`pyrocko.gf.meta.TPDef` (phase definitions)
 * :py:class:`pyrocko.gf.meta.Timing` (onset definition to query the travel
   time tables)

Download :download:`cake_raytracing.py </../../examples/cake_raytracing.py>`

.. literalinclude :: /../../examples/cake_raytracing.py
    :language: python

