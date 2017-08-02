Traveltime calculation and raytracing
=====================================

Travel time table interpolation
-------------------------------

This example demonstrates how to interpolate and query travel time tables.

Classes covered in this example:
 * :py:class:`pyrocko.spit.SPTree` (interpolation of travel time tables)
 * :py:class:`pyrocko.gf.meta.TPDef` (phase definitions)
 * :py:class:`pyrocko.gf.meta.Timing` (onset definition to query the travel
   time tables)

.. literalinclude :: /../../src/tutorials/cake_ray_tracing.py
    :language: python

 Download :download:`cake_ray_tracing.py </../../src/tutorials/cake_ray_tracing.py>`
