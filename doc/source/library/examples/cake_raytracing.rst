Traveltime calculation and raytracing
=====================================

Calculate traveltimes in layered media
--------------------------------------

Here we will excercise two example how to calculate traveltimes for the phases ``P`` and ``Pg`` for different earth velocity models.

Modules covered in this example:
 * :py:class:`pyrocko.cake`

 The first example is minimalistic and will give you a simple traveltime table.

Download :download:`cake_ray_tracing.py </../../examples/cake_arrivals.py>`

.. literalinclude :: /../../examples/cake_arrivals.py
    :language: python


The second code snippet includes some lines to plot a simple traveltime figure.

Download :download:`cake_ray_tracing.py </../../examples/cake_first_arrivals.py>`

.. literalinclude :: /../../examples/cake_first_arrivals.py
    :language: python



Calculate traveltimes in heterogeneous media
--------------------------------------------

These examples demonstrate how to use the :py:mod:`pyrocko.modelling.eikonal`
module to calculate first arrivals in heterogenous media.


.. literalinclude :: /../../examples/eikonal_example1.py
    :caption: :download:`eikonal_example1.py </../../examples/eikonal_example1.py>`
    :language: python

.. figure :: /static/eikonal_example1.png
    :align: center
    :width: 90%
    :alt: output of eikonal_example1.py

    First arrivals (contours) from a seismic source (star) at 15 km depth in a
    5-layer crustal model where velocities increase with depth.

.. literalinclude :: /../../examples/eikonal_example2.py
    :caption: :download:`eikonal_example2.py </../../examples/eikonal_example2.py>`
    :language: python

.. figure :: /static/eikonal_example2.png
    :align: center
    :width: 90%
    :alt: output of eikonal_example2.py

    First arrivals (contours) from a distant seismic source in a 2-layer
    crustal model with intrusions. The planar wave front entering from below is
    simulated by a source at 10 km depth moving quickly from left to right with
    a given constant speed.


Traveltime table interpolation
-------------------------------

This example demonstrates how to interpolate and query traveltime tables.

Classes covered in this example:
 * :py:class:`pyrocko.spit.SPTree` (interpolation of traveltime tables)
 * :py:class:`pyrocko.gf.meta.TPDef` (phase definitions)
 * :py:class:`pyrocko.gf.meta.Timing` (onset definition to query the travel
   time tables)

Download :download:`cake_raytracing.py </../../examples/cake_raytracing.py>`

.. literalinclude :: /../../examples/cake_raytracing.py
    :language: python
