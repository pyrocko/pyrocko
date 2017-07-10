Plotting functions
========================================

Generating topographic maps with ``automap``
--------------------------------------------

The :py:mod:`pyrocko.automap` module provides a painless and clean interface
for the `Generic Mapping Tool (GMT) <http://gmt.soest.hawaii.edu/>`_ [#f1]_.

Classes covered in these examples:
 * :py:class:`pyrocko.automap.Map`

For details on GMT wrapping module:
 * :py:mod:`pyrocko.gmtpy`

Topographic map of Dead Sea basin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example demonstrates how to create a map of the Dead Sea area with largest
cities, topography and gives a hint on how to access genuine GMT methods.

Download :download:`automap_example.py </static/automap_example.py>`

Station file used in the example: :download:`stations_deadsea.pf </static/stations_deadsea.pf>`

.. literalinclude :: /static/automap_example.py
    :language: python

.. figure :: /static/automap_deadsea.jpg
    :align: center
    :alt: Map created using automap

.. rubric:: Footnotes

.. [#f1] Wessel, P., W. H. F. Smith, R. Scharroo, J. F. Luis, and F. Wobbe, Generic Mapping Tools: Improved version released, EOS Trans. AGU, 94, 409-410, 2013.


Plotting beachballs (focal mechanisms)
--------------------------------------

Classes covered in these examples:
 * :py:class:`pyrocko.beachball` (visual representation of a focal mechanism)
 * :py:mod:`pyrocko.moment_tensor` (a 3x3 matrix representation of an
   earthquake source)
 * :py:class:`pyrocko.gf.seismosizer.DCSource` (a representation of a double
   couple source object),
 * :py:class:`pyrocko.gf.seismosizer.RectangularExplosionSource` (a
   representation of a rectangular explostion source), 
 * :py:class:`pyrocko.gf.seismosizer.CLVDSource` (a representation of a
   compensated linear vector diploe source object)
 * :py:class:`pyrocko.gf.seismosizer.DoubleDCSource` (a representation of a
   double double-couple source object).

Beachballs from moment tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example demonstrates how to create beachballs from (random) moment tensors.  

Download :download:`beachball_example01.py </static/beachball_example01.py>`

.. literalinclude :: /static/beachball_example01.py
    :language: python

.. figure :: /static/beachball-example01.png
    :align: center
    :alt: Beachballs (focal mechanisms) created by moment tensors.

    An artistic display of focal mechanisms drawn by classes
    :py:class:`pyrocko.beachball` and :py:mod:`pyrocko.moment_tensor`.


This example shows how to plot a full, a deviatoric and a double-couple beachball
for a moment tensor.

Download :download:`beachball_example03.py </static/beachball_example03.py>`

.. literalinclude :: /static/beachball_example03.py
    :language: python

.. figure :: /static/beachball-example03.png
    :align: center
    :alt: Beachballs (focal mechanisms) options created from moment tensor

    The three types of beachballs that can be plotted through pyrocko.

Beachballs from source objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example shows how to add beachballs of various sizes to the corners of a
plot by obtaining the moment tensor from four different source object types:
:py:class:`pyrocko.gf.seismosizer.DCSource` (upper left),
:py:class:`pyrocko.gf.seismosizer.RectangularExplosionSource` (upper right), 
:py:class:`pyrocko.gf.seismosizer.CLVDSource` (lower left) and
:py:class:`pyrocko.gf.seismosizer.DoubleDCSource` (lower right).

Creating the beachball this ways allows for finer control over their location
based on their size (in display units) which allows for a round beachball even
if the axis are not 1:1.

Download :download:`beachball_example02.py </static/beachball_example02.py>`

.. literalinclude :: /static/beachball_example02.py
    :language: python


.. figure :: /static/beachball-example02.png
    :align: center
    :alt: Beachballs (focal mechanisms) created in corners of graph.

    Four different source object types plotted with different beachball sizes.


Add station symbols to focal sphere diagram
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example shows how to add station symbols at the positions where P wave
rays pierce the focal sphere.

The function to plot focal spheres
(:py:func:`pyrocko.beachball.plot_beachball_mpl`) uses the function
:py:func:`pyrocko.beachball.project` in the final projection from 3D to 2D
coordinates. Here we use this function to place additional symbols on the plot.
The take-off angles needed can be computed with some help of the
:py:mod:`pyrocko.cake` module. Azimuth and distance computations are done with
functions from :py:mod:`pyrocko.orthodrome`.

Download :download:`beachball_example04.py </static/beachball_example04.py>`

.. literalinclude :: /static/beachball_example04.py
    :language: python

.. figure :: /static/beachball-example04.png
    :align: center
    :alt: Focal sphere diagram with station symbols

    Focal sphere diagram with markers at positions of P wave ray piercing points.
