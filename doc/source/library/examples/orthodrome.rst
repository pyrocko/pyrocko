Geodesic functions
==================

Pyrocko's :class:`~pyrocko.orthodrome` module offers geodesic functions to solve a variety of common geodetic problems, like distance and angle calculation on a spheroid.


Distance between points on earth
--------------------------------

In this example we use :func:`~pyrocko.orthodrome.distance_accurate50m` and :class:`pyrocko.model` to calculate the distance between two points on earth.

Download :download:`orthodrome_example1.py </../../examples/orthodrome_example1.py>`

.. literalinclude :: /../../examples/orthodrome_example1.py
    :language: python


This can also be serialised for multiple coordinates stored in :class:`numpy.ndarray` through :func:`~pyrocko.orthodrome.distance_accurate50m_numpy`.


Download :download:`orthodrome_example2.py </../../examples/orthodrome_example2.py>`

.. literalinclude :: /../../examples/orthodrome_example2.py
    :language: python


Azimuth and backazimuth
-----------------------

Calculation of azimuths and backazimuths between two points on a spherical earth.

Download :download:`orthodrome_example3.py </../../examples/orthodrome_example3.py>`

.. literalinclude :: /../../examples/orthodrome_example3.py
    :language: python


Conversion to carthesian coordinates
------------------------------------

Given two sets of coordinates, :func:`~pyrocko.orthodrome.latlon_to_ne` returns the distance in meters in north/east direction.

Download :download:`orthodrome_example4.py </../../examples/orthodrome_example4.py>`

.. literalinclude :: /../../examples/orthodrome_example4.py
    :language: python



Relative carthesian coordinates to latitude and longitude
---------------------------------------------------------

Download :download:`orthodrome_example5.py </../../examples/orthodrome_example5.py>`

.. literalinclude :: /../../examples/orthodrome_example5.py
    :language: python
