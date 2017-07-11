The :mod:`gshhg` module
==========================

The `GSHHG database <https://www.ngdc.noaa.gov/mgg/shorelines/gshhs.html>`_ is a high-resolution geography data set.

We implement functions to test points for land/water and mask land areas, excluding lakes.

See tutorial example :doc:`/library/examples/geographical_databases`.

.. note ::

    Please cite:

    **Wessel, P., and W. H. F. Smith, A Global Self-consistent, Hierarchical, High-resolution Shoreline Database, J. Geophys. Res., 101, #B4, pp. 8741-8743, 1996.**

.. autoclass:: pyrocko.gshhg.GSHHG
   :members:

.. autoclass:: pyrocko.gshhg.Polygon
   :members:
