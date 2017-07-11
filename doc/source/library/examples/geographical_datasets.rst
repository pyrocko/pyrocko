Geographical Datasets
======================

Pyrocko offers access to commonly used geographical datasets, such as 

GSHHG Coastal Database
----------------------

The `GSHHG database <https://www.ngdc.noaa.gov/mgg/shorelines/gshhs.html>`_ is a high-resolution geography data set. We implement functions to extract coordinates of landmasks.

Classes covered in this example:
 * :py:class:`pyrocko.gshhg.GSHHG`

 .. literalinclude :: /static/gshhg_example.py
    :language: python

Download :download:`gshhg_example.py </static/gshhg_example.py>`

.. rubric:: Footnotes

.. [#f1] Wessel, P., and W. H. F. Smith, A Global Self-consistent, Hierarchical, High-resolution Shoreline Database, J. Geophys. Res., 101, #B4, pp. 8741-8743, 1996.


Tectonic Plates and Boundaries (PB2003)
---------------------------------------

*An updated digital model of plate boundaries* [#f2]_ offers a global set of present plate boundaries on the Earth. This database used in :mod:`pyrocko.automap`.

Classes covered in this example:
 * :py:class:`pyrocko.tectonics.PeterBird2003`

 .. literalinclude :: /static/tectonics_example.py
    :language: python

Download :download:`tectonics_example.py </static/tectonics_example.py>`

.. rubric:: Footnotes

.. [#f2] Bird, Peter. "An updated digital model of plate boundaries." Geochemistry, Geophysics, Geosystems 4.3 (2003).


Global Srain Rate (GSRM1)
-------------------------

Access to the global strain rate data set from Kreemer et al. (2003) [#f3]_.

Classes to be covered in this example:
 * :py:class:`pyrocko.tectonics.GSRM1`

.. warning :: To be documented by example!

.. rubric:: Footnotes

.. [#f3] Kreemer, C., W.E. Holt, and A.J. Haines, "An integrated global model of present-day plate motions and plate boundary deformation", Geophys. J. Int., 154, 8-34, 2003.
