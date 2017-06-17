Generate maps
========================================

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
Station file used in the example :download:`stations_deadsea.pf </static/stations_deadsea.pf>`

.. literalinclude :: /static/automap_example.py
    :language: python

.. figure :: /static/automap_deadsea.png
    :align: center
    :alt: Map created using automap

.. rubric:: Footnotes

.. [#f1] Wessel, P., W. H. F. Smith, R. Scharroo, J. F. Luis, and F. Wobbe, Generic Mapping Tools: Improved version released, EOS Trans. AGU, 94, 409-410, 2013.
