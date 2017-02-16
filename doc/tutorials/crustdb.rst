Crustal Velocity Profils
========================

Crust 2x2
---------

CrustDB empirical profiles
--------------------------
The `Global Crustal Database <https://earthquake.usgs.gov/data/crust/>`_ gathers empirical 1D velocity models from seismic reflection and refraction profiles. As of 2013 there are 138939 profiles, mainly P and fewer S wave records in the database.

.. note ::

    **Citation:**

    W.D. Mooney, G. Laske and G. Masters, CRUST 5.1: A global crustal model
    at 5°x5°. J. Geophys. Res., 103, 727-747, 1998.

::

    from pyrocko import crustdb

    cdb = crustdb.CrustDB()
    europe = cdb.selectLocation(lat=52., lon=20., radius=15.)\
        .selectMinDepth(20)

    europe.plot(plot_median=False, plot_mode=False)
    europe.plot(plot_median=False, plot_mode=False, vrange=(2000, 6000), phase='s')


.. image:: ../_static/crustdb_plot.png
    :align: center
    :alt: pyrocko.crustdb.CrustDB.plot


Other selection methods are :func:`~pyrocko.crustdb.CrustDB.selectPolygon` and
:func:`~pyrocko.crustdb.CrustDB.selectRegion`.

See :doc:`../reference_crustdb` for more information on the API.
