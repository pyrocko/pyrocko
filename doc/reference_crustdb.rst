The :mod:`crustdb` Module
==========================

The `Global Crustal Database <https://earthquake.usgs.gov/data/crust/>`_ gathers empirical 1D velocity models from seismic reflection and refraction profiles. As of 2013 there are 138939 profiles, mainly P and fewer S wave records in the database.

See the tutorial for usage example, :doc:`examples/velocity_databases`.

.. note ::

    Please cite:

    **W.D. Mooney, G. Laske and G. Masters, CRUST 5.1: A global crustal model
    at 5°x5°. J. Geophys. Res., 103, 727-747, 1998.**

.. autoclass:: pyrocko.crustdb.CrustDB
   :members:

.. autoclass:: pyrocko.crustdb.VelocityProfile
   :members:
