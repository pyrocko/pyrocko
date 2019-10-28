GPS (GNSS) data handling
=========================

Loading a GNSS campaign from CSV files
---------------------------------------

.. highlight:: python

In this example we load GPS station locations and displacement data from the 1994 Northridge Earthquake into a :class:`~pyrocko.model.gnss.GNSSCampaign` data model. The model is then stored in a YAML file and loaded again using :mod:`pyrocko.guts`.

Download :download:`gnss_campaign.py </../../examples/gnss_campaign.py>`

.. literalinclude :: /../../examples/gnss_campaign.py
    :language: python


Loading and mapping of GNSS campaign data from UNR
---------------------------------------------------

The `Nevada Geodetic Laboratory (UNR) <http://geodesy.unr.edu/>`_ releases co-seismic GNSS surface displacements for significant earthquakes. This script shows the import of such co-seismic displacement tables for the 2019 Ridgecrest earthquake and mapping through :class:`~pyrocko.plot.automap`.

Download :download:`gnss_unr_campaign.py </../../examples/gnss_unr_campaign.py>`

.. literalinclude :: /../../examples/gnss_unr_campaign.py
    :language: python
