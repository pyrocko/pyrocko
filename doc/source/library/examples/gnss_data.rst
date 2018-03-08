GPS (GNSS) data handling
=========================

Loading a GPS campaign from CSV files
--------------------------------------

.. highlight:: python

In this example we load GPS station locations and displacement data from the 1994 Northridge Earthquake into a :class:`~pyrocko.model.gnss.GNSSCampaign` data model. The model is then stored in a YAML file and loaded again using :mod:`pyrocko.guts`.

Download :download:`gnss_campaign.py </../../examples/gnss_campaign.py>`

.. literalinclude :: /../../examples/gnss_campaign.py
    :language: python
