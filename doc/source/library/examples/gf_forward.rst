Forward modeling synthetic seismograms and displacements
========================================================

Calculate synthetic seismograms from a local GF store
-----------------------------------------------------

.. highlight:: python

It is assumed that a :class:`~pyrocko.gf.store.Store` with store ID
*crust2_dd* has been downloaded in advance. A list of currently available
stores can be found at http://kinherd.org/gfs.html as well as how to download
such stores.

Further API documentation for the utilized objects can be found at :class:`~pyrocko.gf.targets.Target`,
:class:`~pyrocko.gf.seismosizer.LocalEngine` and :class:`~pyrocko.gf.seismosizer.DCSource`.

Download :download:`gf_forward_example1.py </static/gf_forward_example1.py>`

.. literalinclude :: /static/gf_forward_example1.py
    :language: python

.. figure :: /static/gf_synthetic.png
    :align: center
    :width: 90%
    :alt: Synthetic seismograms calculated through pyrocko.gf

    Synthetic seismograms calculated through :class:`pyrocko.gf` displayed in :doc:`/apps/snuffler/index`. The three traces show the east, north and vertical synthetical displacement stimulated by a double-couple source at 155 km distance.


Calculate spatial surface displacement from a local GF store
-------------------------------------------------------------

In this example we create a :class:`~pyrocko.gf.seismosizer.RectangularSource` and compute the spatial static/geodetic displacement caused by that rupture.

We will utilize :class:`~pyrocko.gf.seismosizer.LocalEngine`, :class:`~pyrocko.gf.targets.StaticTarget` and :class:`~pyrocko.gf.targets.SatelliteTarget`.

.. figure:: /static/gf_static_displacement.png
    :align: center
    :width: 90%
    :alt: Static displacement from a strike-slip fault calculated through pyrocko

    Synthetic surface displacement from a vertical strike-slip fault, with a N104W azimuth, in the Line-of-sight (LOS), east, north and vertical directions. LOS as for Envisat satellite (Look Angle: 23., Heading:-76). Positive motion toward the satellite. 

Download :download:`gf_forward_example2.py </static/gf_forward_example2.py>`

.. literalinclude :: /static/gf_forward_example2.py
    :language: python

Calculate forward model of thrust event and display wrapped phase
-----------------------------------------------------------------

In this example we compare the synthetic unwappred and wrapped LOS displacements caused by a thrust rupture.

.. figure:: /static/gf_static_wrapper.png
    :align: center
    :width: 90%
    :alt: Static displacement from a thrust fault calculated through pyrocko

    Synthetic LOS displacements from a south-dipping thrust fault. LOS as for Sentinel-1 satellite (Look Angle: 36., Heading:-76). Positive motion toward the satellite. Left: unwrapped phase. Right: Wrapped phase.


Download :download:`gf_forward_example3.py </static/gf_forward_example3.py>`

.. literalinclude :: /static/gf_forward_example3.py
    :language: python


Combining severals sources 
---------------------------
In this example we combine two rectangular sources and plot the forward model in profile.

.. figure:: /static/gf_static_several.png
    :align: center
    :width: 90%

    Synthetic LOS displacements from a flower-structure made of one strike-slip
    fault and one thrust fault. LOS as for Sentinel-1 satellite (Look Angle:
    36., Heading:-76). Positive motion toward the satellite. 

Download :download:`gf_forward_example4.py </static/gf_forward_example4.py>`

.. literalinclude :: /static/gf_forward_example4.py
    :language: python
