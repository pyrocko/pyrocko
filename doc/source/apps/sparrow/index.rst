
.. image:: /static/sparrow.svg
   :align: left

Sparrow - *geospatial data visualization*
=========================================

The :program:`Sparrow` application provides a virtual globe to visualize
earthquake hypocenters, earthquake source models, fault traces, topography
datasets and other geoyhisical datasets in 3D and over time. Its primary focus
is on simplicity, allowing for interactive dataset exploration and export of
short animations for scientific presentations.

.. raw:: html

   <div style="clear:both"></div>

.. note::

   The Sparrow is currently in an experimental stage. Bugs and errors might
   occur during its use and some essential features might still be missing.

   Please help us by providing feedback through `Issues
   <https://git.pyrocko.org/pyrocko/pyrocko/issues>`_ `Discussion
   <https://hive.pyrocko.org/pyrocko-support/channels/sparrow>`_, or `Coding
   <https://git.pyrocko.org/pyrocko/pyrocko/projects/5>`_.

Requirements
------------

* The Sparrow has one additional requirement which (on purpose) is not listed
  as a hard requirement for the rest of Pyrocko: `VTK <https://vtk.org/>`_.
  Please install it with your installation method of choice (e.g. on Deb based
  Linuxes, use ``apt install python3-vtk``, in an pip-managed environment use
  ``pip install vtk``, or under Anaconda/Miniconda use ``conda install vtk``).
  Sparrow will complain if VTK cannot be found on startup.
* Make sure your graphics card driver setup allows for hardware-accelerated
  OpenGL (e.g. check with ``glxinfo -B``, look for ``OpenGL renderer``). If you
  have an NVIDIA graphics card, you may want to install their proprietary
  drivers.
* For visualization of InSAR scenes, `Kite <https://pyrocko.org/kite/>`_ is
  needed.
