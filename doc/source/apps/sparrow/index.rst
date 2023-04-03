
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

.. image:: /static/sparrow/okataina.jpg
   :align: left

.. note::

   The Sparrow is currently in an experimental stage. Bugs and errors might
   occur during its use and some essential features might still be missing.

   Please help us by providing feedback through `Issues
   <https://git.pyrocko.org/pyrocko/pyrocko/issues>`_, `Discussion
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

Invocation
----------

Run ``sparrow`` from the command line to launch into Sparrow's graphical user
interface (GUI). To get a list of its options, run ``sparrow --help``.

.. figure :: /static/sparrow/intro1.png
    :align: center
    :alt: Sparrow initial screen
    :figwidth: 100%

    Figure 1: Startup state of the Squirrel GUI, with Icosphere, Grid and
    Coastline elements

First flaps
-----------

**Online help**

Instructions and help are shown in the status bar at the bottom of the window.

**Navigation**

There are two navigation modes, global and fixed. In global navigation mode the
location of the focal point is changed when you click and drag the mouse, in
fixed navigation mode, the location is fixed and the view angle is changed when
you click and drag. Navigate to a point of interest in global mode and then
click "Fix" in the navigation panel to investigate your target. You can also
hold down the control key to toggle between the navigation modes.

The focus point is visually shown when "Crosshair" in the navigation panel is 
checked.

The depth of the focal point can be set by modifying the third value [km] in
the "Location" field. Use negative values to set the focal point into the
atmosphere.

**Elements**

The scene shown in the view is composed of so-called elements. By default the
Icosphere, Grid and Coastline elements are shown. You can add more elements
using the "Add" menu. Elements can be shown/hidden by clicking on the pentagon sym
