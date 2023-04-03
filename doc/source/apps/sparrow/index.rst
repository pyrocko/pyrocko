
.. image:: /static/sparrow.svg
   :align: left


Sparrow - *geospatial data visualization*
=========================================

The :program:`Sparrow` application provides a virtual globe to visualize
earthquake hypocenters, earthquake source models, fault traces, topography
datasets and other geophysical datasets in 3D and over time. Its primary focus
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

The scene shown in the view is composed of a variable number of elements. By
default the Icosphere, Grid and Coastline elements are shown. More elements can
be added through the "Add" menu. Most of them can be configured with an
associated control panel. To hide/show an element without removing it, click on
the pentagram symbol in their control panel title bar.

Two useful elements are:

- **Catalog:** Display earthquake catalogs. Click "File" and select a catalog
  in Pyrocko format (basic or YAML) to get started.
- **Topography:** Display ETOPO and SRTMGL3 topography/bathymetry with
  adjustable transparency and variable vertical exaggeration. Pre-download the
  datasets using the commands ``automap --download-etopo`` and ``automap
  --download-srtmgl3`` (about 36 GB).

Further elements allow display of subtitles, time counters, station locations,
InSAR displacements, source models, faults, volcanoes and more.

If you feel like implementing a new element and need some pointers to get
started, have a chat with us in the `Sparrow channel
<https://hive.pyrocko.org/pyrocko-support/channels/sparrow>`_ on the Pyrocko
Hive.

**Snapshots and animations**

To take a snapshot of the current scene with all its settings, you can use the
Snapshot tool which is available from the "Panels" menu. A snapshot is not
simply a screenshot of the scene. Instead, it is a snapshot of the internal
state of the Sparrow application. You can go back to any snapshot by
double-clicking its thumbnail. The snapshot tool also allows you to transition
between snapshots. This is done by interpolating the numerical state values,
where possible, between the two snapshot states.

Simple animations are made by arranging multiple snapshots and transitions into
a timeline. The animations can be saved as MP4 movies to show-off your research
findings in a feasible and convenient way. By default the movies are exported
as Full HD (1920 x 1080 pixels). Other formats can be chosen by setting the
size though the "View" menu. To export a still image, use "Export Image..."
from the "File" menu.

Snapshot sequences can be saved as YAML text files. You can edit these files
simply with a text editor if you feel that what you want achieve is more
efficiently done this way. Of course you can also create or manipulate them
from a script. This is especially convenient for YAML files. YAML files can be
read/written in Python using the `pyyaml <https://pyyaml.org/>`_ package in an
agnostic way or by using the dedicated loaders in Pyrocko to directly
instantiate objects of the dedicated Pyrocko classes which are also used
internally in Sparrow.

**Textual manipulation of the Sparrow's state**

The complete internal state of the currently visible scene can be represented
as YAML document. If you detach the 3D view into a separate window (using
"Detach" from the "View" menu), this YAML document becomes available in the
main window. Manipulating the state in this textual representation can
sometimes be more efficient or more precise than through the GUI.

