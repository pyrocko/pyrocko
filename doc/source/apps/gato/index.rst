Gato - *generalized array toolkit*
==================================

.. warning::

    Gato is experimental. Its interfaces and configuration formats may still
    change.

Gato is a toolkit for seismic array processing. It is built on top of the
:doc:`Squirrel </topics/squirrel>` data access infrastructure and its
:doc:`processing pipelines </topics/squirrel_mantras>`. It is available as the
Python package :py:mod:`pyrocko.gato`, as the command line tool :app:`gato`
and as a graphical user interface.

Gato provides:

* Sensor array setups, either built-in (e.g. the Gräfenberg array ``:grf``)
  or defined from available station metadata.
* Search grids, e.g. of locations or slownesses, and tables of travel time
  delays from the grid nodes to the sensors, for plane waves, spherical waves
  or seismic phases in a layered earth model.
* Beamforming in the frequency domain, based on cross-spectral matrices, as
  operators which can be used in Squirrel mantras.

Getting started
---------------

List the built-in array setups::

    gato array list --arrays :

Show information about an array, e.g. the Gräfenberg array::

    gato array info --arrays :grf

Print example mantra configurations, which can serve as a starting point for
own processing pipelines::

    gato mantra

Start the graphical user interface::

    gato gui

Command reference
-----------------

The :app:`gato` tool and its subcommands are self-documenting with the
``--help`` option. They share the options for data collection and query with
the :doc:`squirrel </apps/squirrel/index>` tool.

.. toctree::
   :maxdepth: 3

   gato <reference/gato>

Python
------

See :py:mod:`pyrocko.gato` for the API reference.
