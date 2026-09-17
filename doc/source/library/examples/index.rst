Examples
=========

The Pyrocko framework covers many aspects of seismological data handling and
management. This section offers tutorials and real-world examples on how to
utilize the framework.

Jupyter Notebooks
-----------------

Example notebooks are available in the `Pyrocko Notebooks Repository <https://git.pyrocko.org/pyrocko/pyrocko-notebooks>`_ - Contributions to the collection are highly welcome!


Scripts collection
------------------

Annotated scripts and real-world use cases using Pyrocko, grouped roughly by
how deep into the framework they go.

Getting started
................

Data access and handling basics - start here if you are new to Pyrocko.

.. toctree::
    :maxdepth: 2

    squirrel/index
    trace_handling
    metadata
    fdsn_download
    catalog_search
    snuffler_markers

Core toolbox
............

Datasets, coordinates, plotting, and other tools used across most
Pyrocko-based projects.

.. toctree::
    :maxdepth: 2

    cake_raytracing
    velocity_databases
    moment_tensor
    gnss_data
    geographical_datasets
    plotting
    gmtpy/index
    dataset_management
    orthodrome
    obspy_compat
    guts

Advanced functionality
......................

Green's function based forward modeling and source models for seismic and
geodetic problems.

.. toctree::
    :maxdepth: 2

    gf_forward
    kindyn_modeling
