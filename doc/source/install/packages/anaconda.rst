Installation with conda
=======================

`Anaconda <https://www.anaconda.com/>`_, is a cross-platform Python
distribution with its own package manager ``conda``. For a more lightweight
installation, consider installing `Miniconda
<https://docs.conda.io/en/latest/miniconda.html>`_. Only the Python 3 version,
Anaconda3/Miniconda3 is supported for Pyrocko versions above v2021.04.02.

.. _conda_install:

From binary packages using ``conda``
------------------------------------

Pre-built packages are available for 64-bit Linux, MacOS and Windows. You can
use can use the ``conda`` package manager to install the Pyrocko framework:

.. code-block:: bash
    :caption: Pre-built Pyrocko conda packages are available for Linux, MacOS and Windows.

    conda install -c pyrocko pyrocko

More information available at https://anaconda.org/pyrocko/pyrocko


From source
-----------

Here's how to download, compile and install the latest Pyrocko version under
Anaconda on Linux or MacOS. For Windows source installs, please refer to
:ref:`Installation on Windows: From source <windows-install-from-source>`.

.. code-block:: bash
    :caption: Build and install from source

    cd ~/src/   # or wherever you keep your source packages
    git clone https://git.pyrocko.org/pyrocko/pyrocko.git pyrocko
    # make sure the correct conda environment is activated
    python3 install.py deps conda  # install prerequisites with conda
    python3 install.py user        # build and install Pyrocko with pip

**Note:** The native ``conda`` and ``pip`` commands to be run are printed
before execution and must be confirmed.

For instructions on how to install Pyrocko on other systems or if the
installation with the above procedure fails, see :doc:`index` or
:doc:`/install/details`.
