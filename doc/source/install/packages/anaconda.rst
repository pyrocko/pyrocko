Installation under Anaconda
===========================

`Anaconda <https://www.anaconda.com/>`_, is a cross-platform Python
distribution with its own package manager ``conda``. There are two versions:
Anaconda3 for Python 3 and Anaconda2 for Python 2. Pyrocko can be installed
under either of them.


.. _conda_install:

Anaconda3 and 2 using ``conda``
---------------------------------

As of Pyrocko 2017.11, pre-built packages are available for Linux 64-Bit and MacOS. You can use can use the ``conda`` package manager to install Pyrocko framwork:

.. code-block:: bash
    :caption: Pre-build Pyrocko packages are available for Anaconda3 and 2 on Linux64 and OSX

    conda install -c pyrocko pyrocko

More information available at https://anaconda.org/pyrocko/pyrocko

Anaconda: update to latest Pyrocko version (master branch)
----------------------------------------------------------

**All dependencies should be resolved by a previous conda install**. You can
then go ahead and use ``pip`` to update Pyrocko from source

.. code-block:: bash
    :caption: Anaconda's ``pip`` can install straight from the Git repository

    pip install git+https://git.pyrocko.org/pyrocko/pyrocko.git


Anaconda (compilation from source)
-----------------------------------

.. code-block:: bash
    :caption: Compile from sources

    conda install pyqt=5
    conda install progressbar
    cd `conda info --root`

    mkdir src
    cd src
    git clone https://git.pyrocko.org/pyrocko/pyrocko.git
    cd pyrocko
    python setup.py install

For instructions on how to install Pyrocko on other systems or if the
installation with the above procedure fails, see :doc:`index` or
:doc:`/install/details`.
