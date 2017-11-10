Installation under Anaconda
===========================

`Anaconda <https://www.anaconda.com/>`_, is a cross-platform Python
distribution with its own package manager ``conda``. There are two versions:
Anaconda2 for Python 2 and Anaconda3 for Python 3. Pyrocko can be installed
under either of them.


Anaconda3 (using ``conda`` package manager)
-------------------------------------------

Pre-built packages are available for Linux 64-Bit and MacOS. Use can use the
``conda`` package manager to install Pyrocko::

    conda install -c pyrocko pyrocko

More information available at https://anaconda.org/pyrocko/pyrocko

Anaconda2 (compilation from source)
-----------------------------------

::

    conda install pyqt=5    # downgrades some packages
    easy_install pyavl
    pip install progressbar
    cd `conda info --root`
    mkdir src
    cd src
    git clone https://github.com/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    python setup.py install

For instructions on how to install Pyrocko on other systems or if the
installation with the above procedure fails, see :doc:`index` or
:doc:`/install/details`.
