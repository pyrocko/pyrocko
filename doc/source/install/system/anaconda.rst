Installation for Anaconda
............................

* **Anaconda3** using ``conda`` package manager (Preferred)::
    
    conda install -c pyrocko pyrocko

Packages are available for Linux 64-Bit and MacOS.
More information available at https://anaconda.org/pyrocko/pyrocko

* **Anaconda2** (4.2.0)::

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
