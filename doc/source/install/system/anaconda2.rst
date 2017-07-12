Installation under Anaconda2
............................

* **Anaconda2** (4.2.0)::

    conda install pyqt=4    # downgrades some packages
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
