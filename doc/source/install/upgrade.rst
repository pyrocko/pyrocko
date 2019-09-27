Upgrading Pyrocko from v0.3 to v2017.11 or later
================================================

This document describes details of the upgrade from Pyrocko version **v0.3** to
**v2017.11** or later.

Starting with v2017.11,  **Pyrocko supports Python 3**. Additionally, Pyrocko's
internal module hierarchy has changed (see :doc:`../library/index`) - The API
is backward-compatible so your old scripts will still work. **Under Python 2,
upgrading the framework requires purging of the old v0.3 version and installing
some new dependencies**. Pyrocko will continue to work under Python 2.


Python 3
--------

Nothing special has to be considered if you want to start working with Python 3.
Simply do a fresh install of Pyrocko as described in :doc:`index`.

Python 2
--------

If you have previously installed Pyrocko in an older version under Python 2,
please follow the steps in the appropriate section below for a clean new
installation.

Anaconda2
.........

If you have installed Pyrocko v0.3 under Anaconda2 you have to purge the
old installation:

.. code-block:: bash

    conda remove --all pyrocko

Then follow the instructions in :doc:`packages/anaconda`.

Ubuntu, Debian, Mint, ...
.........................

Following the steps described in :doc:`system/deb` will **install the new
dependencies**.

Pyrocko's ``setup.py`` installer recognizes earlier installations of Pyrocko.
If there is a conflict it will tell you how to **purge your old installation**
when you try to run ``python setup.py install``.

.. code-block:: bash
    :caption: Ubuntu, Debian, etc.: install new prerequisites and purge old install

    # upgrade to PyQt5 and other depedencies
    sudo apt-get install -y python-requests python-pyqt5 python-future
    sudo apt-get install -y python-pyqt5 python-pyqt5.qtopengl python-pyqt5.qtsvg
    sudo apt-get install -y python-pyqt5.qtwebengine || sudo apt-get install -y python-pyqt5.qtwebkit

    # clean Git clone
    cd src
    rm -rf pyrocko
    git clone https://git.pyrocko.org/pyrocko/pyrocko.git
    cd pyrocko
    sudo python setup.py install  # on failure, reports <directory> to remove
    rm -rf <directory>
    sudo python setup.py install 


Mac OS X with MacPorts
......................

If you had Pyrocko v0.3 installed through MacPorts you have to delete that
installation:

.. code-block:: bash
    :caption: MacPorts has to install new dependencies 

    sudo rm -rf /opt/local/lib/python2.7/dist-packages/pyrocko*
    sudo port install py27-pyqt5 py27-requests py27-future

    cd ~/src/  # or wherever you keep your source packages
    rm -rf pyrocko
    git clone git://git.pyrocko.org/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    sudo python setup.py install --install-scripts=/usr/local/bin

More information here :doc:`system/mac`.
