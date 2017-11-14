Upgrading Pyrocko from v0.3 to v2017.11 or later
================================================

This document describes details of the upgrade from Pyrocko version **v0.3** to
**v2017.11** or later. Starting with v2017.11,  **Pyrocko supports Python 3**.

With the upgrade the module hierarchy changed (see :doc:`../library/index`) -
The API is backwards-compatible so your old scripts still work. Upgrading the
framework requires purging of the old v0.3 version and installing some new
dependencies.


Python 3
--------

Nothing special need to be considered when you want to start working with Python 3.
Simply do a fresh install of Pyrocko as described in :doc:`index`.

Python 2
--------

If you have previously installed Pyrocko in an older version on Python 2,
please follow the steps below for a clean new installation.

Anaconda2
.........

.. note::
    Prebuilt Pyrocko packages for Anaconda3 are available from version 2017.11!

Only if you built Pyrocko v0.3 for Anaconda2 you have to purge the old installation:

.. code-block:: bash
    :caption: Upgrade on Anaconda2

    conda remove --all pyrocko

Then follow the install instructions here :doc:`packages/anaconda`.

Ubuntu, Debian, Mint, ...
.........................

The ``setup.py`` installer identifies earlier installations of Pyrocko, if
there is a conflict you have to delete the old pyrocko directory and upgrade
the dependencies:

.. code-block:: bash
    :caption: We have to install new dependencies

    sudo rm -rf /usr/local/lib/python2.7/dist-packages/pyrocko*

    # Upgrade to PyQt5 and other depedencies
    sudo apt-get install -y python-requests python-pyqt5 python-future
    sudo apt-get install -y python-pyqt5 python-pyqt5.qtopengl python-pyqt5.qtsvg
    sudo apt-get install -y python-pyqt5.qtwebengine || sudo apt-get install -y python-pyqt5.qtwebkit

    # Clean GitHub clone
    cd src
    rm -rf pyrocko
    git clone https://github.com/pyrocko/pyrocko.git
    cd pyrocko
    sudo setup.py install

More information here :doc:`system/deb` or :doc:`packages/pip`.

Mac OS X with MacPorts
......................

If you had Pyrocko v0.3 installed throug MacPorts you have to delete that
installation:

.. code-block:: bash
    :caption: MacPorts has to install new dependencies 

    sudo rm -rf /opt/local/lib/python2.7/dist-packages/pyrocko*
    sudo port install py27-pyqt5 py27-requests py27-future

    cd ~/src/  # or wherever you keep your source packages
    rm -rf pyrocko
    git clone git://github.com/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    sudo python setup.py install --install-scripts=/usr/local/bin

More information here :doc:`system/mac`.
