Upgrading Pyrocko from v0.3
===========================

If you have **Pyrocko v0.3** installed and want to upgrade to **Pyrocko 2017.11** you are in the right place! With the upgrade the modules' structure changed (see :doc:`../library/index`) - The API is backwards-compatible so your old scripts still work. Upgrading the framework requires purging of the old v0.3 version and installing some new dependencies.

.. raw:: html

    <h2>Upgrade on Pyhon 3</h2>

If you want to start working with Python 3 (which is recommended) you can do a new install of Pyrocko. Installation instructions for various platforms and systems are accessible here :doc:`index`.

Upgrading on Anaconda2
-------------------------

.. note::
    Pre-build Pyrocko packages for Anaconda3 are available from version 2017.11!

Only if you built Pyrocko v0.3 for Anaconda2 you have to purge the old installation:

.. code-block:: bash
    :caption: Upgrade on Anaconda2

    conda remove --all pyrocko

Then follow the install instructions here :doc:`packages/anaconda`.

Upgrading on Ubuntu, Debian or Mint
------------------------------------

The ``setup.py`` installer identifies earlier installations of Pyrocko, if there is a conflict you have to delete the old pyrocko directory and upgrade the dependencies:

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

Upgrading on Mac OS X using MacPorts
-------------------------------------

If you had Pyrocko v0.3 installed throug MacPorts you have to delete that installation:

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
