Installation on Ubuntu, Debian, Mint, ...
=========================================

These example instructions should work for a system-wide installation of
Pyrocko under deb-based Linuxes with Python 3.

.. code-block:: bash
    :caption: e.g. **Ubuntu** (14.04, 16.04, 18.04, 20.04), **Debian** (8, 9, 10), **Mint** (17, 18, 19, 20)

    sudo apt-get install -y make git python3-dev python3-setuptools
    sudo apt-get install -y python3-yaml python3-progressbar python3-jinja2
    sudo apt-get install -y python3-requests
    sudo apt-get install -y python3-numpy python3-numpy-dev python3-scipy python3-matplotlib
    sudo apt-get install -y python3-pyqt5 python3-pyqt5.qtopengl python3-pyqt5.qtsvg
    # the following may emit an error message which can be ignored
    sudo apt-get install -y python3-pyqt5.qtwebengine || sudo apt-get install -y python3-pyqt5.qtwebkit

    cd ~/src/   # or wherever you keep your source packages
    git clone https://git.pyrocko.org/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    sudo python3 setup.py install

For instructions on how to install Pyrocko on other systems or if the
installation with the above procedure fails, see :doc:`../index` or
:doc:`/install/details`.
