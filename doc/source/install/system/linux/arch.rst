Installation on Arch Linux systems
==================================

These example instructions are for a system-wide installation of Pyrocko under
Arch Linux with default Python 3.

.. code-block:: bash
    :caption: **Arch Linux** (e.g. 2021.06.01)

    sudo pacman -Syu git make gcc patch python python-setuptools \
        python-numpy python-scipy python-matplotlib \
        python-pyqt5 qt5-webengine qt5-svg qt5-webkit \
        python-cairo python-opengl python-progressbar \
        python-requests python-yaml python-jinja \
        python-nose python-coverage

    cd ~/src/   # or wherever you keep your source packages
    git clone https://git.pyrocko.org/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    sudo python setup.py install

For instructions on how to install Pyrocko on other systems or if the
installation with the above procedure fails, see :doc:`../index` or
:doc:`/install/details`.
