Installation on Arch Linux systems
==================================

Python 3.4 and later
--------------------

.. code-block:: bash
    :caption: **Arch Linux** (e.g. 2017.11.01)

    sudo pacman -Syu git make gcc python python-setuptools \
        python-numpy python-scipy python-matplotlib \
        python-pyqt5 qt5-webengine qt5-svg python-pyqt4 \
        python-cairo python-opengl python-progressbar \
        python-requests python-yaml python-jinja \
        python-nose python-coverage

    cd ~/src/   # or wherever you keep your source packages
    git clone https://git.pyrocko.org/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    sudo python setup.py install

For instructions on how to install Pyrocko on other systems or if the
installation with the above procedure fails, see :doc:`index` or
:doc:`/install/details`.
