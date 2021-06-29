Installation on OpenSuse systems
================================

These example instructions should work for a system-wide installation of
Pyrocko under OpenSuse Linuxes with Python 3.

.. code-block:: bash
    :caption: **OpenSuse** (e.g. 42.1)

    sudo zypper -n install make git gcc python3-devel python3-setuptools \
        python3-numpy python3-numpy-devel python3-scipy python3-matplotlib \
        python3-qt5 \
        python3-PyYAML python3-progressbar python3-Jinja2 \
        python3-requests

    cd ~/src/   # or wherever you keep your source packages
    git clone https://git.pyrocko.org/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    sudo python3 setup.py install

For instructions on how to install Pyrocko on other systems or if the
installation with the above procedure fails, see :doc:`../index` or
:doc:`/install/details`.
