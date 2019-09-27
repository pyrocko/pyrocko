Installation on Ubuntu, Debian, Mint, ...
=========================================

Python 3.4 and later
--------------------

.. code-block:: bash
    :caption: e.g. **Ubuntu** (14.04.1 LTS), **Debian** (7 wheezy, 8 jessie), **Mint** (17 - 18.2 Sonya)

    sudo apt-get install -y make git python3-dev python3-setuptools
    sudo apt-get install -y python3-numpy python3-numpy-dev python3-scipy python3-matplotlib
    sudo apt-get install -y python3-pyqt4 python3-pyqt4.qtopengl
    sudo apt-get install -y python3-pyqt5 python3-pyqt5.qtopengl python3-pyqt5.qtsvg
    sudo apt-get install -y python3-pyqt5.qtwebengine || sudo apt-get install -y python3-pyqt5.qtwebkit
    sudo apt-get install -y python3-yaml python3-progressbar python3-jinja2
    sudo apt-get install -y python3-requests
    sudo apt-get install -y python3-future || sudo easy_install3 future
    cd ~/src/   # or wherever you keep your source packages
    git clone https://git.pyrocko.org/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    sudo python3 setup.py install

Python 2.7
----------

.. code-block:: bash
    :caption: e.g. **Ubuntu** (14.04.1 LTS), **Debian** (7 wheezy, 8 jessie), **Mint** (13 Maya)

    sudo apt-get install -y make git python-dev python-setuptools
    sudo apt-get install -y python-numpy python-numpy-dev python-scipy python-matplotlib
    sudo apt-get install -y python-pyqt5 python-pyqt5.qtopengl python-pyqt5.qtsvg
    sudo apt-get install -y python-pyqt5.qtwebengine || sudo apt-get install -y python-pyqt5.qtwebkit
    sudo apt-get install -y python-yaml python-progressbar python-jinja2
    sudo apt-get install -y python-requests
    sudo apt-get install -y python-future || sudo easy_install future
    cd ~/src/   # or wherever you keep your source packages
    git clone https://git.pyrocko.org/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    sudo python setup.py install


Python 2.7 user local installation (no sudo)
--------------------------------------------


For local installations of python modules, please first configure your
``PYTHONPATH`` and ``PYTHONUSERBASE`` variables in your environmment or your
``~/.bashrc``::

    export PYTHONUSERBASE=<path_to_local_python>
    export PYTHONPATH=:<path_to_local_python_site-packages>:$PYTHONPATH

Then install local Python Module dependencies and Pyrocko locally:

.. code-block:: bash
    :caption: e.g. **Ubuntu** (12.04.1 LTS), **Debian** (7 wheezy), **Mint** (13 Maya)

    pip install --user py27-numpy
    pip install --user py27-scipy
    pip install --user py27-matplotlib
    pip install --user py27-yaml
    pip install --user py27-pyqt4
    pip install --user py27-setuptools
    pip install --user py27-jinja2
    easy_install --user progressbar || pip install --user progressbar
    pip install --user Jinja2 
    cd ~/src/   # or wherever you keep your source packages   
    git clone https://git.pyrocko.org/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    python setup.py install --user --install-scripts=<path_to_your_local_binaries>

For instructions on how to install Pyrocko on other systems or if the
installation with the above procedure fails, see :doc:`index` or
:doc:`/install/details`.
