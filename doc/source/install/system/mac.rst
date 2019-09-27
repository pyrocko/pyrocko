Installation on Mac OS X systems
================================

On the Mac, several different package managers are available to ease the
installation of popular open source software tools and libraries. `Anaconda
<https://www.anaconda.com/>`_, `MacPorts <https://www.macports.org/>`_, `Fink
<http://www.finkproject.org/>`_, and `HomeBrew <https://brew.sh/>`_ are among
the most popular choices. To prevent trouble, pick one and stay with one.

Mac OS X with Anaconda 
----------------------

If you are using Anaconda under Mac OS X, see
:doc:`/install/packages/anaconda`.

Mac OS X with MacPorts (Python 3)
----------------------------------

.. code-block:: bash
    :caption: e.g. **Mac OS X** (10.6 - 10.10) with **MacPorts** (2.4.2)

    sudo port install git
    sudo port install python35
    sudo port select python python35
    sudo port install py35-numpy py35-scipy py35-matplotlib py35-yaml py35-pyqt5 py35-setuptools py35-jinja2 py35-requests py35-future
    sudo easy_install progressbar
    cd ~/src/   # or wherever you keep your source packages
    git clone https://git.pyrocko.org/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    sudo python setup.py install --install-scripts=/usr/local/bin


Mac OS X with MacPorts (Python 2.7)
-----------------------------------

.. code-block:: bash
    :caption: e.g. **Mac OS X** (10.6 - 10.10) with **MacPorts** (2.4.2)

    sudo port install git
    sudo port install python27
    sudo port select python python27
    sudo port install py27-numpy py27-scipy py27-matplotlib py27-yaml py27-pyqt5 py27-setuptools py27-jinja2 py27-requests py27-future
    sudo easy_install progressbar

    cd ~/src/  # or wherever you keep your source packages
    git clone https://git.pyrocko.org/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    sudo python setup.py install --install-scripts=/usr/local/bin

Mac OS X with MacPorts and user installation (no sudo)
-------------------------------------------------------

Try this if you don't have sudo rights in ``/usr/bin``.

.. code-block:: bash
    :caption: e.g. **Mac OS X** (10.6 - 10.10) with **MacPorts** (2.4.2)

    port install git
    port install python35
    port select python python35
    port install py35-numpy py35-scipy py35-matplotlib py35-yaml py35-pyqt5 py35-setuptools py35-jinja2 py35-requests py35-future

For local installations of python modules, please first configure your
``PYTHONPATH`` and ``PYTHONUSERBASE`` variables in your environment or your
``~/.bash_profile``::

    export PYTHONUSERBASE=<path_to_local_python>
    export PYTHONPATH=:<path_to_local_python_site-packages>:$PYTHONPATH

Then install local Python module prerequisites and Pyrocko locally.
Depending on your system's default Python version, you may have to install and
use pip3 instead of pip.

.. code-block:: bash
    :caption: e.g. **Mac OS X** (10.6 - 10.10) with **MacPorts** (2.3.3)

    pip install --user progressbar
    pip install --user Jinja2 
    easy_install --user pyavl
    cd ~/src/   # or wherever you keep your source packages   
    git clone https://git.pyrocko.org/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    python setup.py install --user --install-scripts=<path_to_your_local_binaries>

For instructions on how to install Pyrocko on other systems or if the
installation with the above procedure fails, see :doc:`index` or
:doc:`/install/details`.
