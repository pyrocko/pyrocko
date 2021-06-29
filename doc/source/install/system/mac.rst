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

.. note::

    This information is slightly outdated. There should be a newer version
    of Python 3 available with MacPorts. Help on updating the docs is welcome!

.. code-block:: bash
    :caption: e.g. **Mac OS X** (10.6 - 10.10) with **MacPorts** (2.4.2)

    sudo port install git
    sudo port install python35
    sudo port select python python35
    sudo port install py35-numpy py35-scipy py35-matplotlib py35-yaml py35-pyqt5 py35-setuptools py35-jinja2 py35-requests
    sudo easy_install progressbar

    cd ~/src/   # or wherever you keep your source packages
    git clone https://git.pyrocko.org/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    sudo python3 setup.py install --install-scripts=/usr/local/bin

For instructions on how to install Pyrocko on other systems or if the
installation with the above procedure fails, see :doc:`index` or
:doc:`/install/details`.
