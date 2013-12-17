Installation
============

Pyrocko can be installed on every operating system where its prerequisites are
available. This document describes how to install Pyrocko on Unix-like
operating systems, like Linux and MacOSX.

Prerequisites
-------------

The following software packages must be installed before Pyrocko can be installed:

* Try to use normal system packages for these:
   * `Python <http://www.python.org/>`_ (>= 2.6, < 3.0, with development headers)
   * `NumPy <http://numpy.scipy.org/>`_ (with development headers)
   * `SciPy <http://scipy.org/>`_
   * `matplotlib <http://matplotlib.sourceforge.net/>`_
   * `pyyaml <https://bitbucket.org/xi/pyyaml>`_
   * `PyQt4 <http://www.riverbankcomputing.co.uk/software/pyqt/intro>`_ (only needed for the GUI apps)
   * `progressbar <http://pypi.python.org/pypi/progressbar>`_ (optional)

* Try to use `easy_install <http://pythonhosted.org/setuptools/easy_install.html>`_ or `pip install <http://www.pip-installer.org/en/latest/installing.html>`_ for these:
   * `pyavl <http://pypi.python.org/pypi/pyavl/>`_ 
   * `guts <http://pypi.python.org/pypi/guts/>`_

* Manually install these:
   * `slinktool <http://www.iris.edu/data/dmc-seedlink.htm>`_ (optionally, if you want to use the :py:mod:`pyrocko.slink` module)
   * `rdseed <http://www.iris.edu/software/downloads/rdseed_request.htm>`_ (optionally, if you want to use the :py:mod:`pyrocko.rdseed` module)

The names of the system packages to be installed differ from system to system.
Whether there are separate packages for the development headers of NumPy and
Python (the \*-dev packages) is also system specific.

Here some details, what to install on a few popular distributions:

* **Ubuntu** (12.04.1 LTS), **Debian** (7 wheezy), **Mint** (13 Maya)::

    sudo apt-get install git python-dev python-setuptools
    sudo apt-get install python-numpy python-numpy-dev python-scipy python-matplotlib
    sudo apt-get install python-qt4 python-qt4-gl 
    sudo apt-get install python-yaml python-progressbar
    sudo easy_install pyavl
    sudo easy_install guts

* **Fedora** (*outdated, can someone check this?*)::

    sudo yum install python numpy scipy PyQt4
    sudo easy_install progressbar
    sudo easy_install pyavl
    sudo easy_install guts


Download and install Pyrocko
----------------------------

**Either:** the easy way, using *easy_install*::

    sudo easy_install http://github.com/emolch/pyrocko/tarball/master

**Or:** the proper way, using *git* and *setup.py*::

    cd ~/src/   # or wherever you keep your source packages
    git clone git://github.com/emolch/pyrocko.git pyrocko
    cd pyrocko
    sudo python setup.py install

**Warning:** If you switch from one installation method to the other, you have
to manually remove the old installation - otherwise you will end up with two
parallel installations of Pyrocko which will cause trouble.

Updating
--------

If you later want to update Pyrocko, run the following commands (this assumes
that you have used *git* to download Pyrocko):: 

    cd ~/src/pyrocko   # or wherever the Pyrocko source are 
    git pull origin master 
    sudo python setup.py install  

