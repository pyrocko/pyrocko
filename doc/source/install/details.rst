Detailed installation instructions
==================================

Pyrocko can be installed on every operating system where its prerequisites are
available. This document describes how to install Pyrocko on Unix-like
operating systems, like Linux and Mac OS X.

Concrete listings of the commands needed to install Pyrocko are given
in section

* :doc:`system/index`

Prerequisites
-------------

The following software packages must be installed before Pyrocko can be installed:

* Try to use normal system packages for these Python modules:
   * `Python <http://www.python.org/>`_ (>= 2.6 and >=3.4, with development headers)
   * `NumPy <http://numpy.scipy.org/>`_ (>= 1.6, with development headers)
   * `SciPy <http://scipy.org/>`_
   * `matplotlib <http://matplotlib.sourceforge.net/>`_
   * `pyyaml <https://bitbucket.org/xi/pyyaml>`_
   * `PyQt4 <http://www.riverbankcomputing.co.uk/software/pyqt/intro>`_ (only needed for the GUI apps)
   * `future <https://pypi.python.org/pypi/future>`_ (Python2/3 compatibility layer)
   * `requests <http://docs.python-requests.org/en/master/>`_ (HTTP for Humans)

* Optional Python modules:
   * `progressbar2 <http://pypi.python.org/pypi/progressbar2>`_
   * `GMT <http://gmt.soest.hawaii.edu/>`_ (only required for the :py:mod:`pyrocko.plot.automap` module)
   * `Jinja2 <http://jinja.pocoo.org/>`_ (only required for the ``fomosto report`` subcommand)
   * `nosetest <https://pypi.python.org/pypi/nose>`_ (for unittesting)
   * `coverage <https://pypi.python.org/pypi/coverage>`_ (unittest coverage report)

* Manually install these seismological software tools:
   * `slinktool <http://www.iris.edu/data/dmc-seedlink.htm>`_ (optionally, if you want to use the :py:mod:`pyrocko.streaming.slink` module)
   * `rdseed <http://www.iris.edu/software/downloads/rdseed_request.htm>`_ (optionally, if you want to use the :py:mod:`pyrocko.io.rdseed` module)
   * `QSEIS <http://kinherd.org/fomosto-qseis-2006a.tar.gz>`_ (optional, needed for the Fomosto ``qseis.2006a`` backend)
   * `QSSP <http://kinherd.org/fomosto-qssp-2010.tar.gz>`_ (optional, needed for the Fomosto ``qssp.2010`` backend)

The names of the system packages to be installed differ from system to system.
Whether there are separate packages for the development headers of NumPy and
Python (the \*-dev packages) is also system specific.


Download and install Pyrocko
----------------------------

Use *git* to download the software package and the included script *setup.py*
to install::

    cd ~/src/   # or wherever you keep your source packages
    git clone https://github.com/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    sudo python setup.py install

**Note:** If you have previously installed pyrocko using other tools like e.g.
*easy_install*, you should manually remove the old installation - otherwise you
will end up with two parallel installations of Pyrocko which will cause
trouble.

Updating
--------

If you later would like to update Pyrocko, run the following commands (this
assumes that you have used *git* to download Pyrocko):: 

    cd ~/src/pyrocko   # assuming the Pyrocko software package is here
    git pull origin master 
    sudo python setup.py install
