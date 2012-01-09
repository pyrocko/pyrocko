Installation
============


Prerequisites
-------------

The following software packages are required to use Pyrocko. The more important ones might be available through the package manager on your system (see below).

* `Python <http://www.python.org/>`_ including development headers
* `NumPy <http://numpy.scipy.org/>`_ including development headers
* `SciPy <http://scipy.org/>`_
* `PyQt4 <http://www.riverbankcomputing.co.uk/software/pyqt/intro>`_ (>= v4.4.4, Only needed for the GUI apps)
* libmseed (tarball is included)
* evalresp (only libevresp from this package, tarball is included, could be made optional)
* `progressbar <http://pypi.python.org/pypi/progressbar>`_ (optional)
* `slinktool <http://www.iris.edu/data/dmc-seedlink.htm>`_ (optionally, if you want to use the :py:mod:`pyrocko.slink` module)
* `rdseed <http://www.iris.edu/software/downloads/rdseed_request.htm>`_ (optionally, if you want to use the :py:mod:`pyrocko.rdseed` module)

How to install prerequisites available through the package manager of your system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The exact package names may differ from system to system. Whether there are
separate packages for the development headers of NumPy and Python (the \*-dev
packages) is also system specific.

* Debian GNU/Linux::

    apt-get install python-dev python-numpy python-numpy-dev python-scipy python-qt4 python-qt4-opengl


* Fedora::

    yum install python numpy scipy PyQt4

* Ubuntu::

    sudo apt-get install python-dev python-numpy python-scipy python-qt4 python-qt4-gl python-progressbar


Getting Pyrocko
---------------

The simplest way of downloading Pyrocko is by using Git::

    cd ~/src/   # or wherever you keep your source packages
    git clone git://github.com/emolch/pyrocko.git pyrocko

Alternatively, you may download Pyrocko as a `tar archive
<http://github.com/emolch/pyrocko/tarball/master>`_, but updating is easier
with the method described above.

Installing the included prerequisites (libmseed and libevresp)
--------------------------------------------------------------

If you already have these libraries installed, these steps might not be necessary.

First compile libmseed. Its tarball is included in the top directory of Pyrocko::

    cd pyrocko/
    tar -xzvf libmseed-2.5.1.tar.gz
    cd libmseed/
    make gcc
    cd ..

Next, compile and install the evalresp library. Its tarball is also included in
the top directory of Pyrocko. You may install it as a shared library (see
below), or try with a static library (if you don't want to install evalresp)::

    tar -xzvf evalresp-3.3.0.tar.gz
    cd evalresp-3.3.0/
    ./configure --enable-shared
    make
    sudo make install
    cd ..

    # now check that $LD_LIBRARY_PATH contains /usr/local/lib
    echo $LD_LIBRARY_PATH 
    # if it is not in there you have to adjust your environment variables

Installing Pyrocko
------------------

Now compile and install Pyrocko itself::

    sudo python setup.py install

Installing Pyrocko to a custom location
---------------------------------------

If you would like to install it to a different location than ``/usr/local`` (or whatever Python thinks it should be), you may use the ``--prefix`` option of ``setup.py``. You will, of course,  have to adjust the environment variables ``$PATH``, ``$PYTHONPATH``, and ``$LD_LIBRARY_PATH`` to use your custom installation::

    python setup.py install --prefix=/bonus


Updating
--------

If you later want to update Pyrocko, run the following from within Pyrocko's top directory:: 

    git pull origin master 
    sudo python setup.py install  

