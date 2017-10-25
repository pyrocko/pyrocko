Installation on Debian based systems (Debian, Ubuntu, Mint)
...........................................................

* Python 2 Installation on **Ubuntu** (12.04.1 LTS), **Debian** (7 wheezy), **Debian** (8 jessie), **Mint** (13 Maya)::

    sudo apt-get install make git python-dev python-setuptools
    sudo apt-get install python-numpy python-numpy-dev python-scipy python-matplotlib
    sudo apt-get install python-pyqt5 python-pyqt5.qtwebengine python-pyqt5.qtopengl python-pyqt5.qtsvg
    sudo apt-get install python-future python-requests
    sudo apt-get install python-yaml python-progressbar python-jinja2
    cd ~/src/   # or wherever you keep your source packages
    git clone git://github.com/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    sudo python setup.py install

* Python 3 Installation on **Ubuntu** (12.04.1 LTS), **Debian** (7 wheezy), **Debian** (8 jessie), **Mint** (13 Maya)::

    sudo apt-get install make git python3-dev python3-setuptools
    sudo apt-get install python3-numpy python3-numpy-dev python3-scipy python3-matplotlib
    sudo apt-get install python3-pyqt5 python3-pyqt5.qtwebengine python3-pyqt5.qtopengl python3-pyqt5.qtsvg
    sudo apt-get install python3-yaml python3-progressbar python3-jinja2
    cd ~/src/   # or wherever you keep your source packages
    git clone git://github.com/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    sudo python3 setup.py install

For instructions on how to install Pyrocko on other systems or if the
installation with the above procedure fails, see :doc:`index` or
:doc:`/install/details`.
