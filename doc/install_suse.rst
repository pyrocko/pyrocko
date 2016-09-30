Installation on OpenSuse systems
................................

* **OpenSuse** (13)::

    sudo zypper install make git python-devel python-setuptools
    sudo zypper install python-numpy python-numpy-devel python-scipy python-matplotlib
    sudo zypper install python-qt4
    sudo zypper install python-PyYAML python-progressbar
    sudo easy_install pyavl
    cd ~/src/   # or wherever you keep your source packages
    git clone git://github.com/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    sudo python setup.py install

For instructions on how to install Pyrocko on other systems or if the
installation with the above procedure fails, see :doc:`install_quick` or
:doc:`install_details`.
