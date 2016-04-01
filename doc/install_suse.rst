Installation on OpenSuse systems
................................

* **OpenSuse** (13)::

    sudo zypper install make git python-devel python-setuptools
    sudo zypper install python-numpy python-numpy-devel python-scipy python-matplotlib
    sudo zypper install python-qt4
    sudo zypper install python-PyYAML python-progressbar
    sudo easy_install pyavl
    cd ~/src/   # or wherever you keep your source packages
    git clone git://github.com/emolch/pyrocko.git pyrocko
    cd pyrocko
    sudo python setup.py install
