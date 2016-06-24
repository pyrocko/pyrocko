Installation on Mac OS X systems
................................

* **Mac OS X** (10.6 - 10.10) with **MacPorts** (2.3.3)::
  
    sudo port install git
    sudo port install python27
    sudo port select python python27
    sudo port install py27-numpy
    sudo port install py27-scipy
    sudo port install py27-matplotlib
    sudo port install py27-yaml
    sudo port install py27-pyqt4
    sudo port install py27-setuptools
    sudo easy_install pyavl
    sudo easy_install progressbar
    cd ~/src/   # or wherever you keep your source packages
    git clone git://github.com/emolch/pyrocko.git pyrocko
    cd pyrocko
    sudo python setup.py install --install-scripts=/usr/local/bin
