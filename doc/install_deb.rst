Installation on Debian based systems (Debian, Ubuntu, Mint)
...........................................................

* **Ubuntu** (12.04.1 LTS), **Debian** (7 wheezy), **Mint** (13 Maya)::

    sudo apt-get install make git python-dev python-setuptools
    sudo apt-get install python-numpy python-numpy-dev python-scipy python-matplotlib
    sudo apt-get install python-qt4 python-qt4-gl 
    sudo apt-get install python-yaml python-progressbar
    sudo easy_install pyavl
    cd ~/src/   # or wherever you keep your source packages
    git clone git://github.com/emolch/pyrocko.git pyrocko
    cd pyrocko
    sudo python setup.py install
