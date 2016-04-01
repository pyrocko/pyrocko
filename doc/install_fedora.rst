Installation on Fedora systems
..............................

* **Fedora** (20)::

    sudo yum install make git python python-yaml python-matplotlib numpy scipy PyQt4
    sudo easy_install progressbar
    sudo easy_install pyavl
    cd ~/src/   # or wherever you keep your source packages
    git clone git://github.com/emolch/pyrocko.git pyrocko
    cd pyrocko
    sudo python setup.py install
