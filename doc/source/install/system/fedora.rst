Installation on Fedora systems
..............................

* **Fedora** (20)::

    sudo yum install make git python python-yaml python-matplotlib numpy scipy PyQt4
    sudo yum install python-jinja2 python-requests python-future
    sudo easy_install progressbar
    cd ~/src/   # or wherever you keep your source packages
    git clone git://github.com/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    sudo python setup.py install

For instructions on how to install Pyrocko on other systems or if the
installation with the above procedure fails, see :doc:`index` or
:doc:`/install/details`.
