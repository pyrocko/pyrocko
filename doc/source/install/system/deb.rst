Installation on Debian based systems (Debian, Ubuntu, Mint)
===========================================================

Normal installation (with sudo)
-------------------------------

* **Ubuntu** (12.04.1 LTS), **Debian** (7 wheezy), **Mint** (13 Maya)::

    sudo apt-get install make git python-dev python-setuptools
    sudo apt-get install python-numpy python-numpy-dev python-scipy python-matplotlib
    sudo apt-get install python-qt4 python-qt4-gl 
    sudo apt-get install python-yaml python-progressbar python-jinja2
    sudo easy_install pyavl
    cd ~/src/   # or wherever you keep your source packages
    git clone git://github.com/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    sudo python setup.py install


Local installation (no sudo rights in /usr/local/bin)
-----------------------------------------------------

* **Ubuntu** (12.04.1 LTS), **Debian** (7 wheezy), **Mint** (13 Maya)

For local installations of python modules, please first configure your PYTHONPATH and PYTHONUSERBASE variables in your environmment or your .bashrc::

    export PYTHONUSERBASE='path_to_local_python'
    export PYTHONPATH=:'path_to_local_python_site-packages':$PYTHONPATH

Then install local Python Module dependencies and Pyrocko locally::

    pip install --user py27-numpy
    pip install --user py27-scipy
    pip install --user py27-matplotlib
    pip install --user py27-yaml
    pip install --user py27-pyqt4
    pip install --user py27-setuptools
    pip install --user py27-jinja2
    easy_install --user pyavl
    easy_install --user progressbar
    pip install --user progressbar
    pip install --user Jinja2 
    easy_install --user pyavl
    cd ~/src/   # or wherever you keep your source packages   
    git clone git://github.com/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    python setup.py install --user --install-scripts="path_to_your_local_binaries"

For instructions on how to install Pyrocko on other systems or if the
installation with the above procedure fails, see :doc:`index` or
:doc:`/install/details`.
