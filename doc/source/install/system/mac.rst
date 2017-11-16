Installation on Mac OS X systems
................................

Superuser installation (sudo rights in /usr/bin)
------------------------------------------------
  
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
    sudo port install py27-jinja2
    sudo easy_install pyavl
    sudo easy_install progressbar
    cd ~/src/   # or wherever you keep your source packages
    git clone git://github.com/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    sudo python setup.py install --install-scripts=/usr/local/bin

Local installation (no sudo rights in /usr/bin)
-----------------------------------------------

* **Mac OS X** (10.6 - 10.10) with **MacPorts** (2.3.3)::

    port install git
    port install python27
    port select python python27
    port install py27-numpy
    port install py27-scipy
    port install py27-matplotlib
    port install py27-yaml
    port install py27-pyqt4
    port install py27-setuptools
    port install py27-jinja2

For local installations of python modules, please first configure your PYTHONPATH and PYTHONUSERBASE variables in your environmment or your .bash_profile::

    export PYTHONUSERBASE='path_to_local_python'
    export PYTHONPATH=:'path_to_local_python_site-packages':$PYTHONPATH

Then install local Python Module dependencies and Pyrocko locally::

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
