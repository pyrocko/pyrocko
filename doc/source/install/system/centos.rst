Installation on Centos systems
..............................

.. code-block:: bash
    :caption: e.g. **Centos** (7)

    sudo yum install make gcc git python python-yaml python-matplotlib 
    sudo yum install numpy scipy python-requests python-coverage 
    sudo yum install python-jinja2 PyQt4 python-matplotlib-qt4
    sudo yum install python-future || sudo easy_install future
    sudo easy_install progressbar
    cd ~/src/   # or wherever you keep your source packages
    git clone https://git.pyrocko.org/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    sudo python setup.py install

For instructions on how to install Pyrocko on other systems or if the
installation with the above procedure fails, see :doc:`index` or
:doc:`/install/details`.
