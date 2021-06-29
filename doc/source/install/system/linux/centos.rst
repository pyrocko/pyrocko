Installation on Centos and other rpm-based systems
--------------------------------------------------

These example instructions are for a system-wide installation of Pyrocko under
Centos 8 with Python 3. Slight variations may work on other rpm-based systems.

.. code-block:: bash
    :caption: e.g. **Centos 8**, Python 3

    sudo yum install epel-release dnf-plugins-core
    sudo yum config-manager --set-enabled powertools

    sudo yum install make gcc git python3 python3-yaml python3-matplotlib
    sudo yum install python3-numpy python3-scipy python3-requests
    sudo yum install python3-jinja2 python3-qt5 python3-matplotlib-qt5

    cd ~/src/   # or wherever you keep your source packages
    git clone https://git.pyrocko.org/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    sudo python3 setup.py install

For instructions on how to install Pyrocko on other systems or if the
installation with the above procedure fails, see :doc:`../index` or
:doc:`/install/details`.
