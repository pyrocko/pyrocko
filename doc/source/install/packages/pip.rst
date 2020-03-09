Installation with pip
=====================

When installing Pyrocko through ``pip`` we do not allow the installer to
resolve dependencies automatically. We think it is up to you, to decide which
prerequisites to install with the system's native package manager and which
ones to install through ``pip``.

Install from PyPI (Python Package Index)
----------------------------------------

Tagged builds are available for download from https://pypi.python.org/.

.. note :: 

    Pyrocko's build depends on Python and NumPy development header files,
    please install those through your system's package manager. Please see
    :doc:`../system/index` for more information.

.. warning ::
    
    Dependency for **Qt** is not resolved, please use your system package manager to install _PyQt4_ or _PyQt5_!
    For more information see :doc:`../system/index`

.. code-block:: bash
    :caption: Source packages for Pyrocko are available on `pypi.python.org <https://pypi.python.org>`_

    # Install build requirements
    sudo apt-get install python3-dev python3-numpy
    sudo pip install pyrocko

    # Install requirements
    sudo pip install numpy>=1.8 scipy pyyaml matplotlib progressbar2 jinja2 requests PyOpenGL


User local installation (no sudo)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to install Pyrocko in a user environment without root access you
can use ``pip`` to manage the installation as well:

.. code-block:: bash
    :caption: ``pip`` allows to install into user's home directory - **dangerous for system integrity**

    # We assume the build requirements are already installed, see above
    pip install pyrocko

    # Install requirements
    pip install numpy>=1.8 scipy pyyaml matplotlib progressbar2 jinja2 requests PyOpenGL


Install latest version with pip
-------------------------------

If you want to install or update to the latest version of Pyrocko available on
`git.pyrocko.org <https://git.pyrocko.org/pyrocko/pyrocko/>`_ (``master``
branch) you can use ``pip`` to install directly from the repository:

.. code-block:: bash
    :caption: We install straight from the Git repository

    sudo pip install git+https://git.pyrocko.org/pyrocko/pyrocko.git
