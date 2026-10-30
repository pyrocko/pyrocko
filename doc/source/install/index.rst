Installation
============

Pyrocko can be installed on any operating system where its prerequisites are
available.

Under **Linux**, we recommend to use the system's **native package manager** to
install the prerequisites and to install Pyrocko **from source**. Using
third-party package managers like *pip* or *conda* often leads to
conflicts. Follow the instructions in :doc:`system/index`.

For **Anaconda** users, prebuilt packages are available (see
:doc:`packages/anaconda`). Also packages for **pip** are available
(see :doc:`packages/pip`).

For additional information, refer to :doc:`details`.

Quick installation for trying out Pyrocko
-----------------------------------------

To just try out Pyrocko under Linux or macOS, without touching the rest of
your system, run in a *bash* or *zsh* terminal:

.. code-block:: sh

    source <(curl -fsSL https://pyrocko.org/pvc_init) && pvc install

This installs the latest release of Pyrocko from PyPI into a virtual
environment in ``~/.pyrocko/venv``. Afterwards, run ``pvc activate`` to use it
in a terminal, or ``pvc`` for help. It needs Python 3.10 or newer, with the
``venv`` module (on Debian and Ubuntu, install the package ``python3-venv``).

.. toctree::
   :maxdepth: 2

   system/index
   packages/index
   details

Support
-------

If you run into problems during installation, please let us know via `Pyrocko
Issues <https://git.pyrocko.org/pyrocko/pyrocko/issues>`_ or join our `support forum
<https://hive.pyrocko.org/>`_.


