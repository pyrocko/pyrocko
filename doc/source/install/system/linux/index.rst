
Installation under Linux
========================

This section lists the commands to install Pyrocko and its prerequisites on
most popular Linux distributions.

System-wide installation from source
------------------------------------

.. code-block:: bash
    :caption: **Any Linux**

    cd ~/src/   # or wherever you keep your source packages
    git clone https://git.pyrocko.org/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    python install.py deps system  # installs prerequisites using apt/yum/pacman
    python install.py system       # installs Pyrocko with pip, but uses system deps

User installation from source into isolated environment
-------------------------------------------------------

.. code-block:: bash
    :caption: **Any Linux**

    cd ~/src/   # or wherever you keep your source packages
    git clone https://git.pyrocko.org/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    python -m venv myenv
    source myenv/bin/activate
    pip install .                          # pip auto-resolves prerequisites (!)
    pip install --only-binary :all: PyQt5 PyQtWebEngine # for Snuffler


User installation from source using system packages for the prerequisites
-------------------------------------------------------------------------

.. code-block:: bash
    :caption: **Any Linux**

    cd ~/src/   # or wherever you keep your source packages
    git clone https://git.pyrocko.org/pyrocko/pyrocko.git pyrocko
    cd pyrocko
    python -m venv --use-system-packages myenv
    source myenv/bin/activate
    python install.py deps system  # installs prerequisites using apt/yum/pacman
    python install.py user         # installs Pyrocko with pip, but uses system deps

Distribution specific details and prerequisite package lists
------------------------------------------------------------

.. toctree::
   :maxdepth: 1

   Ubuntu, Debian, Mint, ... (deb based) <deb>
   Fedora, Centos, ... (rpm based) <rpm>
   OpenSuse <suse>
   Arch <arch>

For instructions on how to install Pyrocko on other systems or if the
installation with any of the above procedures fails, see :doc:`../index` or
:doc:`/install/details`.
