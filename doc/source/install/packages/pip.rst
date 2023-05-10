Installation with pip
=====================

Using pip, Pyrocko can be installed from source or binary packages which we
have uploaded to the Python Package Index (`PyPI <https://pypi.org/>`_).

Depending on your attitude, different install variants are possible (see
following sections).

**Good to know:**

- Consequently use ``pip3`` instead of ``pip`` if you want to be sure that
  Python3 versions are installed.
- Add the ``--user`` to all pip commands if you want to install into your home
  directory.
- Consider using `virtual environments
  <https://docs.python.org/3/tutorial/venv.html>`_ when using pip to lower the
  risk of package conflicts.

Variant 1: allow pip to resolve dependencies
--------------------------------------------

.. code-block:: bash

    pip install pyrocko

    # and, (only) if you want to use Snuffler:

    pip install --only-binary :all: PyQt5 PyQtWebEngine

**Advantages:**

- Quick and easy.

**Disadvantages:**

- Dependencies installed by pip may shadow native system packages.
- May turn your system into a big mess.


Variant 2: use your system's package manager to install dependencies
--------------------------------------------------------------------

Install Pyrocko's requirements through your system's package manager (see
:doc:`/install/system/index`), then use pip with the ``--no-deps`` option to
install Pyrocko:

.. code-block:: bash

    # first use apt-get/yum/pacman to install prerequisites (see above), then:
    pip install --no-deps pyrocko

**Advantages:**

- Prevents package dependency conflicts.

**Disadvantages:**

- Need root access.
- A bit more work to set up.
