Pyrocko is an open source seismology toolbox and library, written in the Python
programming language. It can be utilized flexibly for a variety of geophysical
tasks, like seismological data processing and analysis, modelling of InSAR, GPS
data and dynamic waveforms, or for seismic source characterization.

Installation with pip
---------------------

Using pip, Pyrocko can be installed from source or binary packages which we
have uploaded to the Python Package Index. Depending on your attitude,
different installation variants are possible (see following sections).
The complete `installation guide <https://pyrocko.org/docs/current/install>`_
is available in the `Pyrocko manual <https://pyrocko.org/docs/current/>`_.

*Good to Know:*

* Consequently use ``pip3`` instead of ``pip`` if you want to be sure that
  Python3 versions are installed
* Add the ``--user`` option to all pip commands if you want to install into
  your home directory.
* Consider using
  `virtual environments <https://docs.python.org/3/tutorial/venv.html>`_ when
  using pip to lower the risk of package conflicts.


Variant 1: allow pip to resolve dependencies
............................................

.. code-block:: bash

    pip install pyrocko

    # and, (only) if you want to use Snuffler:

    pip install --only-binary :all: PyQt5

**Advantages:**

- Quick and easy.

**Disadvantages:**

- Dependencies installed by pip may shadow native system packages.
- May turn your system into a big mess.


Variant 2: use your system's package manager to install dependencies
....................................................................

Install Pyrocko's requirements through your system's package manager (see
`System specific installation instructions <https://pyrocko.org/docs/current/install/system/>`_),
then use pip with the
``--no-deps`` option to install Pyrocko:

.. code-block:: bash

    # first use apt-get/yum/pacman to install prerequisites (see above), then:

    pip install --no-deps pyrocko

**Advantages:**

- Prevents package dependency conflicts.

**Disadvantages:**

- Need root access.
- A bit more work to set up.


Documentation
--------------

Documentation, examples and support at https://pyrocko.org/.


Development
------------

Join us at https://git.pyrocko.org/.


-- The Pyrocko Developers
