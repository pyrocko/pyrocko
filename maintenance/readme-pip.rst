Pyrocko is an open source seismology toolbox and library, written in the Python
programming language. It can be utilized flexibly for a variety of geophysical
tasks, like seismological data processing and analysis, modelling of InSAR, GPS
data and dynamic waveforms, or for seismic source characterization.

Installation
-------------

`Pyrocko is Python2/3 compatible.`

Only source packages for Pyrocko are available on ``pip``, this means that
parts of the code has is compiled locally.

Example for Ubuntu, Debian, Mint...

::

    # Install build requirements
    sudo apt-get install python3-dev python3-numpy
    sudo pip3 install pyrocko

    # Install requirements manually
    sudo pip3 install numpy>=1.8 scipy pyyaml matplotlib progressbar2 jinja2 requests PyOpenGL


For the GUI application ``PyQt4`` or ``PyQt5`` has to be installed:

::
    
    sudo apt-get install -y python3-pyqt5 python3-pyqt5.qtopengl python3-pyqt5.qtsvg
    

More information at https://pyrocko.org/docs/current/install

Documentation
--------------

Documentation, examples and support at https://pyrocko.org


Development
------------

Find us on GitHub - https://github.com/pyrocko


-- The Pyrocko Developers
