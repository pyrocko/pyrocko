Installation on Windows
=======================

.. note::

   Windows support is experimental. We would love to get your feedback in the 
   `Windows channel on the Pyrocko Hive 
   <https://hive.pyrocko.org/pyrocko-support/channels/windows>`_.

We provide binary PIP and Anaconda packages for the Windows platform for Python
3.6 - 3.9. If MSVC is installed, it is also possible to install Pyrocko from
source.

From precompiled Anaconda packages
----------------------------------

If you are using Anaconda under Windows, see :doc:`/install/packages/anaconda`.

From precompiled PIP packages
-----------------------------

If you want to use PIP to install Pyrocko under Windows, see
:doc:`/install/packages/pip`.

.. _windows-install-from-source:

From source
-----------

In this example installation, we assume that you are using Anaconda/Miniconda
to manage your Python environment. Other setups may also be possible as long as
you can run ``git``, ``python``, ``pip``, and ``patch`` from your Windows
command line and you have a C compiler installed.

1. Install `Build Tools for Visual Studio 2019
   <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019>`_.
   You don't need the complete Visual Studio 2019. In Build tools, install C++
   build tools and ensure the latest versions of MSVCv142 - VS 2019 C++ x64/x86
   build tools and Windows 10 SDK are checked. 

2. On the Windows command line, run::

    git clone https://git.pyrocko.org/pyrocko/pyrocko.git
    cd pyrocko
    python install.py deps conda
    python install.py user

For instructions on how to install Pyrocko on other systems or if the
installation with the above procedure fails, see :doc:`index` or
:doc:`/install/details`.
