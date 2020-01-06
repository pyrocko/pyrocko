Introduction
============

`GMT <https://www.generic-mapping-tools.org/>`_ is a great plotting system,
which is capable of producing high quality EPS graphics. 

GMT version 4 and 5 are collections of command line tools well suited to be
called from shell scripts or other script languages - as in this case - Python.

The GmtPy modules aims to ease some aspects of calling GMT from Python.
Additionally, it provides some helpers for autoscaling and layout management.

This version of GmtPy is compatible with GMT version 4 and 5, **but not with
the new GMT version 6**. There are no plans to support GMT 6 from GmtPy as GMT
6 aims to provide a native interface to Python itself. However, lots of
existing scripts depend on GmtPy and GMT 4/5 and we will therefore try to
support this module for quite a while (writing this in 2019).

Features and design goals
-------------------------

* Thin and generic wrapper to GMT command execution 

    GMT commands, options and arguments are formed directly from method calls
    and method arguments by a clear and simple scheme. Although this attempt
    does not cure the ugly naming of GMT option arguments, this way it is
    transparent to the user and there is no need for any additional
    documentation to the GMT commands.

* Plain functional wrapping, no OO interface 

    Wrapping GMT into an object oriented interface would be appealing, but
    making it complete would require not only a lot of code, but also a vast
    amount of documentation. Rewriting GMT would be more straight forward, when
    this was the goal.

* Portability and consistency assistance 

    GmtPy uses its own consistent built-in set of GMT default parameters, such
    that running a GmtPy script on a different machine, or from a different
    user account should not change the appearance of the output graphics. But
    the script can of course override the GMT default parameters as needed.

* Encapsulation and parallel execution 

    GmtPy automatically maintains a temporary directory for each plot to be
    produced. When possible, GMT's 'isolation mode' is activated and tied to
    the plot's temporary directory. If used properly, this makes it painless to
    parallel execute several instances of a GmtPy script or to do GMT plotting
    in a multithreaded application.

* GMT version selection and backward compatibility 

    Parallel installations of different versions of GMT can be used. With GmtPy
    it is simple to select the GMT version to be used on a per plot basis.

* Autoscaling as opt-in 

    A highly configurable auto-scaler is provided. It features automatic range
    calculations, automatic nice tick increment determination, labeling
    assistance, scaled ax annotations and some convenience options such as
    snap-range-to-ticks, symmetric scaling and fixing the aspect ratio.

* Layout management as opt-in 

    An object oriented layout management system may be used to conveniently
    distribute subplots on a page. In its design, it is similar to the layout
    systems provided in various GUI toolkits. Different layout managers may be
    arbitrarily nested for further flexibility.

* Shortcut for fixing the bounding box in the output files 

* Shortcut for PDF output 

