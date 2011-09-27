
Snuffler
========

Snuffler is a seismogram browser.

* read waveforms in Mini-SEED, SAC and other file formats
* visually browse through seismic waveform archives
* display real-time data streams
* snappy interface for zooming, panning, scrolling, filtering, rotating and scaling
* changes to controls give instantaneous visual feedback
* transparent handling of traces splitted into many files
* integrated manual picker
* extendable with user plugins (snufflings)
* gap-aware continuous waveform processing framework


Invocation
----------

::

    snuffler [options] waveforms ...

Input ``waveforms`` can be any combination of the following data sources:

* waveform files of supported formats
* directories, which are recursively seached for waveform files
* pseudo URLs, which are used to open real-time data streams from different sources

The following pseudo URLs are supported:

.. describe:: seedlink://<host>[:<port>]/<pattern>
    
    Acquire data through SeedLink from given ``<host>``.  The specified ``<pattern>`` is matched against strings of the form ``<network>.<station>.<location>.<channel>``,   where the placeholders stand for the corresponding IDs of each SeedLink data stream. To use this feature, ``slinktool`` must be installed.

.. describe:: school://<device>

    Acquire data through serial line from "school seismometer".

.. describe:: usb628://<device>?<rate>

   Acquire data from a USB628 module.

.. describe:: cam://<device>

    Acquire data from Hamburg Univerisity Wiechert camera module.


Options
-------

.. program:: snuffler

.. option:: -h, --help

    show help message and exit 

.. option:: --format=FORMAT

    assume files are of given ``FORMAT`` [default: ``from_extension``]

.. option:: --pattern=REGEX

    only include files whose paths match ``REGEX``

.. option:: --stations=STATIONS

    read station information from file ``STATIONS``

.. option:: --event=EVENT

    read event information from file ``EVENT``

.. option:: --follow=N

    follow real time with a window of N seconds

.. option:: --progressive

    don't wait for file scanning to complete before opening the viewer

.. option:: --force-cache

    use the cache even when trace attribute spoofing is active (may have silly consequences)

.. option:: --ntracks=N

    initially use ``N`` waveform tracks in viewer [default: ``24``]

.. option:: --opengl

    use OpenGL for drawing

.. option:: --debug

    print debugging information to stderr


Keystrokes
----------

==================== ==================================
Key                  Effect
==================== ==================================
*q*                  Quit 
*r*                  Reload modified files 
*f*                  Toggle full screen mode 
*:*                  Enter command 
*<space>*            Forward one page in time 
*b*                  Backward one page in time 
*<pagedown>*         Scroll tracks one page down 
*<pageup>*           Scroll tracks one page up 
*+*                  Show one track more 
*-*                  Show one track less 
*n*                  Go to next marker 
*p*                  Go to previous marker 
*a*                  Select all markers in current view 
*d*                  Deselect all markers 
*<backspace>*        Delete selected markers 
*0* ... *5*          Change type of selected marker 
*<escape>*           Abort picking 
==================== ==================================

Mouse
-----

================================ =========================================
Mouse                            Effect
================================ =========================================
Click and drag                   Zoom and pan 
Click and drag on time axis      Pan only 
Click on marker                  Select marker 
Wheel                            Scroll tracks vertically 
*<ctrl>* + wheel                 Change number of tracks shown 
Right-click                      Menu 
Double-click                     Enter picking mode 
================================ =========================================

Commands
--------

After pressing '*:*' in the trace viewer, a command can be entered. To leave command mode press '*<return>*'.

Some of snuffler's commands take a ``<pattern>`` argument. These may contain the following shell-style wildcards:

============ ===================================
``*``        matches everything
``?``        matches any single character
``[seq]``    matches any character in seq
``[!seq]``   matches any character not in seq
============ ===================================

The pattern matching is done case-insensitive.

Quick-search traces
^^^^^^^^^^^^^^^^^^^

Reduce traces shown in viewer to those matching a given pattern.

``n|s|l|c [ <pattern> ]``

``<pattern>`` is matched against network, station, location, or channel ID of the traces depending on whether the ``n``, ``s``, ``l``, or ``c`` command is used, respectively. Here, implicitly a ``*``-wildcard is inserted after the pattern, so if for example pattern ``LH`` would be given, it would be evaluated as ``LH*``. Only one quick-search pattern is active at any time. The currently active pattern is cleared by calling any of these commands without an argument.

Hide
^^^^

Hide traces whose network, station, location, and channel IDs match a given pattern.

::

  hide <pattern>
  hide n|s|l|c <pattern>

Using the first form, ``<pattern>`` is matched against strings of the form ``<network>.<station>.<location>.<channel>``, where the placeholders stand for the corresponding IDs of each trace.

Using the second form, ``<pattern>`` is matched against network, station, location, or channel ID of the trace depending on whether ``n``, ``s``, ``l``, or ``c`` is given as first argument, respectively. For example, ``hide s <pattern>`` is short for ``hide *.<pattern>.*.*``.

The patterns given to successive invocations of ``hide`` are accumulated in a blacklist. To remove patterns from that blacklist, use the ``unhide`` command.

**Examples:**

* To hide any ``BHZ`` channels of stations with ID ``HAM3``, use ``hide *.ham3.*.bhz``.
* To hide all ``LHZ`` channels use ``hide c lhz``.
* To hide any ``LHE`` and ``LHN`` channels use ``hide c lh[en]``

Unhide
^^^^^^

Unhide traces previously hidden with the ``hide`` command.

::

  unhide [ <pattern> ]
  unhide n|s|l|c <pattern>

The ``<pattern>`` argument must exactly correspond to a pattern previously given to the ``hide`` command. When ``unhide`` is called without any arguments, all currently active hide patterns are cleared.


