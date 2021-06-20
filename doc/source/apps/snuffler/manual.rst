
Snuffler manual
===============

Snuffler is a seismogram browser and workbench.

* read waveforms in Mini-SEED, SAC and other file formats
* visually browse through seismic waveform archives
* display real-time data streams
* snappy interface for zooming, panning, scrolling, filtering, rotating and scaling
* changes to controls give instantaneous visual feedback
* transparent handling of traces split into many files
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

.. describe:: hb628://<device>[?rate=<rate>[&channels=<channels>]]

   Acquire data from eight-channel USB-HB628 module. Example: to read channels
   0, 3 and 5 from serial device ``/dev/ttyS0`` at default rate (50Hz), use
   ``snuffler 'hb628:///dev/ttyS0?channels=035'``. By default all channels are
   shown.

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

    read station information from file ``STATIONS``, this option can be given
    more than once.  The format of the stations file is a simple five-column
    ASCII table where each line has the form

    ::

       <net>.<sta>.<loc>  <latitude>  <longitude>  <altitude>  <sensor-depth>

    ``<net>``, ``<sta>``, and ``<loc>`` are the network, station and location
    codes, respectively. ``<altitude>`` and ``<sensor-depth>`` are given in
    [m]. If network and location code are empty, use ``.<sta>.``, i.e. the dots
    should not be omitted.

.. option:: --stationxml=STATIONSXML

    read station information from XML file STATIONSXML

.. option:: --event=EVENT, --events=EVENT

    read event information from file ``EVENT``, this option can be given more than once

.. option:: --markers=MARKERS

    read marker information from file ``MARKERS``, this option can be given more than once
    
.. option:: --follow=N

    follow real time with a window of N seconds

.. option:: --cache=DIR 

    use directory DIR to cache trace metadata (default: ``pyrocko_0.3_cache_<username>`` in the system's default temporary directory)

.. option:: --force-cache

    use the cache even when trace attribute spoofing is active (may have silly consequences)

.. option:: --store-path=TARGET

    store data received through streams to TARGET. If TARGET is a directory, filnames are automatically choosen. If    more control over the filenames is needed, TARGET can be a filename template containing placeholders like ``%(KEY)s``, where KEY is any of ``network``, ``station``, ``location``, ``channel``, ``tmin`` (time of first sample), ``tmax`` (time of last sample).

.. option:: --store-interval=N

    dump stream data to file every N seconds [default: ``600``]

.. option:: --ntracks=N

    initially use ``N`` waveform tracks in viewer [default: ``24``]

.. option:: --opengl

    use OpenGL for drawing

.. option:: --qt5
    use Qt5 for the GUI
 
.. option:: --qt4
    use Qt4 for the GUI

.. option:: --debug

    print debugging information to stderr


Keystrokes
----------

=========================== ===============================================================
Key                         Effect
=========================== ===============================================================
:kbd:`q`                    Quit 
:kbd:`r`                    Reload modified files 
:kbd:`R`                    Reload snufflings
:kbd:`f`                    Toggle full screen mode 
:kbd:`m`                    Toggle marker sidebar
:kbd:`c`                    Toggle main controls
:kbd:`:`                    Enter command 
:kbd:`<space>`              Forward one page in time 
:kbd:`b`                    Backward one page in time 
:kbd:`<pagedown>`           Scroll tracks one page down 
:kbd:`<pageup>`             Scroll tracks one page up 
:kbd:`+`                    Show one track more 
:kbd:`-`                    Show one track less
:kbd:`=`                    Show initial number of tracks
:kbd:`g`                    Go to selection / show all
:kbd:`G`                    Zoom to selection / zoom to trace visibility
:kbd:`n`                    Go to next marker 
:kbd:`p`                    Go to previous marker 
:kbd:`N`                    Go to next event marker
:kbd:`P`                    Go to previous event marker 
:kbd:`<tab>`                Go to next marker of active event
:kbd:`<shift> + <tab>`      Go to previous marker of active event
:kbd:`a`                    Select all markers currently visible 
:kbd:`A`                    Select all markers 
:kbd:`d`                    Deselect all markers 
:kbd:`0` ... :kbd:`5`       Change color of marker 
:kbd:`<f1>` ... :kbd:`<f5>` Convert to phase marker
:kbd:`e`                    Convert to event marker / set active event / associate to event
:kbd:`<f10>`                Convert phase marker to normal marker
:kbd:`<backspace>`          Delete marker
:kbd:`<up>`, :kbd:`<down>`  Set first motion polarity on selected marker
:kbd:`<shift> + <up>`       Unset first motion polarity on selected marker
:kbd:`<escape>`             Abort picking 
:kbd:`?`                    Help
=========================== ===============================================================

Mouse
-----

================================ =========================================
Mouse                            Effect
================================ =========================================
Click and drag                   Zoom and pan 
Click and drag on time axis      Pan only 
Click on marker                  Select marker 
:kbd:`<shift>` + click on marker Select additional marker
Wheel                            Scroll tracks vertically 
:kbd:`<ctrl>` + wheel            Change number of tracks shown 
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

``<pattern>`` is matched against network, station, location, or channel ID of the traces depending on whether the ``n``, ``s``, ``l``, or ``c`` command is used, respectively. Only one quick-search pattern is active at any time. The currently active pattern is cleared by calling any of these commands without an argument.

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
* Assuming stations are named ``S01`` ... ``S15``, to remove all but ``S02`` and ``S04``, type ``hide s S0[!24]`` followed by ``hide s S1?``.

Unhide
^^^^^^

Unhide traces previously hidden with the ``hide`` command.

::

  unhide [ <pattern> ]
  unhide n|s|l|c <pattern>

The ``<pattern>`` argument must exactly correspond to a pattern previously given to the ``hide`` command. When ``unhide`` is called without any arguments, all currently active hide patterns are cleared.

Markers
^^^^^^^

Toggle marker visibility.

::

  markers [0][1][2][3][4][5]
  markers all
  markers

The visibility of the markers can be set selectively with regard to their kind
(color). Each number given in the argument to this command turns on visibility
of the corresponding marker kind, all other markers are hidden. If no arguments
are given, all markers are hidden. If the argument is ``all``, all markers are shown.

Scaling
^^^^^^^

Set scaling rules.

::

  scaling <vmin> <vmax>
  scaling <pattern> <vmin> <vmax>
  scaling

Traces are scaled according to the range [``<vmin>``, ``<vmax>``]. Either of
``<vmin>`` or ``<vmax>`` may be set to the string 'nan', to maintain automatic
scaling for the corresponding limit.  If three arguments are given, the first
argument should be a pattern, restricting application of the given scaling rule
to matching traces.  If no arguments are given, any previously set scalings
rules are cleared. 

Goto
^^^^

Jump to given time or event.

::

  goto YYYY-MM[-DD[ HH[:MM[:SS[.XXX]]]]]
  goto HH:MM[:SS[.XXX]]
  goto <eventname>

The first form causes the viewer to jump to the given date and time. With the
second form (when no date is given), the date is taken from the center of the
currently visible time range. Using the third form, it jumps to the time of an
event with the given ``<eventname>``. The event marker is neither selected nor
made active through this command.

