Pyrocko-GF - Geophysical forward modelling with pre-calculated Green’s functions
================================================================================

Introduction
------------

Many seismological methods require knowledge of Green’s functions in
dependance of ranges of source and receiver coordinates. Examples range
from synthetic seismogram calculation over source imaging techniques to
source inversion methods. Calculation of Green’s functions is a
computationally expensive operation and it can be of advantage to
calculate them in advance. The same Green’s function traces can then be
reused several or many times as required in a typical application.

Regarding Green’s function creation as an independent step in a
use-case’s processing chain encourages the storage of these in an
application independant form. They can then immediatly be reused when
new data is to be processed, they can be shared between different
applications and they can also be passed to other researchers, allowing
them to focus on their own application rather then spending days of work
to get their Green’s function setup ready.

Furthermore, it is useful to store associated meta information, like
e.g. travel time tables for seismic phases and the earth model used,
together with the Green’s function in order to have a complete and
consistent framework to play with.

.. figure :: /static/software_architecture.svg
    :align: center
    :alt: Overview of the components and interaction for pyrocko-gf


Pyrocko contains a flexible framework to store and work with
pre-calculated Green’s functions. It is implemented in the pyrocko.gf
subpackage. Also included, is a powerful front end tool to create,
inspect, and manipulate Green’s function stores: the fomosto tool
(“forward model storage tool”).

Media models
------------

Pyrocko uses layered 1-d earth models in the calculation of Green’s
functions. The format of the earth model is given in the
`TauP <https://www.seis.sc.edu/downloads/TauP/taup.pdf>`__ style ``.nd``
(named discontinuity) with the following columns:

::

   depth [km]    Vp[km/s]    Vs[km/s]    density[g/cm^3]    qp    qs

The
`Fomosto <https://pyrocko.org/docs/current/apps/fomosto/tutorial.html#configuration>`__
application description holds an exemplary earth model configuration.
Users can define own input models before calculation. Also a number of
predefined common variations of AK135 and PREM models are available.
They can be listed and inspected using the
`cake <https://pyrocko.org/docs/current/apps/cake/manual.html>`__
command line tool.

Green’s function stores
-----------------------

GF Pre-calculation
~~~~~~~~~~~~~~~~~~

Calculating and managing Pyrocko-GF stores is accomplished by Pyrocko’s
``fomosto`` application. More details how GF stores are set-up and
calculated can be found in the `Fomosto
Introduction <https://pyrocko.org/docs/current/apps/fomosto/tutorial.html>`__.

Download GF stores
~~~~~~~~~~~~~~~~~~

Calculating and quality checking Green’s function stores is a time
consuming task. Many pre-calculated stores can be downloaded from our
online repository

   https://greens-mill.pyrocko.org

The available stores include dynamic stores for simulating wave forms at
global and regional extents, as well as static stores for modelling
surface displacements.

Download a store with ``fomosto``:

.. code:: sh

   fomosto download kinherd global_2s_25km 

Using GF Stores
~~~~~~~~~~~~~~~

GF stores are accessed for forward modelling by the
``pyrocko.gf.Engine``. Here is how we can start-up the engine for
modelling:

.. code:: python

   from pyrocko.gf import LocalEngine

   engine = LocalEngine(store_dirs=['gf_stores/global_2s/'])

A complete list of arguments can be found in the library reference,
``pyrocko.gf.seismosizer.LocalEngine``.

Source models
-------------

Pyrocko-GF supports the simulation of various dislocation sources,
focused on earthquake and volcanoe studies.

Point sources
~~~~~~~~~~~~~

+-----------------------------------+-----------------------------------+
| Source                            | Short description                 |
+===================================+===================================+
| ``ExplosionSource``               | An isotrope moment tensor for     |
|                                   | explosions or volume changes.     |
+-----------------------------------+-----------------------------------+
| ``DCSource``                      | Double force couple, for          |
|                                   | pure-shear earthquake ruptures.   |
+-----------------------------------+-----------------------------------+
| ``MTSource``                      | Full moment tensor representation |
|                                   | of force excitation.              |
+-----------------------------------+-----------------------------------+
| ``CLVDSource``                    | A pure compensated linear vector  |
|                                   | dipole source.                    |
+-----------------------------------+-----------------------------------+
| ``VLVDSource``                    | Volumetric linear vector dipole,  |
|                                   | a rotational symmetric volume     |
|                                   | source.                           |
+-----------------------------------+-----------------------------------+

Finite sources
~~~~~~~~~~~~~~

+-----------------------------------+-----------------------------------+
| Source                            | Short description                 |
+===================================+===================================+
| ``RectangularSource``             | Rectangular fault plane.          |
+-----------------------------------+-----------------------------------+
| ``RingFault``                     | Ring fault for volcanic           |
|                                   | processes, e.g. caldera           |
|                                   | collapses.                        |
+-----------------------------------+-----------------------------------+

Multiple sources can be combined through the ``pyrocko.gf.CombiSource``
object.

Explosion source
~~~~~~~~~~~~~~~~

An isotropic explosion point source which can also be used for
volumetric dislocations.

.. code:: python

   # Initialise a simple ExplosionSource
   explosion = ExplosionSource(lat=42., lon=22., depth=8e3, volume_change=5e8)

Double couple
~~~~~~~~~~~~~

A double-couple point source, describing describing simple shear
ruptures.

.. code:: python

   # Initialise a DCSource
   dcsource = DCSource(lat=54., lon=7., depth=5e3, strike=33., dip=20., rake=80.)

Moment tensor
~~~~~~~~~~~~~

A moment tensor point source. This is the most complete form of
describing an ensemble of forces.

.. code:: python

   # Initialise a moment tensor
   mtsource = MTSource(
       lat=20., lon=58., depth=8.3e3,
       mnn=.5, mee=.1, mdd=.7,
       mne=.6, mnd=.2, med=.1,
       magnitude=6.3)
                       
   mtsource = MTSource.from_pyrocko_event(event)

CLVD source
~~~~~~~~~~~

A pure compensated linear vector dipole (CLVD) point source.

.. code:: python

   # Initialise a CLVD source
   clvdsource = CLVDSource(lat=48., lon=17., depth=5e3, dip=31.depth=5e3, , azimuth=83.)

VLVD source
~~~~~~~~~~~

A Volumetric Linear Vector Dipole, a uniaxial rotational symmetric
volume source. This source can be used to constrain sill or dyke like
volume dislocation sources.

.. code:: python

   # Initialise a VLVD source
   vlvdsource = VLVDSource(
       lat=-30., lon=184., depth=5e3, 
       volume_change=1e9, clvd_moment=20e9, dip=10., azimuth=110.)

Rectangular fault
~~~~~~~~~~~~~~~~~

Classical Haskell finite source model modified for bilateral rupture.

.. code:: python

   rectangular_fault = RectangularFault(
       lat=20., lon=44., depth=5e3,
       dip=30., strike=120., rake=50.,
       width=3e3, length=8e3, slip=2.3)

Ring fault
~~~~~~~~~~

A ring fault with vertical double couples. Ring faults can describe
volcanic processes, e.g. caldera collapses.

.. code:: python

   ring_fault = RingFault(
       lat=31., lon=12., depth=2e3,
       diameter=5e3, sign=1.,
       dip=10., strike=30.,
       npointsources=50)

Source Time Functions
---------------------

Source time functions describe the energy radiation of a dislocation
source in time. A number of Source Time Functions (STF) are available
and can be applied in pre- or post-processing. If no specific STF is
defined as a unit pulse response.

+-----------------------------------+-----------------------------------+
| STF                               | Short description                 |
+===================================+===================================+
| ``BoxcarSTF``                     | Boxcar type source time function. |
+-----------------------------------+-----------------------------------+
| ``TriangularSTF``                 | Triangular type source time       |
|                                   | function.                         |
+-----------------------------------+-----------------------------------+
| ``HalfSinusoidSTF``               | Half sinusoid type source time    |
|                                   | function.                         |
+-----------------------------------+-----------------------------------+
| ``SmoothRampSTF``                 | A smooth-ramp type source time    |
|                                   | function for near-field           |
|                                   | displacements.                    |
+-----------------------------------+-----------------------------------+
| ``ResonatorSTF``                  | A simple resonator like source    |
|                                   | time function.                    |
+-----------------------------------+-----------------------------------+

Boxcar STF
~~~~~~~~~~

A classical Boxcar source time function.

.. code:: python

   # Initalize an Boxcar STF function with duration of 5s and centered around the centroid time.
   stf = BoxcarSTF(5., center=0.)

Triangular STF
~~~~~~~~~~~~~~

.. code:: python

   # Initalize an Triangular STF function with duration 5s, which reaches its maximum amplitude after half the duration and centered around the centroid time.
   stf = TriangularSTF(5., peak_ratio=0.5, center=0.)

Half sinusoid STF
~~~~~~~~~~~~~~~~~

.. code:: python

   # Initalize an Half sinusoid type STF function with duration of 5s and centered around the centroid time.
   stf = HalfSinusoidSTF(5., center=0.)

Smooth ramp STF
~~~~~~~~~~~~~~~

.. code:: python

   # Initalize an Smooth ramp type STF function with duration of 5s, which reaches its maximum amplitude after half the duration and centered around the centroid time.
   stf = SmoothRampSTF(5., rise_ratio=0.5, center=0.)

Resonator STF
~~~~~~~~~~~~~

.. code:: python

   # Initalize an Resonator STF function with duration of 5s and a resonance frequency of 1 Hz. 
   stf = SmoothRampSTF(5., frequency=1.0)

Modelling targets
-----------------

Targets are generic data representations, derived or postprocessed from
observables or synthesised data. A
`Target <https://pyrocko.org/docs/current/library/reference/gf.html#module-pyrocko.gf.targets>`__
can be, a filtered waveform, a spectrum or InSAR displacement. Each
target has properties and essentially is associated to a Green’s
functions store, which will model the synthetics for a particular
target. The target also defines the interpolation used for the discrete,
gridded Green’s fucntion components. Please also see the
`Examples <https://pyrocko.org/docs/current/library/examples/gf_forward.html>`__.

Note: In Pyrocko locations are given with five coordinates:
``latitude``, ``longitude``, ``east_shift``, ``north_shift`` and
``depth``. Latitude and longitude are the origin of an optional local
cartesian coordinate system for which an ``east_shift`` and a
``north_shift`` [m] can be defined. The horizontal location is the
combination of latitude and longitude to which ``north_shift`` and
``east_shift``, respectively, are added. A target has a depth below the
surface. However, the surface can have topography and the target can
have an assigned elevation.

Waveforms
~~~~~~~~~

Waveforms are the most classical target and are therefore called
``pyrocko.gf.Target`` (see also
`reference <https://pyrocko.org/docs/current/library/reference/gf.html#module-pyrocko.gf.targets>`__.
They have a single location (e.g. the station), define a certain
orientation (e.g. vertical or radial), a time, and a time dependency.

.. code:: python

   # Define a list of pyrocko.gf.Target objects, representing the recording
   # devices. In this case one station with a three component sensor 
   channel_codes = 'ENU'
   targets = [
       Target(
           lat=10., lon=10.,
           store_id='global_2s_25km',
           codes=('NET', 'STA', 'LOC', channel_code))
       for channel_code in channel_codes]

See the `Example
Target <https://pyrocko.org/docs/current/library/examples/gf_forward.html#calculate-synthetic-seismograms-from-a-local-gf-store>`__
for instructions of usage.

Surface displacements
~~~~~~~~~~~~~~~~~~~~~

Surface displacements are modelled as ``pyrocko.gf.StaticTarget``, they
have no time evolution, but can hold many locations. Special forms of
the ``pyrocko.gf.StaticTarget`` and derived from it are the
``pyrocko.gf.SatelliteTarget``, which is used for the forward modelling
of InSAR data, and the ``pyrocko.gf.GNSSCampaignTarget`` (e.g. GPS
displacements).

.. code:: python

   # Initialising a StaticTarget
   # east and north are NumPy arrays in meters
   import numpy as num

   km = 1e3
   norths = num.linspace(-20*km, 20*km, 100)
   easts = num.linspace(-20*km, 20*km, 100)
   north_shifts, east_shifts = num.meshgrid(norths, easts)

   static_target = StaticTarget(
       lats=43., lons=20.,
       north_shifts=north_shifts,
       east_shifts=east_shifts,
       tsnapshot=24. * 3600.,  # one day
       interpolation='nearest_neighbor',
       store_id='ak135_static')

The ``pyrocko.gf.SatelliteTarget`` defines the locations of displacement
measurements and the direction of the measurement, which is the
so-called line-of-sight of the radar. See the `Example
SatelliteTarget <https://pyrocko.org/docs/current/library/examples/gf_forward.html#calculate-spatial-surface-displacement-from-a-local-gf-store>`__
for detailed instructions of usage.

.. code:: python

   # Initialising a SatelliteTarget.
   # east/north shifts are NumPy in [m]
   # line-of-sight angles are NumPy arrays,
   # - phi is _towards_ the satellite clockwise from east in [rad]
   # - theta is the elevation angle from the horizon

   satellite_target = gf.SatelliteTarget(
       lats=43., lons=20.,
       north_shifts=north_shifts,
       east_shifts=east_shifts,
       tsnapshot=24. * 3600.,  # one day
       interpolation='nearest_neighbor',
       phi=phi,
       theta=theta,
       store_id='ak135_static')

The ``pyrocko.gf.GNSSCampaignTarget`` defines station locations and the
three components east, north and up.

Forward modelling with Pyrocko-GF
---------------------------------

Forward modelling based on a defined source model and for a defined
target is handled in the so-called ``pyrocko.gf.engine``. The engine
initialisation requires the setting of the folder, where the Green’s
function stores are. It is possible to configure your
``store_superdirs`` in the ``~/.pyrocko/config.pf``. Note, that
modelling of dynamic targets requires GFs that have many times samples
and modelling of static targets have usually only one. It is therefore
meaningful to use dynamic GF stores for dynamic targets and effcient to
use static GF stores for static targets.

.. code:: python

   # (...)
   # Setup the LocalEngine and point it to the GF store you want to use.
   # *store_superdirs* is a list of directories where to look for GF Stores.
   engine = gf.LocalEngine(store_superdirs=['.'])
   # The computation is performed by calling process on the engine
   response = engine.process(sources=rect_source, targets=[satellite_target])

For waveform targets, traces can be derived directly from the response:

.. code:: python

   # convert results in response to traces
   synthetic_traces = response.pyrocko_traces()
   # to directly visualise the response, use snuffler
   synthetic_traces.snuffle()

For static targets, generally, the results are retrieved in the
following way:

.. code:: python

   # This will return an list of static results:
   synth_disp = response.static_results()

For regularly gridded satellite targets, specifically, the forward modelling of the engine's response can be directly converted to a synthetic `Kite <https://pyrocko.org/kite/docs/current/>`__ scene:

.. code:: python

   # This will produce kite scenes from the results
   kite_scenes = response.kite_scenes()
   # to look at the synthetics of the first targets of a list, use spool
   kite_scenes[0].spool()

The kite scene request can be complemented with an individual scene configuration to modify the output scenes meta data accordingly, e.g. changing the line-of-sight direction of the results from the response.

