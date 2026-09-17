Working with the ``guts`` package
=================================

What is Guts?
--------------

:py:mod:`pyrocko.guts` is a declarative data binding layer: you describe a
class' attributes using typed declarations, and get, for free, validation,
type coercion, and serialization to/from YAML and XML. Pyrocko uses it
pervasively for anything that needs to be saved to a file, read back, or
passed between tools - station and event metadata, source and receiver
models, Squirrel and Snuffler configuration, scenario descriptions, and
more.

Guts started out as a standalone project (its `original documentation
<https://github.com/emolch/guts>`_ still gives a decent, if now somewhat
outdated, overview). It has since been developed further as part of
Pyrocko and gained a few Pyrocko-specific extensions, most notably
:py:mod:`pyrocko.guts_array` for embedding :py:class:`numpy.ndarray` data.
This page is Pyrocko's own introduction to it.


Defining classes and composition
---------------------------------

A Guts class is defined by subclassing :py:class:`~pyrocko.guts.Object` and
declaring its attributes as class variables, using a type marker like
:py:class:`~pyrocko.guts.Float`, :py:class:`~pyrocko.guts.String`, or
:py:class:`~pyrocko.guts.List`, called through their ``.T()`` class method.
Objects can be nested to build up composite structures, just like nested
Python objects normally would be:

::

    from pyrocko.guts import Object, Float, String, List

    guts_prefix = 'demo'


    class Channel(Object):
        code = String.T(help='Channel code, e.g. ``HHZ``.')
        azimuth = Float.T(default=0.0, help='Azimuth [deg], 0 = North.')
        dip = Float.T(default=-90.0, help='Dip [deg], -90 = up.')


    class Station(Object):
        code = String.T(help='Station code, e.g. ``FUR``.')
        lat = Float.T(help='Latitude [deg].')
        lon = Float.T(help='Longitude [deg].')
        channels = List.T(Channel.T(), default=[])

Setting ``guts_prefix`` in the module is good practice: it prefixes the
YAML/XML type tags generated for classes defined in that module (e.g.
``!demo.Station`` below), which keeps tags unambiguous when combining
objects from different projects.


Creating objects
-----------------

Guts classes come with an automatically generated ``__init__`` that accepts
one keyword argument per declared attribute:

::

    station = Station(
        code='FUR',
        lat=48.163,
        lon=11.275,
        channels=[
            Channel(code='HHZ', azimuth=0., dip=-90.),
            Channel(code='HHN', azimuth=0., dip=0.),
            Channel(code='HHE', azimuth=90., dip=0.),
        ])

Attributes with a ``default`` are optional at construction time; attributes
without one are required and construction fails immediately with an
:py:exc:`~pyrocko.guts.ArgumentError` if they are missing:

::

    >>> Station(code='XX', lat=0.0)
    ArgumentError: Missing argument to demo.Station: lon

An attribute can instead be marked ``optional=True`` to allow it to be left
unset (``None``) without a default value being required.


Validation
----------

:py:meth:`~pyrocko.guts.Object.validate` checks that an object and all its
children have the exact types declared for them, raising
:py:exc:`~pyrocko.guts.ValidationError` otherwise:

::

    >>> station = Station(code='XX', lat='48.1', lon=11.0)  # lat as a string
    >>> station.validate()
    ValidationError: lat: "48.1" (type: <class 'str'>) is not of type float

This matters most for objects that were built up from loosely-typed data -
a parsed config file, user input, or the result of a YAML load done before
regularization (see below).


Regularization
---------------

:py:meth:`~pyrocko.guts.Object.regularize` is validation's more forgiving
sibling: it tries to convert child values to their declared types before
giving up, which is exactly what happens automatically whenever an object is
loaded from YAML or XML (where everything starts out as strings):

::

    >>> station.regularize()
    >>> station.lat
    48.1
    >>> type(station.lat)
    <class 'float'>

It still raises :py:exc:`~pyrocko.guts.ValidationError` if a value cannot be
converted:

::

    >>> station.lat = 'not-a-number'
    >>> station.regularize()
    ValidationError: lat: could not convert "not-a-number" to type float


Serialization
--------------

:py:meth:`~pyrocko.guts.Object.dump` serializes to YAML,
:py:meth:`~pyrocko.guts.Object.dump_xml` to XML:

::

    >>> print(station.dump())
    --- !demo.Station
    code: FUR
    lat: 48.163
    lon: 11.275
    channels:
    - !demo.Channel
      code: HHZ
      azimuth: 0.0
      dip: -90.0
    - !demo.Channel
      code: HHN
      azimuth: 0.0
      dip: 0.0
    - !demo.Channel
      code: HHE
      azimuth: 90.0
      dip: 0.0

    >>> print(station.dump_xml())
    <Station>
      <code>FUR</code>
      <lat>48.163</lat>
      <lon>11.275</lon>
      <channel>
        <code>HHZ</code>
        ...
      </channel>
      ...
    </Station>

Note how the plural attribute name ``channels`` becomes the singular XML
element name ``channel``, repeated once per list entry - Guts derives
sensible XML tag names automatically, though they can be overridden.

Both methods accept ``filename=...`` to write directly to a file instead of
returning a string.

``Object.__str__`` is defined as ``self.dump()`` by default, so ``print(x)``
or just ``x`` at a debugger prompt shows the same readable YAML dump without
having to call ``.dump()`` explicitly - a handy habit for debugging Guts
objects in general.

Serialization of attributes with defaults
-----------------------------------------

Attributes with a default value, when their assigned value equals the
default, are by default still included in the serialization output. To
suppress output of these, additionally set ``optional=True`` - this makes
Guts skip a value in the dump when (and only when) it equals the declared
default, which is a handy way to keep output minimal, showing only what was
actually customized:

::

    class Channel(Object):
        code = String.T()
        azimuth = Float.T(default=0.0)
        gain = Float.T(default=1.0, optional=True)

    >>> print(Channel(code='HHZ').dump())  # both left at their defaults
    --- !demo.Channel
    code: HHZ
    azimuth: 0.0

    >>> print(Channel(code='HHN', azimuth=90., gain=2.0).dump())
    --- !demo.Channel
    code: HHN
    azimuth: 90.0
    gain: 2.0

``azimuth`` is dumped either way since it is not marked ``optional``;
``gain`` is dumped only for the second channel, where it actually differs
from its default.

Deserialization
-----------------

:py:meth:`~pyrocko.guts.Object.load` and
:py:meth:`~pyrocko.guts.Object.load_xml` (also available as the module-level
functions ``load``/``load_xml``, which don't require knowing the class in
advance) read an object back, regularizing it in the process:

::

    from pyrocko.guts import load_string

    reloaded = load_string(station.dump())
    assert isinstance(reloaded, Station)
    assert reloaded.channels[0].code == 'HHZ'

``load``/``load_xml`` accept ``filename=...`` or ``stream=...`` the same way
``dump``/``dump_xml`` do.


Inheritance
-----------

Guts classes are regular Python classes and support normal inheritance. A
subclass inherits all attributes of its parent and can add its own:

::

    class NetworkStation(Station):
        network = String.T(default='XX')

    >>> NetworkStation(code='FUR', lat=48.1, lon=11.2, network='GR').dump()
    --- !demo.NetworkStation
    code: FUR
    lat: 48.1
    lon: 11.2
    network: GR

The same mechanism is used to build custom, self-validating types by
subclassing an existing one and overriding its inner ``__T`` descriptor
class - here, a latitude that rejects out-of-range values:

::

    from pyrocko.guts import TBase, ValidationError

    class Latitude(Float):
        class __T(TBase):
            def validate_extra(self, val):
                if not (-90. <= val <= 90.):
                    raise ValidationError('latitude must be in [-90, 90]')

    class Station(Object):
        code = String.T()
        lat = Latitude.T()
        lon = Float.T()

    >>> Station(code='XX', lat=120.0, lon=0.0).validate()
    ValidationError: latitude must be in [-90, 90]

The same ``__T`` class also has a ``regularize_extra`` hook, used the same
way to customize type coercion rather than validation.


Default value factories
-------------------------

A ``default=...`` value is normally written as a plain, already-typed
literal (``0.0``, ``[]``, ...). For types that need parsing/regularization
to reach their proper form - anything using
:py:class:`~pyrocko.guts.Timestamp`, for example, which accepts date
strings - a plain literal default is stored as-is, unregularized, and only
breaks later, when it's actually used:

::

    class Station(Object):
        code = String.T()
        since = Timestamp.T(default='1970-01-01 00:00:00')  # a plain string

    >>> Station(code='FUR').dump()
    TypeError: ufunc 'floor' not supported for the input types, ...

``<SomeType>.D(*args, **kwargs)`` builds a *default value factory* instead:
the given arguments are only used to actually construct (and regularize) a
fresh value the moment a default is needed, rather than once, eagerly, when
the class is defined:

::

    class Station(Object):
        code = String.T()
        since = Timestamp.T(default=Timestamp.D('1970-01-01 00:00:00'))

    >>> Station(code='FUR').since
    0.0
    >>> print(Station(code='FUR').dump())
    --- !demo.Station
    code: FUR
    since: '1970-01-01 00:00:00'

The same ``.D(...)`` mechanism works for any Guts type, not just
:py:class:`~pyrocko.guts.Timestamp` - for a default that is itself a
composite :py:class:`~pyrocko.guts.Object`, e.g. ``origin =
Origin.T(default=Origin.D(lat=0., lon=0.))``, it additionally means the
default doesn't have to be fully constructed by hand up front.


Computed properties
--------------------

Ordinary Python ``@property`` works as expected on Guts objects: it is not
a declared, typed attribute, so it is computed on access and, normally,
never appears in ``dump()`` output. It can be made to appear there too, by
additionally declaring a same-named attribute with a trailing ``__`` (which
Guts strips off when registering it, so it still serializes under the plain
name) and giving the property a setter that simply ignores the value it is
handed:

::

    class Station(Object):
        code = String.T()
        channels = List.T(Channel.T(), default=[])

        channel_codes__ = List.T(String.T(), optional=True)

        @property
        def channel_codes(self):
            return [channel.code for channel in self.channels]

        @channel_codes.setter
        def channel_codes(self, value):
            pass  # ignored - always recomputed from `channels` on access

    >>> print(station.dump())
    --- !demo.Station
    code: FUR
    channels:
    - !demo.Channel
      code: HHZ
    ...
    channel_codes:
    - HHZ
    ...

Loading such a value back in silently discards it the same way the setter does,
so this is effectively write-only.

Customized serialization
--------------------------

Sometimes an object is more naturally represented as a single scalar value
than as a mapping of named attributes. Subclassing
:py:class:`~pyrocko.guts.SObject` instead of
:py:class:`~pyrocko.guts.Object` and implementing ``__str__`` (for dumping)
and an ``__init__`` accepting that same string (for loading) serializes the
object as a plain string:

::

    from pyrocko.guts import SObject

    class NSLC(SObject):
        '''Network.Station.Location.Channel code, as one string.'''

        def __init__(self, s):
            self.network, self.station, self.location, self.channel = \\
                s.split('.')
            SObject.__init__(self, init_props=False)

        def __str__(self):
            return '.'.join(
                (self.network, self.station, self.location, self.channel))

    class Pick(Object):
        codes = NSLC.T()
        phase = String.T()

    >>> print(Pick(codes=NSLC('GR.FUR..HHZ'), phase='P').dump())
    --- !demo.Pick
    codes: GR.FUR..HHZ
    phase: P

This is exactly the pattern behind Pyrocko's own NSLC-style code classes,
:py:class:`~pyrocko.model.codes.Codes` and friends - and behind
:py:class:`~pyrocko.color.Color`, which serializes as a hex or named color
string rather than four separate component fields.


Numerical data with ``guts_array``
-------------------------------------

Plain Guts has no native concept of a :py:class:`numpy.ndarray`.
:py:mod:`pyrocko.guts_array` adds one, :py:class:`~pyrocko.guts_array.Array`,
so numeric data can be a regular, typed, (de)serializable attribute like any
other:

::

    import numpy as num
    from pyrocko.guts_array import Array

    class Channel(Object):
        code = String.T()
        gain_curve = Array.T(shape=(None,), dtype=float, serialize_as='list')

    ch = Channel(code='HHZ', gain_curve=num.array([1.0, 2.0, 3.5]))

    >>> print(ch.dump())
    --- !demo.Channel
    code: HHZ
    gain_curve:
    - 1.0
    - 2.0
    - 3.5

``shape`` (use ``None`` for an unconstrained dimension) and ``dtype`` are
validated like any other attribute. ``serialize_as`` controls the on-disk
representation: ``'list'`` (as above, readable but verbose for large arrays
- used this way for example in
:py:class:`~pyrocko.response.SampledResponse`), ``'table'``
(whitespace-separated rows, good for 2D data - see
:py:class:`~pyrocko.plot.automap.FloatTile`), or ``'base64'`` (compact
binary, for large arrays where readability doesn't matter).

Cloning objects
---------------

One more useful helper: :py:func:`~pyrocko.guts.clone` deep-copies a Guts
object tree based only on its declared Guts properties (falling back to
:py:func:`copy.deepcopy` for any plain, non-Guts values it contains) -
unlike :py:func:`copy.deepcopy` applied directly, it won't drag along
unrelated run-time state an object might be carrying.


Reactivity
-----------

Guts objects are otherwise passive: setting an attribute doesn't notify
anyone. :py:mod:`pyrocko.gui.talkie` builds a reactive layer on top of Guts
by subclassing :py:class:`~pyrocko.guts.Object` as
:py:class:`~pyrocko.gui.talkie.Talkie`, overriding attribute assignment to
fire a change notification - identified by its attribute path - every time
a property is set, with nested ``Talkie`` trees propagating notifications up
to a :py:class:`~pyrocko.gui.talkie.TalkieRoot`, where listeners subscribe
to a path (or any of its parent paths) with ``talkie_connect(...)``. A
``@computed(depends_on=[...])`` decorator adds derived properties that fire
their own notification whenever a property they depend on changes, and
``diff``/``diff_update`` compute and apply differences between two Talkie
trees.

:py:mod:`pyrocko.gui.state` builds two-way Qt widget bindings on top of
this (``state_bind_slider`` and friends: a widget updates the state on user
interaction, and reflects the state's value whenever it changes elsewhere).
This is the mechanism behind all of Sparrow's GUI state handling.


How Pyrocko makes use of Guts
-------------------------------

Nearly every persistent configuration or metadata object in Pyrocko is a
Guts object, built from exactly the pieces introduced above:

- Station, event, and moment tensor metadata in :py:mod:`pyrocko.model`.
- Source, target, and source-time-function classes in
  :py:mod:`pyrocko.gf.seismosizer` (e.g.
  :py:class:`~pyrocko.gf.seismosizer.DCSource`,
  :py:class:`~pyrocko.gf.targets.Target`) - the same composition and
  inheritance patterns as this page's ``Station``/``Channel`` example, at
  the scale of a full forward-modeling toolbox.
- Squirrel's data source declarations, e.g.
  :py:class:`~pyrocko.squirrel.client.fdsn.FDSNSource` and
  :py:class:`~pyrocko.squirrel.client.catalog.CatalogSource` (used by
  :py:meth:`~pyrocko.squirrel.base.Squirrel.add_fdsn` and
  :py:meth:`~pyrocko.squirrel.base.Squirrel.add_catalog` respectively), and
  on-disk dataset description files (see :py:mod:`pyrocko.squirrel.dataset`).
- Scenario descriptions in :py:mod:`pyrocko.scenario`.
- The StationXML and QuakeML readers/writers in :py:mod:`pyrocko.io.stationxml`
  and :py:mod:`pyrocko.io.quakeml` are themselves Guts class hierarchies
  mirroring those XML schemas, using ``dump_xml``/``load_xml`` directly.
- Persistent GUI state in Sparrow (Snuffler unfortunately does not use this
  approach).
- Command line tool configuration files, e.g. ``squirrel jackseis
  --config`` (see :py:mod:`pyrocko.squirrel.tool.commands.jackseis`) and
  ``fomosto report`` (see
  :py:class:`~pyrocko.fomosto.report.report_main.GreensFunctionTest`).

This extends beyond Pyrocko itself: downstream projects such as Grond (a
source inversion framework built on Pyrocko) make heavy use of Guts too,
typically by subclassing and nesting Pyrocko's own Guts classes into their
own configuration hierarchies rather than reinventing equivalent
serializable types from scratch.

If you are adding a new tool or file format to Pyrocko and it needs to save
or load structured data, reach for Guts before reaching for
:py:mod:`json`/:py:mod:`pickle` or hand-rolled YAML - it buys validation and
a consistent on-disk format for free.
