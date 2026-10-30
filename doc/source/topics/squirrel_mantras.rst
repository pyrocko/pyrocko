Squirrel processing pipelines (mantras)
=======================================

.. warning::

    Mantras and operators are experimental. Their interfaces and
    configuration formats may still change.

Operators derive new data from the data available through a
:py:class:`~pyrocko.squirrel.base.Squirrel`, e.g. by removing the instrument
response or by rotating the components of a sensor. Each operator takes one
or more inputs (a Squirrel or other operators) and can be queried like a
Squirrel itself, e.g. with ``get_waveforms()``, ``get_channels()`` or
``get_codes()``. Processing happens on demand, when data is requested.

A *mantra* is a named chain of operators, configured in a YAML file. Mantras
can be used from the command line, e.g. with :app:`squirrel mantra` or
:app:`squirrel service`, and from Python scripts.

Operators
---------

Some of the available operators:

* :py:class:`~pyrocko.squirrel.operators.base.Restitution` -- remove the
  instrument response, converting to displacement, velocity or acceleration.
* :py:class:`~pyrocko.squirrel.operators.base.ToENZ`,
  :py:class:`~pyrocko.squirrel.operators.base.ToTRZ`,
  :py:class:`~pyrocko.squirrel.operators.base.ToLQT` -- rotate the components
  of each sensor to East-North-Up, Transverse-Radial-Up or ray-based
  Longitudinal-Transverse-SH coordinates.
* :py:class:`~pyrocko.squirrel.operators.spectrogram.MultiSpectrogramOperator`
  -- compute multi-resolution spectrograms, see :app:`squirrel spectrogram`.

Derived data is identified by its codes. The basic operators above append a
short tag to the extra code (the fifth, non-standard, element of the codes),
e.g. ``Rv`` for response restitution to velocity, or ``Tenz`` for rotation to
ENZ. A vertical channel ``GR.GRA1..BHZ.`` processed with the mantra below
becomes ``GR.GRA1..BHZ.RvTenz``. How the output codes are formed can be
configured with codes templates, see :ref:`squirrel-mantras-codes-templates`.

.. note::

    Code patterns with four elements, like ``GR.GRA1..BHZ``, only match
    codes with an empty extra code. To select derived data, append ``.*``,
    e.g. ``--codes 'GR.GRA1..BHZ.*'``.

.. note::

    Mini-SEED cannot store the extra code, and it limits the length of the
    other codes to 2 (network), 5 (station), 2 (location) and 3 (channel)
    characters. Derived waveforms written to Mini-SEED lose their extra code,
    so they can no longer be told apart from the original data. Keep this in
    mind when writing templates which put information into the extra code,
    or which produce longer codes.

Configuration
-------------

Example mantra configuration file ``velocity.yaml``, restituting to velocity
and rotating to East-North-Up components:

.. code-block:: yaml

    --- !squirrel.Mantra
    name: velocity
    operators:
    - !squirrel.ops.Restitution
      quantity: velocity
      frequency_min: 0.1
      frequency_max: 5.0
    - !squirrel.ops.ToENZ {}

The operators are chained in the given order: the first operator gets its
input from the Squirrel, each following operator from the previous one.

.. _squirrel-mantras-codes-templates:

Codes templates
---------------

Each operator has a ``codes_projection`` attribute, which determines what
input it acts on, which inputs are processed together and how the output codes
are named. It contains a template string in Python's ``str.format`` syntax.
The default template of
:py:class:`~pyrocko.squirrel.operators.base.Restitution`, for example, is::

    {i.network}.{i.station}.{i.location}.{i.channel}.{i.extra}R{o.quantity}

Placeholders ``{i.*}`` are replaced by the elements of the input codes:
``network``, ``station``, ``location``, ``channel``, ``extra``, and
``channel_no_component``, the channel code without its last character.
Placeholders ``{o.*}`` are replaced by the attributes of the operator's
*outlets* (e.g. ``quantity``, or ``system`` and ``component`` for rotations),
and by the operator's ``name`` and the ``mantra`` name.

The template also defines which input codes are processed together (fan-in):
all input codes giving the same result, with the ``{o.*}`` placeholders left
empty, form one group. For the rotation operators, the default template uses
``{i.channel_no_component}``, so that the three components of a sensor form
one group. An operator with several outlets produces one output per outlet
(fan-out), e.g. the three output components of a rotation. To group
differently than the template suggests, an explicit ``group_by`` template can
be given. With ``include`` and ``exclude``, lists of codes patterns, the
operator can be restricted to part of its input.

``squirrel mantra show`` lists the resulting mappings, e.g. for the rotation
in the example above::

    GR.GRA1..{BHE,BHN,BHZ}.RvTenz <- GR.GRA1..{BHE,BHN,BHZ}.Rv

Example: restitute only vertical components and mark the output with the
mantra name instead of the default tag:

.. code-block:: yaml

    --- !squirrel.Mantra
    name: vel
    operators:
    - !squirrel.ops.Restitution
      quantity: velocity
      frequency_min: 0.1
      frequency_max: 5.0
      codes_projection: !squirrel.ops.CodesProjection
        template: '{i.network}.{i.station}.{i.location}.{i.channel}.{o.mantra}'
        include: ['*.*.*.*Z.*']

This gives mappings like ``GR.GRA1..BHZ.vel <- GR.GRA1..BHZ.``.

Example: compute spectrograms of all stations and average them per channel,
i.e. over all stations of an array:

.. code-block:: yaml

    --- !squirrel.Mantra
    name: specsum
    operators:
    - !squirrel.ops.MultiSpectrogramOperator {}
    - !squirrel.ops.CarpetSum
      normalize: true
      codes_projection: !squirrel.ops.CodesProjection
        template: '.SUM..{i.channel}.{i.extra}'

The template of the second operator does not contain the station, so all
stations with the same channel end up in one group::

    .SUM..BHZ.MS <- GR.{GRA1,GRA2,...,GRC4}..BHZ.MS

Note that derived codes become part of stored file names and of the codes
seen by later processing steps. Changing a template later therefore changes
how existing results are identified.

Command line
------------

To see which input codes are mapped to which output codes::

    squirrel mantra show --mantra velocity.yaml --add data/raw meta/stations.xml

The processed data can be viewed in the web browser with :app:`squirrel
service`, where it appears as a separate *gate*, named like the mantra::

    squirrel service --mantra velocity.yaml --add data/raw meta/stations.xml

Mantras which produce carpets (e.g. spectrograms) can be applied to a whole
dataset with :app:`squirrel mantra process`, storing the results in files.

Example: compute spectrograms of the vertical components of some stations for
one day, processing in batches of one hour:

.. code-block:: yaml

    --- !squirrel.Mantra
    name: spec
    operators:
    - !squirrel.ops.MultiSpectrogramOperator {}

::

    squirrel mantra process --mantra spec.yaml \
        --add data/raw meta/stations.xml \
        --tmin 2017-09-03 --tmax 2017-09-04 --tinc 3600 \
        --codes 'GR.GRA*..BHZ.*' \
        --out data/spectrograms/spec

The ``--codes`` patterns are matched against the output codes of the mantra,
e.g. ``GR.GRA1..BHZ.MS``, so the ``.*`` for the extra code is needed. When
stored, the extra element of the codes is replaced by the mantra name, e.g.
``GR.GRA1..BHZ.spec``. Use ``--append`` or ``--merge`` to add results to
existing files, e.g. when processing a dataset piece by piece. The results
can be viewed with ``squirrel service --add data/spectrograms/spec``.


Python
------

.. code-block:: python

    from pyrocko import squirrel, guts

    sq = squirrel.Squirrel()
    sq.add(['data/raw', 'meta/stations.xml'])

    mantra = guts.load(filename='velocity.yaml')
    mantra.setup(sq)

    tmin, tmax = sq.get_time_span(['waveform'])

    # Query processed data from the last operator of the chain.
    traces = mantra.outlet.get_waveforms(
        tmin=tmin, tmax=tmin+3600., codes='GR.GRA1.*.*.*')

Operators can also be created and connected directly in Python, without a
mantra:

.. code-block:: python

    rest = squirrel.Restitution(
        quantity='velocity', frequency_min=0.1, frequency_max=5.0)
    rest.add_input(sq)

    traces = rest.get_waveforms(tmin=tmin, tmax=tmin+3600.)

See :py:mod:`pyrocko.squirrel.operators.base` and
:py:mod:`pyrocko.squirrel.mantra` for the API reference.
