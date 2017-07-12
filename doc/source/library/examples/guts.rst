Working with the ``guts`` package
=================================

Introduction to ``guts``
-------------------------------

Guts is a `Lightweight declarative YAML and XML data binding for Python 
<https://github.com/emolch/guts>`_ (the link provides a basic introduction
and usage tutorial).


``guts`` and ``pyrocko``
------------------------

When building add-on functionality to pyrocko, objects that will be 
in/ex-ported can use the guts wrapping to conform easily to prycoko standards.  
There is some overhead, so only certain items implement the guts package in 
the base pyrocko project.  One add-on functionality that implement it is the 
``fomosto report`` command.  Within it, the base class 
:py:class:`pyrocko.fomosto_report.GeensFunctionTest` 
and a small utility class :py:class:`pyrocko.fomosto_report.SensoryArray` are 
implemented with guts.  We shall look at 
:py:class:`pyrocko.fomosto_report.SensoryArray` below.

Implementing ``guts``
---------------------

When creating a class that implements :py:mod:`pyrocko.guts`, first determine 
which attribute are important to the class.  In the case of 
:py:class:`pyrocko.fomosto_report.SensoryArray` those attributes are: 
``distance_min``, ``distance_max``, ``strike``, and ``sensor_count``.  Now 
create the class and attributes, where the attributes are defined as guts 
types (good practice is to define a ``guts_prefix`` to the module, as it helps 
with in/ex-porting).

::

    from pyrocko.guts import load, Object, Float, Int, String
    from pyrocko.gf import Target

    guts_prefix = 'gft'

    class SensorArray(Target):

        distance_min = Float.T()
        distance_max = Float.T()
        strike = Float.T()
        sensor_count = Int.T(default=50)

        # this attribute is only used in this example
        name = String.T(optional=True)

        def __init__(self, **kwargs):

            # call the guts initilizer
            Object.__init__(self, **kwargs)

As you can see only one type has a default value, so when initializing the 
object if values do not get passed in for the attributes without default 
values, then the ``guts`` initializer will throw an error.

Other attributes whose values are desired to be keep through in/ex-porting, 
but do not need to be initialized by default can have the ``optional`` 
parameter set to ``True``.  This will allow them to be exported with when the 
attribute has been assigned a value.

Usage examples
-----------------

Now that a class has been defined (and imported) we can see how to use it.

::

    sa1 = SensorArray(distance_min=1e3, distance_max=100e3, strike=0.)
    sa2 = SensorArray(distance_min=1e3, distance_max=100e3, strike=0.,
                      name='Sensor array 2')

    print sa1
    '''
    output would look like
    --- !gft.SensorArray
    # properies defined by the base type Target
    depth: 0.0
    codes: ['', STA, '', Z]
    elevation: 0.0
    interpolation: nearest_neighbor

    # attributes defined within the SensorArray class
    distance_min: 1000.0
    distance_max: 100000.0
    strike: 0.0
    sensor_count: 50
    '''

    print sa2
    '''
    output would look like
    --- !gft.SensorArray
    # properies defined by the base type Target
    depth: 0.0
    codes: ['', STA, '', Z]
    elevation: 0.0
    interpolation: nearest_neighbor

    # attributes defined within the SensorArray class
    distance_min: 1000.0
    distance_max: 100000.0
    strike: 0.0
    sensor_count: 50
    name: Sensor array 2
    '''

    # export the object definition to a file
    sa1.dump(filename='sensorarray1')

    # import object definition from file
    sa3 = load('sensorarray1')
    sa3.name = 'Sensor array 3'
    print sa3
    '''
    output would look like
    --- !gft.SensorArray
    # properies defined by the base type Target
    depth: 0.0
    codes: ['', STA, '', Z]
    elevation: 0.0
    interpolation: nearest_neighbor

    # attributes defined within the SensorArray class
    distance_min: 1000.0
    distance_max: 100000.0
    strike: 0.0
    sensor_count: 50
    name: Sensory array 3
    '''

