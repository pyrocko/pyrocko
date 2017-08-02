from guts_sensor_array import SensorArray


sa1 = SensorArray(distance_min=1e3, distance_max=100e3, strike=0.)
sa2 = SensorArray(distance_min=1e3, distance_max=100e3, strike=0.,
                  name='Sensor array 2')

print(sa1)
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

print(sa2)
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
print(sa3)
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

