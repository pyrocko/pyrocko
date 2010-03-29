from pyrocko import model, io, util
import unittest, math
import numpy as num

d2r = num.pi/180.

class ModelTestCase(unittest.TestCase):
    
    def testMissingComponents(self):
        
        ne = model.Channel('NE', azimuth=45., dip=0.)
        se = model.Channel('SE', azimuth=135., dip=0.)
        
        station = model.Station('', 'STA','', 0.,0., 0., channels=[ne,se])
        
        print station.projection_to_enu(('NE', 'SE', 'Z'), ('E', 'N', 'U'))
        
        
        n = model.Channel('D', azimuth=0., dip=90.)
        station.set_channels([n])
        print station.projection_to_enu(('N', 'E', 'D'), ('E', 'N', 'U'))

if __name__ == "__main__":
    util.setup_logging('test_trace', 'warning')
    unittest.main()
