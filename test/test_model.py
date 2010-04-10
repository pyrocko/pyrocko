from pyrocko import model, io, util
import unittest, math
import numpy as num

d2r = num.pi/180.

eps = 1e-15

def assertOrtho(a,b,c):
    xeps = max(abs(max(a)),abs(max(b)),abs(max(c)))*eps
    assert abs(num.dot(a,b)) < xeps
    assert abs(num.dot(a,c)) < xeps
    assert abs(num.dot(b,c)) < xeps

class ModelTestCase(unittest.TestCase):
    

    
    def testMissingComponents(self):
        
        ne = model.Channel('NE', azimuth=45., dip=0.)
        se = model.Channel('SE', azimuth=135., dip=0.)
        
        station = model.Station('', 'STA','', 0.,0., 0., channels=[ne,se])
        
        
        mat = station.projection_to_enu(('NE', 'SE', 'Z'), ('E', 'N', 'U'))[0]
        assertOrtho(mat[:,0],mat[:,1],mat[:,2])  
        
        n = model.Channel('D', azimuth=0., dip=90.)
        station.set_channels([n])
        mat = station.projection_to_enu(('N', 'E', 'D'), ('E', 'N', 'U'))[0]
        assertOrtho(mat[:,0],mat[:,1],mat[:,2])  

if __name__ == "__main__":
    util.setup_logging('test_trace', 'warning')
    unittest.main()
