from pyrocko import orthodrome, model

e = model.Event(lat=10., lon=20.)
s = model.Station(lat=15., lon=120.)

# one possibility:
d = orthodrome.distance_accurate50m(e,s)
print 'Distance between e and s is %g km' % (d/1000.)

# another possibility:
s.set_event_relative_data(e)
print 'Distance between e and s is %g km' % (s.dist_m/1000.)


