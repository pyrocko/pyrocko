from pyrocko import util, model
from pyrocko.client import catalog

tmin = util.str_to_time('2011-01-01 00:00:00')  # beginning time of query
tmax = util.str_to_time('2011-12-31 23:59:59')

# create an instance of the global CMT catalog
global_cmt_catalog = catalog.GlobalCMT()

# query the catalog
events = global_cmt_catalog.get_events(
    time_range=(tmin, tmax),
    magmin=2.,
    latmin=-35.,
    latmax=-20.,
    lonmin=-76.,
    lonmax=-65.)

print('Downloaded %s events' % len(events))
print('The last one is:')
print(events[-1])

# dump events to catalog
model.dump_events(events, 'northern_chile_events.txt')
