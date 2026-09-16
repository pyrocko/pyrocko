from pyrocko import util, model
from pyrocko.squirrel import Squirrel

tmin = util.str_to_time('2011-01-01 00:00:00')  # beginning time of query
tmax = util.str_to_time('2011-12-31 23:59:59')

sq = Squirrel()

# Add the GlobalCMT catalog as an online data source. `query_args` are
# common arguments appended to every query made to it - here used to
# restrict results to northern Chile and a minimum magnitude. Values must
# be given as strings.
sq.add_catalog('gcmt', query_args=dict(
    magmin='2.', latmin='-35.', latmax='-20.', lonmin='-76.', lonmax='-65.'))

# Refresh the local copy of the catalog for the time span of interest. This
# is when the actual query to the remote data source happens, if needed.
sq.update(tmin=tmin, tmax=tmax)

events = sq.get_events(tmin=tmin, tmax=tmax)

print('Downloaded %s events' % len(events))
print('The last one is')
print(events[-1])

# dump events to catalog
model.dump_events(events, 'northern_chile_events.txt')
