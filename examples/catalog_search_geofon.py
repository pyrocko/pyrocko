from pyrocko import util
from pyrocko.client import catalog

tmin = util.ctimegm('2010-01-12 21:50:00')
tmax = util.ctimegm('2010-01-13 03:17:00')  # ending time of query
mag = 6.                                    # minimum magntiude (open end)

# download event information from GEOFON web page

geofon = catalog.Geofon()
event_names = geofon.get_event_names(
    time_range=(tmin, tmax),
    magmin=mag)

for event_name in event_names:
    event = geofon.get_event(event_name)
    print(event)
