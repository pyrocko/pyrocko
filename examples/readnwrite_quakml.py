from pyrocko import model
from pyrocko.io import quakeml
from pyrocko.example import get_example_data


# small catalog containing two events (data from ingv):
catalog = get_example_data('example-catalog.xml')

# read quakeml events
qml = quakeml.QuakeML.load_xml(filename=catalog)

# get pyrocko events
events = qml.get_pyrocko_events()

for event in events:
    print(event)

# save events as pyrocko catalog:
model.event.dump_events(events, filename='test_pyrocko.pf')
