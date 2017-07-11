from pyrocko import util, marker as pm

markers = pm.load_markers('my_markers.pf')
pm.associate_phases_to_events(markers)

for marker in markers:
    print util.time_to_str(marker.tmin), util.time_to_str(marker.tmax)

    # only event and phase markers have an event attached
    if isinstance(marker, (pm.EventMarker, pm.PhaseMarker)):
        ev = marker.get_event()
        print ev  # may be shared between markers
