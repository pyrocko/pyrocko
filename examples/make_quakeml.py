from pyrocko.io import quakeml as q
from pyrocko.util import str_to_time as stt


table = [
    # name     time                        lat  lon   depth    mag
    ('ev0001', stt('2021-09-29 01:02:03'), 55., 120., 11000.,  5.5),
    ('ev0002', stt('2021-09-29 04:05:06'), 57., 122., 9000.,  4.0),
    ('ev0003', stt('2021-09-29 07:08:09'), 56., 121., 1200.,  4.1)]


qml = q.QuakeML(
    event_parameters=q.EventParameters(
        public_id='quakeml:test/eventParameters/test',
        event_list=[
            q.Event(
                preferred_origin_id='quakeml:test/origin/%s' % name,
                preferred_magnitude_id='quakeml:test/magnitude/%s' % name,
                public_id='quakeml:test/event/%s' % name,
                origin_list=[
                    q.Origin(
                        public_id='quakeml:test/origin/%s' % name,
                        time=q.TimeQuantity(value=time),
                        longitude=q.RealQuantity(value=lon),
                        latitude=q.RealQuantity(value=lat),
                        depth=q.RealQuantity(value=depth),
                    ),
                ],
                magnitude_list=[
                    q.Magnitude(
                        public_id='quakeml:test/magnitude/%s' % name,
                        origin_id='quakeml:test/origin/%s' % name,
                        mag=q.RealQuantity(value=magnitude),
                    )
                ],
            )
            for (name, time, lat, lon, depth, magnitude) in table
        ]
    )
)

print(qml.dump_xml())
