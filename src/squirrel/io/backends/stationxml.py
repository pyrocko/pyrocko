# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

import logging
import time
from pyrocko.io.io_common import get_stats, touch  # noqa
from ... import model


logger = logging.getLogger('psq.io.stationxml')


Y = 60*60*24*365


def provided_formats():
    return ['stationxml']


def detect(first512):
    if first512.find(b'<FDSNStationXML') != -1:
        return 'stationxml'

    return None


def iload(format, file_path, segment, content):
    assert format == 'stationxml'

    far_future = time.time() + 20*Y

    from pyrocko.fdsn import station as fdsn_station
    value_or_none = fdsn_station.value_or_none

    sx = fdsn_station.load_xml(filename=file_path)

    inut = 0

    for network in sx.network_list:
        for station in network.station_list:
            net = network.code
            sta = station.code
            agn = ''

            tmin = station.start_date
            tmax = station.end_date
            if tmax is not None and tmax > far_future:
                tmax = None

            station_nut = model.make_station_nut(
                file_segment=0,
                file_element=inut,
                agency=agn,
                network=net,
                station=sta,
                location='*',
                tmin=tmin,
                tmax=tmax)

            if 'station' in content:
                station_nut.content = model.Station(
                    lat=station.latitude.value,
                    lon=station.longitude.value,
                    elevation=value_or_none(station.elevation),
                    **station_nut.station_kwargs)

            yield station_nut
            inut += 1

            for channel in station.channel_list:
                cha = channel.code
                loc = channel.location_code.strip()

                tmin = channel.start_date
                tmax = channel.end_date
                if tmax is not None and tmax > far_future:
                    tmax = None

                deltat = None
                if channel.sample_rate is not None \
                        and channel.sample_rate.value != 0.0:

                    deltat = 1.0 / channel.sample_rate.value

                nut = model.make_channel_nut(
                    file_segment=0,
                    file_element=inut,
                    agency=agn,
                    network=net,
                    station=sta,
                    location=loc,
                    channel=cha,
                    tmin=tmin,
                    tmax=tmax,
                    deltat=deltat)

                if 'channel' in content:
                    nut.content = model.Channel(
                        lat=channel.latitude.value,
                        lon=channel.longitude.value,
                        elevation=value_or_none(channel.elevation),
                        depth=value_or_none(channel.depth),
                        azimuth=value_or_none(channel.azimuth),
                        dip=value_or_none(channel.dip),
                        **nut.channel_kwargs)

                yield nut
                inut += 1

                context = '%s.%s.%s.%s' % (net, sta, loc, cha)

                if channel.response:
                    nut = model.make_response_nut(
                        file_segment=0,
                        file_element=inut,
                        agency=agn,
                        network=net,
                        station=sta,
                        location=loc,
                        channel=cha,
                        tmin=tmin,
                        tmax=tmax,
                        deltat=deltat)

                    try:
                        resp = channel.response.get_squirrel_response(
                            context, **nut.response_kwargs)

                        if 'response' in content:
                            nut.content = resp

                        yield nut
                        inut += 1

                    except Exception as e:
                        logger.warn('Bad instrument response: %s' % str(e))
