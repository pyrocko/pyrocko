#!/usr/bin/env python3
import numpy as num
import requests

from pyrocko.model import gnss, location
from pyrocko.plot.automap import Map

km = 1e3
DIST_MAX = 80. * km
RIDGECREST_EQ = location.Location(lat=35.766, lon=-117.605)
GNSS_URL = 'http://geodesy.unr.edu/news_items/20190707/ci38457511_forweb.txt'

fname = 'ci38457511_forweb.txt'
# Download the data
with open(fname, 'wb') as f:
    print('Downloading data from UNR...')
    r = requests.get(GNSS_URL)
    f.write(r.content)

campaign = gnss.GNSSCampaign(name='Ridgecrest coseismic (UNR)')
print('Loading Ridgecrest GNSS data from %s (max_dist = %.1f km)' %
      (fname, DIST_MAX / km))

# load station names and data
with open(fname, 'r') as f:
    for header in range(2):
        next(f)
    names = [line.split(' ')[0] for line in f]
gnss_data = num.loadtxt(fname, skiprows=2, usecols=(1, 2, 3, 4, 5, 6, 7, 8))

for ista, sta_data in enumerate(gnss_data):
    name = names[ista]
    lon, lat, de, dn, du, sde, sdn, sdu = map(float, sta_data)

    station = gnss.GNSSStation(
        code=name,
        lat=lat,
        lon=lon,
        elevation=0.)

    if station.distance_to(RIDGECREST_EQ) > DIST_MAX:
        continue

    station.east = gnss.GNSSComponent(
        shift=de,
        sigma=sde)
    station.north = gnss.GNSSComponent(
        shift=dn,
        sigma=sdn)
    station.up = gnss.GNSSComponent(
        shift=du,
        sigma=sdu)

    campaign.add_station(station)

print('Selected %d stations, saving to ridgecrest_coseis_gnss.yml'
      % (campaign.nstations))
campaign.dump(filename='2019-ridgecrest_gnss_unr.yml')

print('Creating map...')
m = Map(
    lat=RIDGECREST_EQ.lat,
    lon=RIDGECREST_EQ.lon,
    radius=DIST_MAX * 1.5,
    width=20.,
    height=14.)
m.add_gnss_campaign(campaign)
m.save('2019-ridgecrest_GNSS-UNR.pdf')
