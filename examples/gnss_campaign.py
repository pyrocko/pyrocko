from pyrocko import guts
from pyrocko.model import gnss
from pyrocko.example import get_example_data

# Data from Hudnut et al. 1996

# K. W. Hudnut, Z. Shen, M. Murray, S. McClusky, R. King, T. Herring,
# B. Hager, Y. Feng, P. Fang, A. Donnellan, Y. Bock;
# Co-seismic displacements of the 1994 Northridge, California, earthquake.
# Bulletin of the Seismological Society of America ; 86 (1B): S19–S36. doi:

# https://pasadena.wr.usgs.gov/office/hudnut/hudnut/nr_bssa.html

mm = 1e3

fn_stations = 'GPS_Stations-Northridge-1994_Hudnut.csv'
fn_displacements = 'GPS_Data-Northridge-1994_Hudnut.csv'

get_example_data(fn_stations)
get_example_data(fn_displacements)

campaign = gnss.GNSSCampaign(name='Northridge 1994')

# Load the stations
with open(fn_stations, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        row = line.split(',')
        sta = gnss.GNSSStation(
            code=row[0].strip(),
            lat=float(row[1]),
            lon=float(row[2]),
            elevation=float(row[3]))
        campaign.add_station(sta)

# Load the displacements
with open(fn_displacements, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        row = line.split(',')

        station_id = row[0].strip()
        station = campaign.get_station(station_id)

        station.east = gnss.GNSSComponent(
            shift=float(row[1]) / mm,
            sigma=float(row[2]) / mm)
        station.north = gnss.GNSSComponent(
            shift=float(row[7]) / mm,
            sigma=float(row[8]) / mm)
        station.up = gnss.GNSSComponent(
            shift=float(row[14]) / mm,
            sigma=float(row[15]) / mm)

print('Campaign %s has %d stations' % (campaign.name, campaign.nstations))
campaign.dump(filename='GPS_Northridge-1994_Hudnut.yml')

# Load the campaign back in
campaign_loaded = guts.load(filename='GPS_Northridge-1994_Hudnut.yml')
