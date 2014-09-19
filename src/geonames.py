import sys
from collections import namedtuple

GeoName = namedtuple('GeoName', '''
geonameid
name
asciiname
alt_names
latitude
longitude
feature_class
feature_code
country_code
alt_country_code
admin1_code
admin2_code
admin3_code
admin4_code
population
elevation
dem
timezone
modification_date'''.split())

GeoName2 = namedtuple('GeoName2', '''
name
asciiname
latitude
longitude
population
'''.split())


def positive_region(region):
    west, east, south, north = [float(x) for x in region]

    assert -180. - 360. <= west < 180.
    assert -180. < east <= 180. + 360.
    assert -90. <= south < 90.
    assert -90. < north <= 90.

    if east < west:
        east += 360.

    if west < -180.:
        west += 360.
        east += 360.

    return (west, east, south, north)


def ascii_str(u):
    return u.encode('ascii', 'replace')

geoname_types = (
    int, unicode, ascii_str, unicode, float, float, str, str, str, str,
    unicode, unicode, unicode, unicode, int, str, str, str, str)


def load(filepath):
    with open(filepath, 'r') as f:
        for line in f:
            w = line.split('\t')
            yield GeoName(*[t(x.decode('utf8')) for (t, x) in
                            zip(geoname_types, w)])


def load2(filepath, minpop=1000000, region=None):

    if region:
        w, e, s, n = positive_region(region)

    with open(filepath, 'r') as f:
        for line in f:
            t = line.split('\t')
            pop = int(t[14])
            if minpop <= pop:
                lat = float(t[4])
                lon = float(t[5])
                if not region or (
                        (w <= lon <= e or w <= lon + 360. <= e)
                        and (s <= lat <= n)):

                    yield GeoName2(
                        t[1].decode('utf8'),
                        t[2].decode('utf8').encode('ascii', 'replace'),
                        lat, lon, pop)

if __name__ == '__main__':
    for c in load2(sys.argv[1]):
        print c
