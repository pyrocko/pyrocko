import logging
import sys
import os
from collections import namedtuple
import os.path as op

from pyrocko import util, config

logger = logging.getLogger('pyrocko.geonames')

GeoName = namedtuple('GeoName', '''
geonameid
name
asciiname
alt_names
lat
lon
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
lat
lon
population
'''.split())

base_url = 'http://download.geonames.org/export/dump/'


def download_file(fn, dirpath):
    import urllib2
    url = base_url + '/' + fn
    fpath = op.join(dirpath, fn)
    logger.info('starting download of %s' % url)

    util.ensuredirs(fpath)
    f = urllib2.urlopen(url)
    fpath_tmp = fpath + '.%i.temp' % os.getpid()
    g = open(fpath_tmp, 'wb')
    while True:
        data = f.read(1024)
        if not data:
            break
        g.write(data)

    g.close()
    f.close()

    os.rename(fpath_tmp, fpath)

    logger.info('finished download of %s' % url)


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


def load(zfn, fn, minpop=1000000, region=None):
    geonames_dir = config.config().geonames_dir
    filepath = op.join(geonames_dir, zfn or fn)
    if not os.path.exists(filepath):
        download_file(zfn or fn, geonames_dir)

    if region:
        w, e, s, n = positive_region(region)

    if zfn is not None:
        import zipfile
        z = zipfile.ZipFile(filepath, 'r')
        f = z.open(fn, 'r')
    else:
        z = None
        f = open(filepath, 'r')


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

    f.close()
    if z is not None:
        z.close()

