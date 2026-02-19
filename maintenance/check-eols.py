import time
from pyrocko import util
import requests
from subprocess import check_output


def parse_date(s):
    return util.str_to_time(s, format='%Y-%m-%d')


def get_python_version(distribution, release):
    return check_output([
        'docker',
        'run',
        '--rm',
        '-q',
        '%s:%s' % (distribution, release),
        'bash',
        '-c',
        'apt update >/dev/null 2>&1 ; '
        'apt install -y python3 python3-numpy python3-matplotlib python3-setuptools >/dev/null 2>&1 ; '  # noqa
        'python3 -c \'import sys, numpy, matplotlib, setuptools ; print("python: %s, numpy: %s, matplotlib: %s, setuptools: %s" % (sys.version.split()[0], numpy.__version__, matplotlib.__version__, setuptools.__version__))\'']).decode('utf8')  # noqa


distributions = [
    'debian',
    'ubuntu',
]


tnow = time.time()
for distribution in distributions:
    response = requests.get(
        'https://endoflife.date/api/v1/products/%s/' % distribution).json()

    for release in response['result']['releases']:
        # print(release)
        tmin = parse_date(release['releaseDate'])
        tmax = parse_date(release['eolFrom'])
        if tmin <= tnow <= tmax:
            python_version = get_python_version(distribution, release['name'])
            print(
                distribution,
                release['name'],
                release['releaseDate'],
                release['eolFrom'],
                python_version)
