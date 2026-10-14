# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Client to get earthquake catalog information from
`GlobalCMT <http://www.globalcmt.org/>`_.
'''

import os
import math
import time
import calendar
import re
import logging
import email.utils

from pyrocko import model, util, config
from pyrocko.moment_tensor import MomentTensor, moment_to_magnitude
from pyrocko.util import Request, urlopen
from .base_catalog import EarthquakeCatalog, NotFound


import numpy as num


logger = logging.getLogger('pyrocko.client.globalcmt')

km = 1000.


class Anon(object):
    pass


class GlobalCMT(EarthquakeCatalog):
    '''
    Client to get earthquake catalog information from
    `GlobalCMT <http://www.globalcmt.org/>`_.
    '''

    def __init__(self):
        self.events = {}

    def flush(self):
        self.events = {}

    def iter_event_names(
            self,
            time_range=None,
            magmin=0.,
            magmax=10.,
            latmin=-90.,
            latmax=90.,
            lonmin=-180.,
            lonmax=180.,
            depthmin=0.,
            depthmax=1000*km):

        yearbeg, monbeg, daybeg = time.gmtime(time_range[0])[:3]
        yearend, monend, dayend = time.gmtime(time_range[1])[:3]

        url = 'http://www.globalcmt.org/cgi-bin/globalcmt-cgi-bin/CMT5/form?' \
            + '&'.join([
                'itype=ymd',
                'yr=%i' % yearbeg, 'mo=%i' % monbeg, 'day=%i' % daybeg,
                'otype=ymd',
                'oyr=%i' % yearend, 'omo=%i' % monend, 'oday=%i' % dayend,
                'jyr=1976', 'jday=1', 'ojyr=1976', 'ojday=1', 'nday=1',
                'lmw=%g' % magmin, 'umw=%g' % magmax,
                'lms=0', 'ums=10',
                'lmb=0', 'umb=10',
                'llat=%g' % latmin, 'ulat=%g' % latmax,
                'llon=%g' % lonmin, 'ulon=%g' % lonmax,
                'lhd=%g' % (depthmin/km), 'uhd=%g' % (depthmax/km),
                'lts=-9999', 'uts=9999',
                'lpe1=0', 'upe1=90',
                'lpe2=0', 'upe2=90',
                'list=5'])

        while True:
            logger.debug('Opening URL: %s' % url)
            req = Request(url)
            page = urlopen(req).read()
            logger.debug('Received page (%i bytes)' % len(page))

            events, more = self._parse_events_page(page)

            for ev in events:
                self.events[ev.name] = ev

            for ev in events:
                if time_range[0] <= ev.time and ev.time <= time_range[1]:
                    yield ev.name

            if more:
                url = more.decode('ascii')
            else:
                break

    def get_event(self, name):
        if name not in self.events:
            t = self._name_to_date(name)
            for name2 in self.iter_event_names(
                    time_range=(t-24*60*60, t+2*24*60*60)):

                if name2 == name:
                    break

        return self.events[name]

    def _parse_events_page(self, page):

        lines = page.splitlines()
        state = 0

        events = []

        def complete(data):
            try:
                t = util.to_time_float(
                    calendar.timegm((
                        data.year, data.month, data.day,
                        data.hour, data.minute, data.seconds)))

                m = num.array(
                    [data.mrr, data.mrt, data.mrp,
                     data.mrt, data.mtt, data.mtp,
                     data.mrp, data.mtp, data.mpp],
                    dtype=float).reshape(3, 3)

                m *= 10.0**(data.exponent-7)
                mt = MomentTensor(m_up_south_east=m)
                ev = model.Event(
                    lat=data.lat,
                    lon=data.lon,
                    time=t,
                    name=data.eventname,
                    depth=data.depth_km*1000.,
                    magnitude=float(mt.moment_magnitude()),
                    duration=data.half_duration * 2.,
                    region=data.region.rstrip(),
                    catalog=data.catalog)

                ev.moment_tensor = mt
                events.append(ev)

            except AttributeError:
                pass

        catalog = 'gCMT'

        data = None
        more = None
        for line in lines:
            if state == 0:

                m = re.search(br'<a href="([^"]+)">More solutions', line)
                if m:
                    more = m.group(1)

                m = re.search(br'From Quick CMT catalog', line)
                if m:
                    catalog = 'gCMT-Q'

                m = re.search(br'Event name:\s+(\S+)', line)
                if m:
                    if data:
                        complete(data)

                    data = Anon()
                    data.eventname = str(m.group(1).decode('ascii'))
                    data.catalog = catalog

                if data:
                    m = re.search(br'Region name:\s+([^<]+)', line)
                    if m:
                        data.region = str(m.group(1).decode('ascii'))

                    m = re.search(
                        br'Date \(y/m/d\): (\d\d\d\d)/(\d+)/(\d+)', line)

                    if m:
                        data.year, data.month, data.day = (
                            int(m.group(1)), int(m.group(2)), int(m.group(3)))

                    m = re.search(br'Timing and location information', line)
                    if m:
                        state = 1

            if state == 1:
                toks = line.split()
                if toks and toks[0] == b'CMT':
                    data.hour, data.minute = [int(x) for x in toks[1:3]]
                    data.seconds, data.lat, data.lon, data.depth_km = [
                        float(x) for x in toks[3:]]

                m = re.search(br'Assumed half duration:\s+(\S+)', line)
                if m:
                    data.half_duration = float(m.group(1))

                m = re.search(br'Mechanism information', line)
                if m:
                    state = 2

            if state == 2:
                m = re.search(br'Exponent for moment tensor:\s+(\d+)', line)
                if m:
                    data.exponent = int(m.group(1))

                toks = line.split()
                if toks and toks[0] == b'CMT':
                    data.mrr, data.mtt, data.mpp, \
                        data.mrt, data.mrp, data.mtp = [
                            float(x) for x in toks[1:]]

                m = re.search(br'^Eigenvector:', line)
                if m:
                    state = 0

        if data is not None:
            complete(data)

        return events, more

    def _name_to_date(self, name):

        if len(name) == 7:
            y, m, d = time.strptime(name[:6], '%m%d%y')[:3]
            if y > 2005:
                y -= 100

        else:
            y, m, d = time.strptime(name[:8], '%Y%m%d')[:3]

        t = util.to_time_float(
            calendar.timegm((y, m, d, 0, 0, 0)))
        return t


g_url_ndk = 'https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog'

# The bulk file covers 1976-01 to 2020-12, monthly files start in 2021.
g_tmin_monthly = util.str_to_time('2021-01-01 00:00:00')

g_month_names = [
    'jan', 'feb', 'mar', 'apr', 'may', 'jun',
    'jul', 'aug', 'sep', 'oct', 'nov', 'dec']


class NDKError(Exception):
    pass


def _touch(path):
    with open(path, 'a'):
        os.utime(path, None)


class NDKRecord(object):
    '''
    Parsed GlobalCMT NDK record.

    The :py:class:`~pyrocko.model.event.Event` object is only created on
    demand, as setting up the moment tensor is comparatively expensive.
    '''

    __slots__ = (
        'name', 'time', 'lat', 'lon', 'depth', 'magnitude', 'half_duration',
        'region', 'm_up_south_east', 'solution_type', 'time_modified',
        '_event')

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        self._event = None

    @property
    def solution_time(self):
        return self.time_modified or 0.0

    def get_event(self):
        if self._event is None:
            mt = MomentTensor(m_up_south_east=self.m_up_south_east)

            extras = {}
            if self.solution_type:
                extras['catalog_version'] = self.solution_type

            if self.time_modified is not None:
                extras['catalog_time_modified'] = self.time_modified

            self._event = model.Event(
                lat=self.lat,
                lon=self.lon,
                time=self.time,
                name=self.name,
                depth=self.depth,
                magnitude=float(mt.moment_magnitude()),
                magnitude_type='Mw',
                duration=None if self.half_duration is None
                else self.half_duration * 2.,
                region=self.region,
                catalog='gCMT',
                moment_tensor=mt,
                extras=extras)

        return self._event


def parse_ndk_records(data):
    '''
    Parse GlobalCMT NDK formatted text into :py:class:`NDKRecord` objects.

    See :py:func:`parse_ndk`.
    '''

    lines = data.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()

    if len(lines) % 5 != 0:
        raise NDKError(
            'Number of lines (%i) is not a multiple of 5.' % len(lines))

    records = []
    for i in range(0, len(lines), 5):
        l1, l2, l3, l4, _ = [line.ljust(80) for line in lines[i:i+5]]
        try:
            # Line 1: hypocenter reference
            year, month, day = [int(x) for x in l1[5:15].split('/')]
            hour, minute = int(l1[16:18]), int(l1[19:21])
            seconds = float(l1[22:26])
            region = l1[56:80].strip()

            # Line 2: CMT event name, moment rate function
            name = l2[0:16].strip()[1:]
            m = re.search(r'(TRI|BOX)HD:\s*([0-9.]+)', l2[61:])
            half_duration = float(m.group(2)) if m else None

            # Line 3: centroid parameters relative to hypocenter reference,
            # solution type and timestamp
            if not l3.startswith('CENTROID:'):
                raise NDKError('Expected CENTROID line.')

            time_shift = float(l3[9:18])
            lat = float(l3[23:29])
            lon = float(l3[35:42])
            depth = float(l3[48:53]) * km

            stamp = l3[64:80].strip()
            solution_type = stamp[:1] or None
            digits = stamp[2:].replace(' ', '0')
            if len(digits) == 14 and digits.strip('0'):
                time_modified = util.to_time_float(calendar.timegm(
                    time.strptime(digits, '%Y%m%d%H%M%S')))
            else:
                time_modified = None

            # Line 4: moment tensor
            toks = l4.split()
            exponent = int(toks[0])
            mrr, mtt, mpp, mrt, mrp, mtp = [float(x) for x in toks[1:13:2]]

        except (ValueError, IndexError, NDKError) as e:
            raise NDKError(
                'Cannot parse NDK record starting at line %i: %s' % (
                    i+1, e)) from None

        t = util.to_time_float(calendar.timegm(
            (year, month, day, hour, minute, 0))) + seconds + time_shift

        m = num.array(
            [[mrr, mrt, mrp],
             [mrt, mtt, mtp],
             [mrp, mtp, mpp]], dtype=float) * 10.0**(exponent-7)

        records.append(NDKRecord(
            name=name,
            time=t,
            lat=lat,
            lon=lon,
            depth=depth,
            # Same definition as MomentTensor.moment_magnitude, used for
            # filtering before the event is created.
            magnitude=float(moment_to_magnitude(
                num.linalg.norm(m) / math.sqrt(2.))),
            half_duration=half_duration,
            region=region,
            m_up_south_east=m,
            solution_type=solution_type,
            time_modified=time_modified))

    return records


def parse_ndk(data):
    '''
    Parse GlobalCMT NDK formatted text into Pyrocko events.

    Fixed column positions are used, as given in the NDK format description
    (https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/allorder.ndk_explained).
    Events are named by the CMT event name without its leading source type
    character, e.g. ``'201001122153A'``, consistent with :py:class:`GlobalCMT`.
    The solution type (``'Q'``: quick, ``'S'``: standard, ``'O'``: old) is
    stored in ``extras['catalog_version']``, the time the solution has been
    computed, if available, in ``extras['catalog_time_modified']``.

    :param data:
        NDK file content.
    :type data:
        :py:class:`str`

    :returns:
        List of :py:class:`~pyrocko.model.event.Event` objects.
    '''

    return [rec.get_event() for rec in parse_ndk_records(data)]


class GlobalCMTNDK(EarthquakeCatalog):
    '''
    Client to get the GlobalCMT catalog from its NDK files.

    Instead of querying the GlobalCMT web search, the catalog files are
    downloaded and cached locally: the bulk file for 1976-2020 (about 23 MB),
    monthly files with standard solutions since 2021, and the file of quick
    solutions for recent events. For months without a published monthly
    file, quick solutions are used.

    Unlike :py:class:`GlobalCMT`, quick and standard solutions share the
    catalog name ``'gCMT'``, so that a standard solution can be recognized
    as an update of a quick one. The solution type and time are available
    in the event's ``extras``, see :py:func:`parse_ndk`.

    :param cache_dir:
        Directory where downloaded files are kept. Default is a subdirectory
        of Pyrocko's cache directory.
    :param max_age:
        Time [s] after which files which may change (quick solutions,
        not yet published monthly files) are checked for updates.
    '''

    def __init__(self, cache_dir=None, max_age=3600.):
        if cache_dir is None:
            cache_dir = os.path.join(
                config.config().cache_dir, 'globalcmt-ndk')

        self._cache_dir = cache_dir
        self._max_age = max_age
        self._parsed = {}
        self.events = {}

    def flush(self):
        self.events = {}
        self._parsed = {}

    def iter_event_names(
            self,
            time_range=None,
            magmin=0.,
            magmax=10.,
            latmin=-90.,
            latmax=90.,
            lonmin=-180.,
            lonmax=180.,
            depthmin=0.,
            depthmax=1000*km):

        tmin, tmax = time_range
        records = [
            rec for rec in self._get_records(tmin, tmax)
            if tmin <= rec.time <= tmax
            and magmin <= rec.magnitude <= magmax
            and latmin <= rec.lat <= latmax
            and lonmin <= rec.lon <= lonmax
            and depthmin <= rec.depth <= depthmax]

        records.sort(key=lambda rec: rec.time)
        events = [rec.get_event() for rec in records]
        for ev in events:
            self.events[ev.name] = ev

        for ev in events:
            yield ev.name

    def get_event(self, name):
        if name not in self.events:
            t = GlobalCMT._name_to_date(None, name)
            for name2 in self.iter_event_names(
                    time_range=(t-24*60*60, t+2*24*60*60)):

                if name2 == name:
                    break

        if name not in self.events:
            raise NotFound()

        return self.events[name]

    def _get_records(self, tmin, tmax):
        records = []
        if tmin < g_tmin_monthly:
            records.extend(self._load('jan76_dec20.ndk', static=True))

        quick_months = set()
        for year, month in self._iter_months(
                max(tmin, g_tmin_monthly), min(tmax, time.time())):

            path = 'NEW_MONTHLY/%04i/%s%02i.ndk' % (
                year, g_month_names[month-1], year % 100)

            monthly = self._load(path, static=True, optional=True)
            if monthly is None:
                quick_months.add((year, month))
            else:
                records.extend(monthly)

        if quick_months:
            for rec in self._load('NEW_QUICK/qcmt.ndk', static=False):
                if time.gmtime(rec.time)[:2] in quick_months:
                    records.append(rec)

        # In case an event appears more than once, use latest solution.
        latest = {}
        for rec in records:
            other = latest.get(rec.name)
            if other is None or rec.solution_time > other.solution_time:
                latest[rec.name] = rec

        return list(latest.values())

    def _iter_months(self, tmin, tmax):
        if tmax < tmin:
            return

        year, month = time.gmtime(tmin)[:2]
        year_end, month_end = time.gmtime(tmax)[:2]
        while (year, month) <= (year_end, month_end):
            yield year, month
            month += 1
            if month == 13:
                year, month = year + 1, 1

    def _load(self, path, static, optional=False):
        fpath = self._fetch(path, static, optional)
        if fpath is None:
            return None

        mtime = os.stat(fpath).st_mtime
        if path not in self._parsed or self._parsed[path][0] != mtime:
            logger.debug('Parsing NDK file: %s' % fpath)
            with open(fpath, 'r', encoding='latin-1') as f:
                self._parsed[path] = (mtime, parse_ndk_records(f.read()))

        return self._parsed[path][1]

    def _fetch(self, path, static, optional):
        '''
        Get local path of cached file, downloading it if needed.

        Static files are downloaded once. Other files and files known to be
        missing are checked again when the last check is older than
        ``max_age``. Returns ``None`` for missing optional files.
        '''

        import requests

        fpath = os.path.join(self._cache_dir, path)
        fpath_checked = fpath + '.checked'
        fpath_missing = fpath + '.missing'

        def age(p):
            try:
                return time.time() - os.stat(p).st_mtime
            except FileNotFoundError:
                return float('inf')

        if os.path.exists(fpath) and (
                static or age(fpath_checked) < self._max_age):
            return fpath

        if os.path.exists(fpath_missing) \
                and age(fpath_missing) < self._max_age:
            return None

        url = '%s/%s' % (g_url_ndk, path)
        headers = {}
        if os.path.exists(fpath):
            headers['If-Modified-Since'] = email.utils.formatdate(
                os.stat(fpath).st_mtime, usegmt=True)

        logger.info('Checking GlobalCMT NDK file: %s' % url)
        r = requests.get(url, headers=headers, timeout=120.)

        util.ensuredirs(fpath)
        if r.status_code == 404 and optional:
            _touch(fpath_missing)
            return None

        if r.status_code != 304:
            r.raise_for_status()
            logger.info('Downloaded %i bytes.' % len(r.content))
            fpath_tmp = fpath + '.tmp-%i' % os.getpid()
            with open(fpath_tmp, 'wb') as f:
                f.write(r.content)

            if 'Last-Modified' in r.headers:
                tmod = email.utils.parsedate_to_datetime(
                    r.headers['Last-Modified']).timestamp()
                os.utime(fpath_tmp, (tmod, tmod))

            os.replace(fpath_tmp, fpath)

        if os.path.exists(fpath_missing):
            os.unlink(fpath_missing)

        _touch(fpath_checked)
        return fpath
