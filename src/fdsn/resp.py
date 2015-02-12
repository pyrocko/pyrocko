import time
import re
import logging

from pyrocko import util, guts, io_common
from pyrocko.fdsn import station as fs

logger = logging.getLogger('pyrocko.fdsn.resp')


class RespError(io_common.FileLoadError):
    pass


def ppolezero(s):
    v = s.split()
    return fs.PoleZero(
        number=int(v[0]),
        real=fs.FloatNoUnit(
            value=float(v[1]),
            plus_error=float(v[3]) or None,
            minus_error=float(v[3]) or None),
        imaginary=fs.FloatNoUnit(
            value=float(v[2]),
            plus_error=float(v[4]) or None,
            minus_error=float(v[4]) or None))


def pcfu(s):
    v = map(float, s.split())
    return fs.FloatWithUnit(
        value=float(v[-2]),
        plus_error=float(v[-1]) or None,
        minus_error=float(v[-1]) or None)


def pnc(s):
    v = map(float, s.split())
    return fs.NumeratorCoefficient(i=int(v[0]), value=float(v[1]))


def punit(s):
    return s.split()[0]


def psymmetry(s):
    return {
        'A': 'NONE',
        'B': 'ODD',
        'C': 'EVEN'}[s.upper()]


def ptftype(s):
    if s.startswith('A'):
        return 'LAPLACE (RADIANS/SECOND)'
    elif s.startswith('B'):
        return 'LAPLACE (HERTZ)'
    elif s.startswith('D'):
        return 'DIGITAL (Z-TRANSFORM)'
    else:
        raise RespError('unknown pz transfer function type')


def pcftype(s):
    if s.startswith('A'):
        return 'ANALOG (RADIANS/SECOND)'
    elif s.startswith('B'):
        return 'ANALOG (HERTZ)'
    elif s.startswith('D'):
        return 'DIGITAL'
    else:
        raise RespError('unknown cf transfer function type')


def pblock_053(content):
    stage_number = int(get1(content, '04'))

    pzs = fs.PolesZeros(
        pz_transfer_function_type=ptftype(get1(content, '03')),
        input_units=fs.Units(name=punit(get1(content, '05'))),
        output_units=fs.Units(name=punit(get1(content, '06'))),
        normalization_factor=float(get1(content, '07')),
        normalization_frequency=fs.Frequency(
            value=float(get1(content, '08'))),

        zero_list=map(ppolezero, getn(content, '10-13')),
        pole_list=map(ppolezero, getn(content, '15-18')))

    for i, x in enumerate(pzs.zero_list):
        x.number = i

    for i, x in enumerate(pzs.pole_list):
        x.number = i

    return stage_number, pzs


def pblock_058(content):
    stage_number = int(get1(content, '03'))

    gain = fs.Gain(
        value=float(get1(content, '04')),
        frequency=float(get1(content, '05').split()[0]))

    return stage_number, gain


def pblock_054(content):
    stage_number = int(get1(content, '04'))

    cfs = fs.Coefficients(
        cf_transfer_function_type=pcftype(get1(content, '03')),
        input_units=fs.Units(name=punit(get1(content, '05'))),
        output_units=fs.Units(name=punit(get1(content, '06'))),
        numerator_list=map(pcfu, getn(content, '08-09')),
        denominator_list=map(pcfu, getn(content, '11-12')))

    return stage_number, cfs


def pblock_057(content):
    stage_number = int(get1(content, '03'))

    deci = fs.Decimation(
        input_sample_rate=fs.Frequency(value=float(get1(content, '04'))),
        factor=int(get1(content, '05')),
        offset=int(get1(content, '06')),
        delay=fs.FloatWithUnit(value=float(get1(content, '07'))),
        correction=fs.FloatWithUnit(value=float(get1(content, '08'))))

    return stage_number, deci


def pblock_061(content):
    stage_number = int(get1(content, '03'))

    fir = fs.FIR(
        name=get1(content, '04', optional=True),
        input_units=fs.Units(name=punit(get1(content, '06'))),
        output_units=fs.Units(name=punit(get1(content, '07'))),
        symmetry=psymmetry(get1(content, '05')),
        numerator_coefficient_list=map(pnc, getn(content, '09')))

    return stage_number, fir


bdefs = {
    '050': {
        'name': 'Station Identifier Blockette',
    },
    '052': {
        'name': 'Channel Identifier Blockette',
    },
    '053': {
        'name': 'Response (Poles & Zeros) Blockette',
        'parse': pblock_053,
    },
    '054': {
        'name': 'Response (Coefficients) Blockette',
        'parse': pblock_054,
    },
    '057': {
        'name': 'Decimation Blockette',
        'parse': pblock_057,
    },
    '058': {
        'name': 'Channel Sensitivity/Gain Blockette',
        'parse': pblock_058,
    },
    '061': {
        'name': 'FIR Response Blockette',
        'parse': pblock_061,
    },
}


def parse1(f):
    for line in f:
        line = line.rstrip('\r\n')
        m = re.match(
            r'\s*(#(.+)|B(\d\d\d)F(\d\d(-\d\d)?)\s+(([^:]+):\s*)?(.*))', line)
        if m:
            if m.group(2):
                pass

            elif m.group(3):
                block = m.group(3)
                field = m.group(4)
                key = m.group(7)
                value = m.group(8)
                yield block, field, key, value


def parse2(f):
    current_b = None
    content = []
    for block, field, key, value in parse1(f):
        if current_b != block or field == '03':
            if current_b is not None:
                yield current_b, content

            current_b = block
            content = []

        content.append((field, key, value))

    if current_b is not None:
        yield current_b, content


def parse3(f):
    state = [None, None, []]
    for block, content in parse2(f):
        if block == '050' and state[0] and state[1]:
            yield state
            state = [None, None, []]

        if block == '050':
            state[0] = content
        elif block == '052':
            state[1] = content
        else:
            state[2].append((block, content))

    if state[0] and state[1]:
        yield state


def get1(content, field, default=None, optional=False):
    for field_, _, value in content:
        if field_ == field:
            return value
    else:
        if optional:
            return None
        elif default is not None:
            return default
        else:
            raise RespError('key not found: %s' % field)


def getn(content, field):
    l = []
    for field_, _, value in content:
        if field_ == field:
            l.append(value)
    return l


def pdate(s):
    if s.startswith('2599') or s.startswith('2999'):
        return None
    elif s.lower().startswith('no'):
        return None
    else:
        return util.str_to_time(s, format='%Y,%j,%H:%M:%S.OPTFRAC')


def ploc(s):
    if s == '??':
        return ''
    else:
        return s


def gett(l, t):
    return [x for x in l if isinstance(x, t)]


def gett1o(l, t):
    l = [x for x in l if isinstance(x, t)]
    if len(l) == 0:
        return None
    elif len(l) == 1:
        return l[0]
    else:
        raise RespError('duplicate entry')


def gett1(l, t):
    l = [x for x in l if isinstance(x, t)]
    if len(l) == 0:
        raise RespError('entry not found')
    elif len(l) == 1:
        return l[0]
    else:
        raise RespError('duplicate entry')


class ChannelResponse(guts.Object):
    '''Response information + channel codes and time span.'''

    codes = guts.Tuple.T(4, guts.String.T(default=''))
    start_date = guts.Timestamp.T()
    end_date = guts.Timestamp.T()
    response = fs.Response.T()


def iload_fh(f):
    '''Read RESP information from open file handle.'''

    for sc, cc, rcs in parse3(f):
        nslc = (
            get1(sc, '16'),
            get1(sc, '03'),
            ploc(get1(cc, '03', '')),
            get1(cc, '04'))

        tmin = pdate(get1(cc, '22'))
        tmax = pdate(get1(cc, '23'))

        stage_elements = {}

        for block, content in rcs:
            istage, x = bdefs[block]['parse'](content)
            x.validate()
            if istage not in stage_elements:
                stage_elements[istage] = []

            stage_elements[istage].append(x)

        istages = sorted(stage_elements.keys())
        stages = []
        totalgain = None
        for istage in istages:
            elements = stage_elements[istage]
            if istage == 0:
                totalgain = gett1(elements, fs.Gain)
            else:
                stage = fs.ResponseStage(
                    number=istage,
                    poles_zeros_list=gett(elements, fs.PolesZeros),
                    coefficients_list=gett(elements, fs.Coefficients),
                    fir=gett1o(elements, fs.FIR),
                    decimation=gett1o(elements, fs.Decimation),
                    stage_gain=gett1o(elements, fs.Gain))

                stages.append(stage)

        if totalgain and stages:
            resp = fs.Response(
                instrument_sensitivity=fs.Sensitivity(
                    value=totalgain.value,
                    frequency=totalgain.frequency,
                    input_units=stages[0].input_units,
                    output_units=stages[-1].output_units),
                stage_list=stages)

            yield ChannelResponse(
                codes=nslc,
                start_date=tmin,
                end_date=tmax,
                response=resp)

        else:
            raise RespError('incomplete response information')


iload_filename, iload_dirname, iload_glob, iload = util.make_iload_family(
    iload_fh, 'RESP', ':py:class:`ChannelResponse`')


def make_stationxml(pyrocko_stations, channel_responses):
    '''Create stationxml from pyrocko station list and RESP information.

    :param pyrocko_stations: list of :py:class:`pyrocko.model.Station` objects
    :param channel_responses: iterable yielding :py:class:`ChannelResponse`
        objects
    :returns: :py:class:`pyrocko.fdsn.station.FDSNStationXML` object with
        merged information

    If no station information is available for any response information, it
    is skipped and a warning is emitted.
    '''
    pstations = dict((s.nsl(), s) for s in pyrocko_stations)
    networks = {}
    stations = {}
    for (net, sta, loc) in sorted(pstations.keys()):
        pstation = pstations[net, sta, loc]
        if net not in networks:
            networks[net] = fs.Network(code=net)

        if (net, sta) not in stations:
            stations[net, sta] = fs.Station(
                code=sta,
                latitude=fs.Latitude(pstation.lat),
                longitude=fs.Longitude(pstation.lon),
                elevation=fs.Distance(pstation.elevation))

            networks[net].station_list.append(stations[net, sta])

    for cr in channel_responses:
        net, sta, loc, cha = cr.codes
        if (net, sta, loc) in pstations:
            pstation = pstations[net, sta, loc]
            channel = fs.Channel(
                code=cha,
                location_code=loc,
                start_date=cr.start_date,
                end_date=cr.end_date,
                latitude=fs.Latitude(pstation.lat),
                longitude=fs.Longitude(pstation.lon),
                elevation=fs.Distance(pstation.elevation),
                depth=fs.Distance(pstation.depth),
                response=cr.response)

            stations[net, sta].channel_list.append(channel)
        else:
            logger.warn('no station information for %s.%s.%s' %
                        (net, sta, loc))

    for station in stations.values():
        station.channel_list.sort(key=lambda c: (c.location_code, c.code))

    return fs.FDSNStationXML(
        source='Converted from Pyrocko stations file and RESP information',
        created=time.time(),
        network_list=[networks[net_] for net_ in sorted(networks.keys())])


if __name__ == '__main__':
    import sys
    from pyrocko import model

    util.setup_logging(__name__)

    if len(sys.argv) < 2:
        sys.exit('usage: python -m pyrocko.station.resp <stations> <resp> ...')

    stations = model.load_stations(sys.argv[1])

    sxml = make_stationxml(stations, iload(sys.argv[2:]))

    print sxml.dump_xml()
