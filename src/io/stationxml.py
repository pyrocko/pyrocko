# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
from __future__ import absolute_import, division, print_function

import sys
import time
import logging
import datetime
import calendar
import math
import copy

import numpy as num

from pyrocko.guts import (StringChoice, StringPattern, UnicodePattern, String,
                          Unicode, Int, Float, List, Object, Timestamp,
                          ValidationError, TBase, re_tz)
from pyrocko.guts import load_xml  # noqa

import pyrocko.model
from pyrocko import trace, util

try:
    newstr = unicode
except NameError:
    newstr = str

guts_prefix = 'sx'

guts_xmlns = 'http://www.fdsn.org/xml/station/1'

logger = logging.getLogger('pyrocko.io.stationxml')

conversion = {
    ('M', 'M'): None,
    ('M/S', 'M'): trace.IntegrationResponse(1),
    ('M/S**2', 'M'): trace.IntegrationResponse(2),
    ('M', 'M/S'): trace.DifferentiationResponse(1),
    ('M/S', 'M/S'): None,
    ('M/S**2', 'M/S'): trace.IntegrationResponse(1),
    ('M', 'M/S**2'): trace.DifferentiationResponse(2),
    ('M/S', 'M/S**2'): trace.DifferentiationResponse(1),
    ('M/S**2', 'M/S**2'): None}


class Inconsistencies(Exception):
    pass


class NoResponseInformation(Exception):
    pass


class MultipleResponseInformation(Exception):
    pass


def wrap(s, width=80, indent=4):
    words = s.split()
    lines = []
    t = []
    n = 0
    for w in words:
        if n + len(w) >= width:
            lines.append(' '.join(t))
            n = indent
            t = [' '*(indent-1)]

        t.append(w)
        n += len(w) + 1

    lines.append(' '.join(t))
    return '\n'.join(lines)


def same(x, eps=0.0):
    if any(type(x[0]) != type(r) for r in x):
        return False

    if isinstance(x[0], float):
        return all(abs(r-x[0]) <= eps for r in x)
    else:
        return all(r == x[0] for r in x)


def same_sample_rate(a, b, eps=1.0e-6):
    return abs(a - b) < min(a, b)*eps


def evaluate1(resp, f):
    return resp.evaluate(num.array([f], dtype=num.float))[0]


class InconsistentResponseInformation(Exception):
    pass


def check_resp(resp, value, frequency, limit_db, prelude=''):
    if not value or not frequency:
        logger.warn('Cannot validate frequency response')
        return

    value_resp = num.abs(evaluate1(resp, frequency))

    if value_resp == 0.0:
        raise InconsistentResponseInformation(
            '%s\n'
            '  computed response is zero' % prelude)

    diff_db = 20.0 * num.log10(value_resp/value)

    if num.abs(diff_db) > limit_db:
        raise InconsistentResponseInformation(
            '%s\n'
            '  reported value: %g\n'
            '  computed value: %g\n'
            '  at frequency [Hz]: %g\n'
            '  difference [dB]: %g\n'
            '  limit [dB]: %g' % (
                prelude,
                value,
                value_resp,
                frequency,
                diff_db,
                limit_db))


def tts(t):
    if t is None:
        return '?'
    else:
        return util.tts(t, format='%Y-%m-%d')


this_year = time.gmtime()[0]


class DummyAwareOptionalTimestamp(Object):
    dummy_for = float

    class __T(TBase):

        def regularize_extra(self, val):
            if isinstance(val, datetime.datetime):
                tt = val.utctimetuple()
                val = calendar.timegm(tt) + val.microsecond * 1e-6

            elif isinstance(val, datetime.date):
                tt = val.timetuple()
                val = float(calendar.timegm(tt))

            elif isinstance(val, (str, newstr)):
                val = val.strip()

                tz_offset = 0

                m = re_tz.search(val)
                if m:
                    sh = m.group(2)
                    sm = m.group(4)
                    tz_offset = (int(sh)*3600 if sh else 0) \
                        + (int(sm)*60 if sm else 0)

                    val = re_tz.sub('', val)

                if val[10] == 'T':
                    val = val.replace('T', ' ', 1)

                try:
                    val = util.str_to_time(val) - tz_offset

                except util.TimeStrError:
                    year = int(val[:4])
                    if sys.maxsize > 2**32:  # if we're on 64bit
                        if year > this_year + 100:
                            return None  # StationXML contained a dummy date

                    else:  # 32bit end of time is in 2038
                        if this_year < 2037 and year > 2037 or year < 1903:
                            return None  # StationXML contained a dummy date

                    raise

            elif isinstance(val, int):
                val = float(val)

            else:
                raise ValidationError(
                    '%s: cannot convert "%s" to float' % (self.xname(), val))

            return val

        def to_save(self, val):
            return datetime.datetime.utcfromtimestamp(val)

        def to_save_xml(self, val):
            return datetime.datetime.utcfromtimestamp(val).isoformat() + 'Z'


class Nominal(StringChoice):
    choices = [
        'NOMINAL',
        'CALCULATED']


class Email(UnicodePattern):
    pattern = u'[\\w\\.\\-_]+@[\\w\\.\\-_]+'


class RestrictedStatus(StringChoice):
    choices = [
        'open',
        'closed',
        'partial']


class Type(StringChoice):
    choices = [
        'TRIGGERED',
        'CONTINUOUS',
        'HEALTH',
        'GEOPHYSICAL',
        'WEATHER',
        'FLAG',
        'SYNTHESIZED',
        'INPUT',
        'EXPERIMENTAL',
        'MAINTENANCE',
        'BEAM']

    class __T(StringChoice.T):
        def validate_extra(self, val):
            if val not in self.choices:
                logger.warn(
                    'channel type: "%s" is not a valid choice out of %s' %
                    (val, repr(self.choices)))


class PzTransferFunction(StringChoice):
    choices = [
        'LAPLACE (RADIANS/SECOND)',
        'LAPLACE (HERTZ)',
        'DIGITAL (Z-TRANSFORM)']


class Symmetry(StringChoice):
    choices = [
        'NONE',
        'EVEN',
        'ODD']


class CfTransferFunction(StringChoice):

    class __T(StringChoice.T):
        def validate(self, val, regularize=False, depth=-1):
            if regularize:
                try:
                    val = str(val)
                except ValueError:
                    raise ValidationError(
                        '%s: cannot convert to string %s' % (self.xname,
                                                             repr(val)))

                val = self.dummy_cls.replacements.get(val, val)

            self.validate_extra(val)
            return val

    choices = [
        'ANALOG (RADIANS/SECOND)',
        'ANALOG (HERTZ)',
        'DIGITAL']

    replacements = {
        'ANALOG (RAD/SEC)': 'ANALOG (RADIANS/SECOND)',
        'ANALOG (HZ)': 'ANALOG (HERTZ)',
    }


class Approximation(StringChoice):
    choices = [
        'MACLAURIN']


class PhoneNumber(StringPattern):
    pattern = '[0-9]+-[0-9]+'


class Site(Object):
    '''Description of a site location using name and optional
    geopolitical boundaries (country, city, etc.).'''

    name = Unicode.T(xmltagname='Name')
    description = Unicode.T(optional=True, xmltagname='Description')
    town = Unicode.T(optional=True, xmltagname='Town')
    county = Unicode.T(optional=True, xmltagname='County')
    region = Unicode.T(optional=True, xmltagname='Region')
    country = Unicode.T(optional=True, xmltagname='Country')


class ExternalReference(Object):
    '''This type contains a URI and description for external data that
    users may want to reference in StationXML.'''

    uri = String.T(xmltagname='URI')
    description = Unicode.T(xmltagname='Description')


class Units(Object):
    '''A type to document units. Corresponds to SEED blockette 34.'''

    def __init__(self, name=None, **kwargs):
        Object.__init__(self, name=name, **kwargs)

    name = String.T(xmltagname='Name')
    description = Unicode.T(optional=True, xmltagname='Description')


class Counter(Int):
    pass


class SampleRateRatio(Object):
    '''Sample rate expressed as number of samples in a number of
    seconds.'''

    number_samples = Int.T(xmltagname='NumberSamples')
    number_seconds = Int.T(xmltagname='NumberSeconds')


class Gain(Object):
    '''Complex type for sensitivity and frequency ranges. This complex
    type can be used to represent both overall sensitivities and
    individual stage gains. The FrequencyRangeGroup is an optional
    construct that defines a pass band in Hertz ( FrequencyStart and
    FrequencyEnd) in which the SensitivityValue is valid within the
    number of decibels specified in FrequencyDBVariation.'''

    def __init__(self, value=None, **kwargs):
        Object.__init__(self, value=value, **kwargs)

    value = Float.T(optional=True, xmltagname='Value')
    frequency = Float.T(optional=True, xmltagname='Frequency')

    def summary(self):
        return 'gain(%g @ %g)' % (self.value, self.frequency)


class NumeratorCoefficient(Object):
    i = Int.T(optional=True, xmlstyle='attribute')
    value = Float.T(xmlstyle='content')


class FloatNoUnit(Object):
    def __init__(self, value=None, **kwargs):
        Object.__init__(self, value=value, **kwargs)

    plus_error = Float.T(optional=True, xmlstyle='attribute')
    minus_error = Float.T(optional=True, xmlstyle='attribute')
    value = Float.T(xmlstyle='content')


class FloatWithUnit(FloatNoUnit):
    unit = String.T(optional=True, xmlstyle='attribute')


class Equipment(Object):
    resource_id = String.T(optional=True, xmlstyle='attribute')
    type = String.T(optional=True, xmltagname='Type')
    description = Unicode.T(optional=True, xmltagname='Description')
    manufacturer = Unicode.T(optional=True, xmltagname='Manufacturer')
    vendor = Unicode.T(optional=True, xmltagname='Vendor')
    model = Unicode.T(optional=True, xmltagname='Model')
    serial_number = String.T(optional=True, xmltagname='SerialNumber')
    installation_date = DummyAwareOptionalTimestamp.T(
        optional=True,
        xmltagname='InstallationDate')
    removal_date = DummyAwareOptionalTimestamp.T(
        optional=True,
        xmltagname='RemovalDate')
    calibration_date_list = List.T(Timestamp.T(xmltagname='CalibrationDate'))


class PhoneNumber(Object):
    description = Unicode.T(optional=True, xmlstyle='attribute')
    country_code = Int.T(optional=True, xmltagname='CountryCode')
    area_code = Int.T(xmltagname='AreaCode')
    phone_number = PhoneNumber.T(xmltagname='PhoneNumber')


class BaseFilter(Object):
    '''The BaseFilter is derived by all filters.'''

    resource_id = String.T(optional=True, xmlstyle='attribute')
    name = String.T(optional=True, xmlstyle='attribute')
    description = Unicode.T(optional=True, xmltagname='Description')
    input_units = Units.T(optional=True, xmltagname='InputUnits')
    output_units = Units.T(optional=True, xmltagname='OutputUnits')


class Sensitivity(Gain):
    '''Sensitivity and frequency ranges. The FrequencyRangeGroup is an
    optional construct that defines a pass band in Hertz
    (FrequencyStart and FrequencyEnd) in which the SensitivityValue is
    valid within the number of decibels specified in
    FrequencyDBVariation.'''

    input_units = Units.T(optional=True, xmltagname='InputUnits')
    output_units = Units.T(optional=True, xmltagname='OutputUnits')
    frequency_start = Float.T(optional=True, xmltagname='FrequencyStart')
    frequency_end = Float.T(optional=True, xmltagname='FrequencyEnd')
    frequency_db_variation = Float.T(optional=True,
                                     xmltagname='FrequencyDBVariation')


class Coefficient(FloatNoUnit):
    number = Counter.T(optional=True, xmlstyle='attribute')


class PoleZero(Object):
    '''Complex numbers used as poles or zeros in channel response.'''

    number = Int.T(optional=True, xmlstyle='attribute')
    real = FloatNoUnit.T(xmltagname='Real')
    imaginary = FloatNoUnit.T(xmltagname='Imaginary')

    def value(self):
        return self.real.value + 1J * self.imaginary.value


class ClockDrift(FloatWithUnit):
    unit = String.T(default='SECONDS/SAMPLE', optional=True,
                    xmlstyle='attribute')  # fixed


class Second(FloatWithUnit):
    '''A time value in seconds.'''

    unit = String.T(default='SECONDS', optional=True, xmlstyle='attribute')
    # fixed unit


class Voltage(FloatWithUnit):
    unit = String.T(default='VOLTS', optional=True, xmlstyle='attribute')
    # fixed unit


class Angle(FloatWithUnit):
    unit = String.T(default='DEGREES', optional=True, xmlstyle='attribute')
    # fixed unit


class Azimuth(FloatWithUnit):
    '''Instrument azimuth, degrees clockwise from North.'''

    unit = String.T(default='DEGREES', optional=True, xmlstyle='attribute')
    # fixed unit


class Dip(FloatWithUnit):
    '''Instrument dip in degrees down from horizontal. Together
    azimuth and dip describe the direction of the sensitive axis of
    the instrument.'''

    unit = String.T(default='DEGREES', optional=True, xmlstyle='attribute')
    # fixed unit


class Distance(FloatWithUnit):
    '''Extension of FloatWithUnit for distances, elevations, and depths.'''

    unit = String.T(default='METERS', optional=True, xmlstyle='attribute')
    # NOT fixed unit!


class Frequency(FloatWithUnit):
    unit = String.T(default='HERTZ', optional=True, xmlstyle='attribute')
    # fixed unit


class SampleRate(FloatWithUnit):
    '''Sample rate in samples per second.'''

    unit = String.T(default='SAMPLES/S', optional=True, xmlstyle='attribute')
    # fixed unit


class Person(Object):
    '''Representation of a person's contact information. A person can
    belong to multiple agencies and have multiple email addresses and
    phone numbers.'''

    name_list = List.T(Unicode.T(xmltagname='Name'))
    agency_list = List.T(Unicode.T(xmltagname='Agency'))
    email_list = List.T(Email.T(xmltagname='Email'))
    phone_list = List.T(PhoneNumber.T(xmltagname='Phone'))


class FIR(BaseFilter):
    '''Response: FIR filter. Corresponds to SEED blockette 61. FIR
    filters are also commonly documented using the Coefficients
    element.'''

    symmetry = Symmetry.T(xmltagname='Symmetry')
    numerator_coefficient_list = List.T(
        NumeratorCoefficient.T(xmltagname='NumeratorCoefficient'))

    def summary(self):
        return 'fir(%i%s)' % (
            self.get_ncoefficiencs(),
            ',sym' if self.get_effective_symmetry() != 'NONE' else '')

    def get_effective_coefficients(self):
        b = num.array(
            [v.value for v in self.numerator_coefficient_list],
            dtype=num.float)

        if self.symmetry == 'ODD':
            b = num.concatenate((b, b[-2::-1]))
        elif self.symmetry == 'EVEN':
            b = num.concatenate((b, b[::-1]))

        return b

    def get_effective_symmetry(self):
        if self.symmetry == 'NONE':
            b = self.get_effective_coefficients()
            if num.all(b - b[::-1] == 0):
                return ['EVEN', 'ODD'][b.size % 2]

        return self.symmetry

    def get_ncoefficiencs(self):
        nf = len(self.numerator_coefficient_list)
        if self.symmetry == 'ODD':
            nc = nf * 2 + 1
        elif self.symmetry == 'EVEN':
            nc = nf * 2
        else:
            nc = nf

        return nc

    def estimate_delay(self, deltat):
        nc = self.get_ncoefficiencs()
        if nc > 0:
            return deltat * (nc - 1) / 2.0
        else:
            return 0.0

    def get_pyrocko_response(
            self, context, deltat, delay_responses, normalization_frequency):

        if not self.numerator_coefficient_list:
            return []

        b = self.get_effective_coefficients()

        if not deltat:
            raise NoResponseInformation(
                'cannot get digital response without knowing sampling '
                'interval (%s)' % context)

        drop_phase = self.get_effective_symmetry() != 'NONE'

        resps = []
        resp = trace.DigitalFilterResponse(
            b.tolist(), [1.0], deltat, drop_phase=drop_phase)

        if normalization_frequency is not None:
            normalization_frequency = 0.0
            normalization = num.abs(evaluate1(resp, normalization_frequency))

            if num.abs(normalization - 1.0) > 1e-2:
                logger.warn(
                    'FIR filter coefficients are not normalized. Normalizing '
                    'them. Factor: %g (%s)' % (normalization, context))

            resp = trace.DigitalFilterResponse(
                (b/normalization).tolist(), [1.0], deltat,
                drop_phase=drop_phase)

        resps.append(resp)

        if not drop_phase:
            resps.extend(delay_responses)

        return resps


class Coefficients(BaseFilter):
    '''Response: coefficients for FIR filter. Laplace transforms or
    IIR filters can be expressed using type as well but the
    PolesAndZeros should be used instead. Corresponds to SEED
    blockette 54.'''

    cf_transfer_function_type = CfTransferFunction.T(
        xmltagname='CfTransferFunctionType')
    numerator_list = List.T(FloatWithUnit.T(xmltagname='Numerator'))
    denominator_list = List.T(FloatWithUnit.T(xmltagname='Denominator'))

    def summary(self):
        return 'coef_%s(%i,%i%s)' % (
            'ABC?'[
                CfTransferFunction.choices.index(
                    self.cf_transfer_function_type)],
            len(self.numerator_list),
            len(self.denominator_list),
            ',sym' if self.is_symmetric_fir else '')

    def estimate_delay(self, deltat):
        nc = len(self.numerator_list)
        if nc > 0:
            return deltat * (len(self.numerator_list) - 1) / 2.0
        else:
            return 0.0

    def is_symmetric_fir(self):
        if len(self.denominator_list) != 0:
            return False
        b = [v.value for v in self.numerator_list]
        return b == b[::-1]

    def get_pyrocko_response(
            self, context, deltat, delay_responses, normalization_frequency):

        factor = 1.0
        if self.cf_transfer_function_type == 'ANALOG (HERTZ)':
            factor = 2.0*math.pi

        if not self.numerator_list and not self.denominator_list:
            return []

        b = num.array(
            [v.value*factor for v in self.numerator_list], dtype=num.float)

        a = num.array(
            [1.0] + [v.value*factor for v in self.denominator_list],
            dtype=num.float)

        resps = []
        if self.cf_transfer_function_type in [
                'ANALOG (RADIANS/SECOND)', 'ANALOG (HERTZ)']:
            resps.append(trace.AnalogFilterResponse(b, a))

        elif self.cf_transfer_function_type == 'DIGITAL':
            if not deltat:
                raise NoResponseInformation(
                    'cannot get digital response without knowing sampling '
                    'interval (%s)' % context)

            drop_phase = self.is_symmetric_fir()
            resp = trace.DigitalFilterResponse(
                b, a, deltat, drop_phase=drop_phase)

            if normalization_frequency is not None:
                normalization = num.abs(evaluate1(
                    resp, normalization_frequency))

                if num.abs(normalization - 1.0) > 1e-2:
                    logger.warn(
                        'FIR filter coefficients are not normalized. '
                        'Normalizing them. Factor: %g (%s)' % (
                            normalization, context))

                resp = trace.DigitalFilterResponse(
                    (b/normalization).tolist(), [1.0], deltat,
                    drop_phase=drop_phase)

            resps.append(resp)

            if not drop_phase:
                resps.extend(delay_responses)

        else:
            raise ValueError(self.cf_transfer_function_type)

        return resps


class Latitude(FloatWithUnit):
    '''Type for latitude coordinate.'''

    unit = String.T(default='DEGREES', optional=True, xmlstyle='attribute')
    # fixed unit
    datum = String.T(default='WGS84', optional=True, xmlstyle='attribute')


class Longitude(FloatWithUnit):
    '''Type for longitude coordinate.'''

    unit = String.T(default='DEGREES', optional=True, xmlstyle='attribute')
    # fixed unit
    datum = String.T(default='WGS84', optional=True, xmlstyle='attribute')


class PolesZeros(BaseFilter):
    '''Response: complex poles and zeros. Corresponds to SEED
    blockette 53.'''

    pz_transfer_function_type = PzTransferFunction.T(
        xmltagname='PzTransferFunctionType')
    normalization_factor = Float.T(default=1.0,
                                   xmltagname='NormalizationFactor')
    normalization_frequency = Frequency.T(xmltagname='NormalizationFrequency')
    zero_list = List.T(PoleZero.T(xmltagname='Zero'))
    pole_list = List.T(PoleZero.T(xmltagname='Pole'))

    def summary(self):
        return 'pz_%s(%i,%i)' % (
            'ABC?'[
                PzTransferFunction.choices.index(
                    self.pz_transfer_function_type)],
            len(self.pole_list),
            len(self.zero_list))

    def get_pyrocko_response(self, context=None, deltat=None):

        factor = 1.0
        cfactor = 1.0
        if self.pz_transfer_function_type == 'LAPLACE (HERTZ)':
            factor = 2. * math.pi
            cfactor = (2. * math.pi)**(
                len(self.pole_list) - len(self.zero_list))

        if self.normalization_factor is None \
                or self.normalization_factor == 0.0:

            logger.warn(
                'no pole-zero normalization factor given. Assuming a value of '
                '1.0 (%s)' % context)

            nfactor = 1.0
        else:
            nfactor = self.normalization_factor

        if self.pz_transfer_function_type != 'DIGITAL (Z-TRANSFORM)':
            resp = trace.PoleZeroResponse(
                constant=nfactor*cfactor,
                zeros=[z.value()*factor for z in self.zero_list],
                poles=[p.value()*factor for p in self.pole_list])
        else:
            if not deltat:
                raise NoResponseInformation(
                    'cannot get digital pz response without knowing sampling '
                    'interval (%s)' % context)

            resp = trace.DigitalPoleZeroResponse(
                constant=nfactor*cfactor,
                zeros=[z.value()*factor for z in self.zero_list],
                poles=[p.value()*factor for p in self.pole_list],
                deltat=deltat)

        if not self.normalization_frequency.value:
            logger.warn(
                'cannot check pole-zero normalization factor (%s)' % context)

        else:
            computed_normalization_factor = nfactor / abs(evaluate1(
                resp, self.normalization_frequency.value))

            db = 20.0 * num.log10(
                computed_normalization_factor / nfactor)

            if abs(db) > 0.17:
                logger.warn(
                    'computed and reported normalization factors differ by '
                    '%g dB: computed: %g, reported: %g (%s)' % (
                        db,
                        computed_normalization_factor,
                        nfactor,
                        context))

        return [resp]


class ResponseListElement(Object):
    frequency = Frequency.T(xmltagname='Frequency')
    amplitude = FloatWithUnit.T(xmltagname='Amplitude')
    phase = Angle.T(xmltagname='Phase')


class Polynomial(BaseFilter):
    '''Response: expressed as a polynomial (allows non-linear sensors
    to be described). Corresponds to SEED blockette 62. Can be used to
    describe a stage of acquisition or a complete system.'''

    approximation_type = Approximation.T(default='MACLAURIN',
                                         xmltagname='ApproximationType')
    frequency_lower_bound = Frequency.T(xmltagname='FrequencyLowerBound')
    frequency_upper_bound = Frequency.T(xmltagname='FrequencyUpperBound')
    approximation_lower_bound = Float.T(xmltagname='ApproximationLowerBound')
    approximation_upper_bound = Float.T(xmltagname='ApproximationUpperBound')
    maximum_error = Float.T(xmltagname='MaximumError')
    coefficient_list = List.T(Coefficient.T(xmltagname='Coefficient'))

    def summary(self):
        return 'poly(%i)' % len(self.coefficient_list)


class Decimation(Object):
    '''Corresponds to SEED blockette 57.'''

    input_sample_rate = Frequency.T(xmltagname='InputSampleRate')
    factor = Int.T(xmltagname='Factor')
    offset = Int.T(xmltagname='Offset')
    delay = FloatWithUnit.T(xmltagname='Delay')
    correction = FloatWithUnit.T(xmltagname='Correction')

    def summary(self):
        return 'deci(%i, %g -> %g, %g)' % (
            self.factor,
            self.input_sample_rate.value,
            self.input_sample_rate.value / self.factor,
            self.delay.value)

    def get_pyrocko_response(self):
        if self.delay and self.delay.value != 0.0:
            return [trace.DelayResponse(delay=-self.delay.value)]

        return []


class Operator(Object):
    agency_list = List.T(Unicode.T(xmltagname='Agency'))
    contact_list = List.T(Person.T(xmltagname='Contact'))
    web_site = String.T(optional=True, xmltagname='WebSite')


class Comment(Object):
    '''Container for a comment or log entry. Corresponds to SEED
    blockettes 31, 51 and 59.'''

    id = Counter.T(optional=True, xmlstyle='attribute')
    value = Unicode.T(xmltagname='Value')
    begin_effective_time = DummyAwareOptionalTimestamp.T(
        optional=True,
        xmltagname='BeginEffectiveTime')
    end_effective_time = DummyAwareOptionalTimestamp.T(
        optional=True,
        xmltagname='EndEffectiveTime')
    author_list = List.T(Person.T(xmltagname='Author'))


class ResponseList(BaseFilter):
    '''Response: list of frequency, amplitude and phase values.
    Corresponds to SEED blockette 55.'''

    response_list_element_list = List.T(
        ResponseListElement.T(xmltagname='ResponseListElement'))

    def summary(self):
        return 'list(%i)' % len(self.response_list_element_list)


class Log(Object):
    '''Container for log entries.'''

    entry_list = List.T(Comment.T(xmltagname='Entry'))


class ResponseStage(Object):
    '''This complex type represents channel response and covers SEED
    blockettes 53 to 56.'''

    number = Counter.T(xmlstyle='attribute')
    resource_id = String.T(optional=True, xmlstyle='attribute')
    poles_zeros_list = List.T(
        PolesZeros.T(optional=True, xmltagname='PolesZeros'))
    coefficients_list = List.T(
        Coefficients.T(optional=True, xmltagname='Coefficients'))
    response_list = ResponseList.T(optional=True, xmltagname='ResponseList')
    fir = FIR.T(optional=True, xmltagname='FIR')
    polynomial = Polynomial.T(optional=True, xmltagname='Polynomial')
    decimation = Decimation.T(optional=True, xmltagname='Decimation')
    stage_gain = Gain.T(optional=True, xmltagname='StageGain')

    def summary(self):
        elements = []

        for stuff in [
                self.poles_zeros_list, self.coefficients_list,
                self.response_list, self.fir, self.polynomial,
                self.decimation, self.stage_gain]:

            if stuff:
                if isinstance(stuff, Object):
                    elements.append(stuff.summary())
                else:
                    elements.extend(obj.summary() for obj in stuff)

        return '%i: %s %s -> %s' % (
            self.number,
            ', '.join(elements),
            self.input_units.name.upper() if self.input_units else '?',
            self.output_units.name.upper() if self.output_units else '?')

    def get_pyrocko_response(self, context, gain_only=False):

        context = context + ', stage %i' % self.number

        responses = []
        if self.stage_gain:
            normalization_frequency = self.stage_gain.frequency
        else:
            normalization_frequency = 0.0

        if not gain_only:
            deltat = None
            delay_responses = []
            if self.decimation:
                deltat = 1.0 / self.decimation.input_sample_rate.value
                delay_responses = self.decimation.get_pyrocko_response()

            for pzs in self.poles_zeros_list:
                pz_resps = pzs.get_pyrocko_response(context, deltat)
                responses.extend(pz_resps)

                # emulate incorrect? evalresp behaviour
                if pzs.normalization_frequency != self.stage_gain.frequency:
                    trial = trace.MultiplyResponse(pz_resps)
                    anorm = num.abs(evaluate1(
                        trial, pzs.normalization_frequency.value))
                    asens = num.abs(
                        evaluate1(trial, self.stage_gain.frequency))

                    factor = anorm/asens

                    if abs(factor - 1.0) > 0.01:
                        logger.warn(
                            'PZ normalization frequency (%g) is different '
                            'from stage gain frequency (%s) -> Emulating '
                            'possibly incorrect evalresp behaviour Correction '
                            'factor: %g (%s)' % (
                                pzs.normalization_frequency.value,
                                self.stage_gain.frequency,
                                factor,
                                context))

                        responses.append(
                            trace.PoleZeroResponse(constant=factor))

            if len(self.poles_zeros_list) > 1:
                logger.warn(
                    'multiple poles and zeros records in single response '
                    'stage (%s)' % context)

            for cfs in self.coefficients_list + (
                    [self.fir] if self.fir else []):

                responses.extend(cfs.get_pyrocko_response(
                    context, deltat, delay_responses,
                    normalization_frequency))

            if len(self.coefficients_list) > 1:
                logger.warn(
                    'multiple filter coefficients lists in single response '
                    'stage (%s)' % context)

            if (self.response_list or self.fir or self.polynomial):
                logger.debug('unhandled response at stage %i' % self.number)

        if self.stage_gain:
            responses.append(
                trace.PoleZeroResponse(constant=self.stage_gain.value))

        return responses

    @property
    def input_units(self):
        for e in (self.poles_zeros_list + self.coefficients_list +
                  [self.response_list, self.fir, self.polynomial]):
            if e is not None:
                return e.input_units

        return None

    @property
    def output_units(self):
        for e in (self.poles_zeros_list + self.coefficients_list +
                  [self.response_list, self.fir, self.polynomial]):
            if e is not None:
                return e.output_units

        return None


class Response(Object):
    resource_id = String.T(optional=True, xmlstyle='attribute')
    instrument_sensitivity = Sensitivity.T(optional=True,
                                           xmltagname='InstrumentSensitivity')
    instrument_polynomial = Polynomial.T(optional=True,
                                         xmltagname='InstrumentPolynomial')
    stage_list = List.T(ResponseStage.T(xmltagname='Stage'))

    def check_sample_rates(self, channel):

        if self.stage_list:
            sample_rate = None

            for stage in self.stage_list:
                if stage.decimation:
                    input_sample_rate = \
                        stage.decimation.input_sample_rate.value

                    if sample_rate is not None and not same_sample_rate(
                            sample_rate, input_sample_rate):

                        logger.warn(
                            'Response stage %i has unexpected input sample '
                            'rate: %g Hz (expected: %g Hz)' % (
                                stage.number,
                                input_sample_rate,
                                sample_rate))

                    sample_rate = input_sample_rate / stage.decimation.factor

            if sample_rate is not None and channel.sample_rate.value \
                    and not same_sample_rate(
                        sample_rate, channel.sample_rate.value):

                logger.warn(
                    'Channel sample rate (%g Hz) does not match sample rate '
                    'deducted from response stages information (%g Hz).' % (
                        channel.sample_rate.value,
                        sample_rate))

    def check_units(self):

        if self.instrument_sensitivity \
                and self.instrument_sensitivity.input_units:

            units = self.instrument_sensitivity.input_units.name.upper()

        if self.stage_list:
            for stage in self.stage_list:
                if units and stage.input_units \
                        and stage.input_units.name.upper() != units:

                    logger.warn(
                        'Input units of stage %i (%s) do not match %s (%s).'
                        % (
                            stage.number,
                            units,
                            'output units of stage %i'
                            if stage.number == 0
                            else 'sensitivity input units',
                            units))

                if stage.output_units:
                    units = stage.output_units.name.upper()
                else:
                    units = None

            sout_units = self.instrument_sensitivity.output_units
            if self.instrument_sensitivity and sout_units:
                if units is not None and units != sout_units.name.upper():
                    logger.warn(
                        'Output units of stage %i (%s) do not match %s (%s).'
                        % (
                            stage.number,
                            units,
                            'sensitivity output units',
                            sout_units.name.upper()))

    def get_pyrocko_response(
            self, context, fake_input_units=None, stages=(0, 1)):

        if self.stage_list:
            responses = []
            for istage, stage in enumerate(self.stage_list):
                responses.extend(stage.get_pyrocko_response(
                    context, gain_only=not (stages[0] <= istage < stages[1])))

        elif self.instrument_sensitivity:
            responses = [trace.PoleZeroResponse(
                constant=self.instrument_sensitivity.value)]
        else:
            responses = []

        checkpoints = []
        if self.instrument_sensitivity:
            trial = trace.MultiplyResponse(responses)
            sval = self.instrument_sensitivity.value
            sfreq = self.instrument_sensitivity.frequency
            checkpoints.append(trace.FrequencyResponseCheckpoint(
                frequency=sfreq, value=sval))

            try:
                check_resp(
                    trial, sval, sfreq, 0.1,
                    prelude='Instrument sensitivity value inconsistent with '
                            'sensitivity computed from complete response (%s)'
                            % context)
            except InconsistentResponseInformation as e:
                logger.warn(str(e))

        if fake_input_units is not None:
            if not self.instrument_sensitivity or \
                    self.instrument_sensitivity.input_units is None:

                raise NoResponseInformation('no input units given')

            input_units = self.instrument_sensitivity.input_units.name.upper()

            try:
                conresp = conversion[
                    fake_input_units.upper(), input_units]

            except KeyError:
                raise NoResponseInformation(
                    'cannot convert between units: %s, %s'
                    % (fake_input_units, input_units))

            if conresp is not None:
                responses.append(conresp)
                for checkpoint in checkpoints:
                    checkpoint.value *= num.abs(evaluate1(
                        conresp, checkpoint.frequency))

        return trace.MultiplyResponse(responses, checkpoints=checkpoints)

    @classmethod
    def from_pyrocko_pz_response(cls, presponse, input_unit, output_unit,
                                 normalization_frequency=1.0):
        '''
        Convert Pyrocko pole-zero response to StationXML response.

        :param presponse: Pyrocko pole-zero response
        :type presponse: :py:class:`~pyrocko.trace.PoleZeroResponse`
        :param input_unit: Input unit to be reported in the StationXML
            response.
        :type input_unit: str
        :param output_unit: Output unit to be reported in the StationXML
            response.
        :type output_unit: str
        :param normalization_frequency: Frequency where the normalization
            factor for the StationXML response should be computed.
        :type normalization_frequency: float
        '''

        norm_factor = 1.0/float(abs(
            evaluate1(presponse, normalization_frequency)
            / presponse.constant))

        pzs = PolesZeros(
            pz_transfer_function_type='LAPLACE (RADIANS/SECOND)',
            normalization_factor=norm_factor,
            normalization_frequency=Frequency(normalization_frequency),
            zero_list=[PoleZero(real=FloatNoUnit(z.real),
                                imaginary=FloatNoUnit(z.imag))
                       for z in presponse.zeros],
            pole_list=[PoleZero(real=FloatNoUnit(z.real),
                                imaginary=FloatNoUnit(z.imag))
                       for z in presponse.poles])

        pzs.validate()

        stage = ResponseStage(
            number=1,
            poles_zeros_list=[pzs],
            stage_gain=Gain(float(abs(presponse.constant))/norm_factor))

        resp = Response(
            instrument_sensitivity=Sensitivity(
                value=stage.stage_gain.value,
                input_units=Units(input_unit),
                output_units=Units(output_unit)),

            stage_list=[stage])

        return resp


class BaseNode(Object):
    '''A base node type for derivation from: Network, Station and
    Channel types.'''

    code = String.T(xmlstyle='attribute')
    start_date = DummyAwareOptionalTimestamp.T(optional=True,
                                               xmlstyle='attribute')
    end_date = DummyAwareOptionalTimestamp.T(optional=True,
                                             xmlstyle='attribute')
    restricted_status = RestrictedStatus.T(optional=True, xmlstyle='attribute')
    alternate_code = String.T(optional=True, xmlstyle='attribute')
    historical_code = String.T(optional=True, xmlstyle='attribute')
    description = Unicode.T(optional=True, xmltagname='Description')
    comment_list = List.T(Comment.T(xmltagname='Comment'))

    def spans(self, *args):
        if len(args) == 0:
            return True
        elif len(args) == 1:
            return ((self.start_date is None or
                     self.start_date <= args[0]) and
                    (self.end_date is None or
                     args[0] <= self.end_date))

        elif len(args) == 2:
            return ((self.start_date is None or
                     args[1] >= self.start_date) and
                    (self.end_date is None or
                     self.end_date >= args[0]))


def overlaps(a, b):
    return (
        a.start_date is None or b.end_date is None
        or a.start_date < b.end_date
    ) and (
        b.start_date is None or a.end_date is None
        or b.start_date < a.end_date
    )


class Channel(BaseNode):
    '''Equivalent to SEED blockette 52 and parent element for the
    related the response blockettes.'''

    location_code = String.T(xmlstyle='attribute')
    external_reference_list = List.T(
        ExternalReference.T(xmltagname='ExternalReference'))
    latitude = Latitude.T(xmltagname='Latitude')
    longitude = Longitude.T(xmltagname='Longitude')
    elevation = Distance.T(xmltagname='Elevation')
    depth = Distance.T(xmltagname='Depth')
    azimuth = Azimuth.T(optional=True, xmltagname='Azimuth')
    dip = Dip.T(optional=True, xmltagname='Dip')
    type_list = List.T(Type.T(xmltagname='Type'))
    sample_rate = SampleRate.T(optional=True, xmltagname='SampleRate')
    sample_rate_ratio = SampleRateRatio.T(optional=True,
                                          xmltagname='SampleRateRatio')
    storage_format = String.T(optional=True, xmltagname='StorageFormat')
    clock_drift = ClockDrift.T(optional=True, xmltagname='ClockDrift')
    calibration_units = Units.T(optional=True, xmltagname='CalibrationUnits')
    sensor = Equipment.T(optional=True, xmltagname='Sensor')
    pre_amplifier = Equipment.T(optional=True, xmltagname='PreAmplifier')
    data_logger = Equipment.T(optional=True, xmltagname='DataLogger')
    equipment = Equipment.T(optional=True, xmltagname='Equipment')
    response = Response.T(optional=True, xmltagname='Response')

    @property
    def position_values(self):
        lat = self.latitude.value
        lon = self.longitude.value
        elevation = value_or_none(self.elevation)
        depth = value_or_none(self.depth)
        return lat, lon, elevation, depth


class Station(BaseNode):
    '''This type represents a Station epoch. It is common to only have
    a single station epoch with the station's creation and termination
    dates as the epoch start and end dates.'''

    latitude = Latitude.T(xmltagname='Latitude')
    longitude = Longitude.T(xmltagname='Longitude')
    elevation = Distance.T(xmltagname='Elevation')
    site = Site.T(optional=True, xmltagname='Site')
    vault = Unicode.T(optional=True, xmltagname='Vault')
    geology = Unicode.T(optional=True, xmltagname='Geology')
    equipment_list = List.T(Equipment.T(xmltagname='Equipment'))
    operator_list = List.T(Operator.T(xmltagname='Operator'))
    creation_date = DummyAwareOptionalTimestamp.T(
        optional=True, xmltagname='CreationDate')
    termination_date = DummyAwareOptionalTimestamp.T(
        optional=True, xmltagname='TerminationDate')
    total_number_channels = Counter.T(
        optional=True, xmltagname='TotalNumberChannels')
    selected_number_channels = Counter.T(
        optional=True, xmltagname='SelectedNumberChannels')
    external_reference_list = List.T(
        ExternalReference.T(xmltagname='ExternalReference'))
    channel_list = List.T(Channel.T(xmltagname='Channel'))

    @property
    def position_values(self):
        lat = self.latitude.value
        lon = self.longitude.value
        elevation = value_or_none(self.elevation)
        return lat, lon, elevation


class Network(BaseNode):
    '''This type represents the Network layer, all station metadata is
    contained within this element. The official name of the network or
    other descriptive information can be included in the Description
    element. The Network can contain 0 or more Stations.'''

    total_number_stations = Counter.T(optional=True,
                                      xmltagname='TotalNumberStations')
    selected_number_stations = Counter.T(optional=True,
                                         xmltagname='SelectedNumberStations')
    station_list = List.T(Station.T(xmltagname='Station'))

    @property
    def station_code_list(self):
        return sorted(set(s.code for s in self.station_list))

    @property
    def sl_code_list(self):
        sls = set()
        for station in self.station_list:
            for channel in station.channel_list:
                sls.add((station.code, channel.location_code))

        return sorted(sls)

    def summary(self, width=80, indent=4):
        sls = self.sl_code_list or [(x,) for x in self.station_code_list]
        lines = ['%s (%i):' % (self.code, len(sls))]
        if sls:
            ssls = ['.'.join(x for x in c if x) for c in sls]
            w = max(len(x) for x in ssls)
            n = (width - indent) / (w+1)
            while ssls:
                lines.append(
                    ' ' * indent + ' '.join(x.ljust(w) for x in ssls[:n]))

                ssls[:n] = []

        return '\n'.join(lines)


def value_or_none(x):
    if x is not None:
        return x.value
    else:
        return None


def pyrocko_station_from_channels(nsl, channels, inconsistencies='warn'):

    pos = lat, lon, elevation, depth = \
        channels[0].position_values

    if not all(pos == x.position_values for x in channels):
        info = '\n'.join(
            '    %s: %s' % (x.code, x.position_values) for
            x in channels)

        mess = 'encountered inconsistencies in channel ' \
               'lat/lon/elevation/depth ' \
               'for %s.%s.%s: \n%s' % (nsl + (info,))

        if inconsistencies == 'raise':
            raise InconsistentChannelLocations(mess)

        elif inconsistencies == 'warn':
            logger.warn(mess)
            logger.warn(' -> using mean values')

    apos = num.array([x.position_values for x in channels], dtype=num.float)
    mlat, mlon, mele, mdep = num.nansum(apos, axis=0) \
        / num.sum(num.isfinite(apos), axis=0)

    groups = {}
    for channel in channels:
        if channel.code not in groups:
            groups[channel.code] = []

        groups[channel.code].append(channel)

    pchannels = []
    for code in sorted(groups.keys()):
        data = [
            (channel.code, value_or_none(channel.azimuth),
                value_or_none(channel.dip))
            for channel in groups[code]]

        azimuth, dip = util.consistency_merge(
            data,
            message='channel orientation values differ:',
            error=inconsistencies)

        pchannels.append(
            pyrocko.model.Channel(code, azimuth=azimuth, dip=dip))

    return pyrocko.model.Station(
        *nsl,
        lat=mlat,
        lon=mlon,
        elevation=mele,
        depth=mdep,
        channels=pchannels)


class FDSNStationXML(Object):
    '''Top-level type for Station XML. Required field are Source
    (network ID of the institution sending the message) and one or
    more Network containers or one or more Station containers.'''

    schema_version = Float.T(default=1.0, xmlstyle='attribute')
    source = String.T(xmltagname='Source')
    sender = String.T(optional=True, xmltagname='Sender')
    module = String.T(optional=True, xmltagname='Module')
    module_uri = String.T(optional=True, xmltagname='ModuleURI')
    created = Timestamp.T(optional=True, xmltagname='Created')
    network_list = List.T(Network.T(xmltagname='Network'))

    xmltagname = 'FDSNStationXML'
    guessable_xmlns = [guts_xmlns]

    def get_pyrocko_stations(self, nslcs=None, nsls=None,
                             time=None, timespan=None,
                             inconsistencies='warn'):

        assert inconsistencies in ('raise', 'warn')

        if nslcs is not None:
            nslcs = set(nslcs)

        if nsls is not None:
            nsls = set(nsls)

        tt = ()
        if time is not None:
            tt = (time,)
        elif timespan is not None:
            tt = timespan

        pstations = []
        for network in self.network_list:
            if not network.spans(*tt):
                continue

            for station in network.station_list:
                if not station.spans(*tt):
                    continue

                if station.channel_list:
                    loc_to_channels = {}
                    for channel in station.channel_list:
                        if not channel.spans(*tt):
                            continue

                        loc = channel.location_code.strip()
                        if loc not in loc_to_channels:
                            loc_to_channels[loc] = []

                        loc_to_channels[loc].append(channel)

                    for loc in sorted(loc_to_channels.keys()):
                        channels = loc_to_channels[loc]
                        if nslcs is not None:
                            channels = [channel for channel in channels
                                        if (network.code, station.code, loc,
                                            channel.code) in nslcs]

                        if not channels:
                            continue

                        nsl = network.code, station.code, loc
                        if nsls is not None and nsl not in nsls:
                            continue

                        pstations.append(
                            pyrocko_station_from_channels(
                                nsl,
                                channels,
                                inconsistencies=inconsistencies))
                else:
                    pstations.append(pyrocko.model.Station(
                        network.code, station.code, '*',
                        lat=station.latitude.value,
                        lon=station.longitude.value,
                        elevation=value_or_none(station.elevation),
                        name=station.description or ''))

        return pstations

    @classmethod
    def from_pyrocko_stations(
            cls, pyrocko_stations, add_flat_responses_from=None):

        ''' Generate :py:class:`FDSNStationXML` from list of
        :py:class;`pyrocko.model.Station` instances.

        :param pyrocko_stations: list of :py:class;`pyrocko.model.Station`
            instances.
        :param add_flat_responses_from: unit, 'M', 'M/S' or 'M/S**2'
        '''
        from collections import defaultdict
        network_dict = defaultdict(list)

        if add_flat_responses_from:
            assert add_flat_responses_from in ('M', 'M/S', 'M/S**2')
            extra = dict(
                response=Response(
                    instrument_sensitivity=Sensitivity(
                        value=1.0,
                        frequency=1.0,
                        input_units=Units(name=add_flat_responses_from))))
        else:
            extra = {}

        have_offsets = set()
        for s in pyrocko_stations:

            if s.north_shift != 0.0 or s.east_shift != 0.0:
                have_offsets.add(s.nsl())

            network, station, location = s.nsl()
            channel_list = []
            for c in s.channels:
                channel_list.append(
                    Channel(
                        location_code=location,
                        code=c.name,
                        latitude=Latitude(value=s.effective_lat),
                        longitude=Longitude(value=s.effective_lon),
                        elevation=Distance(value=s.elevation),
                        depth=Distance(value=s.depth),
                        azimuth=Azimuth(value=c.azimuth),
                        dip=Dip(value=c.dip),
                        **extra
                    )
                )

            network_dict[network].append(
                Station(
                    code=station,
                    latitude=Latitude(value=s.effective_lat),
                    longitude=Longitude(value=s.effective_lon),
                    elevation=Distance(value=s.elevation),
                    channel_list=channel_list)
            )

        if have_offsets:
            logger.warn(
                'StationXML does not support Cartesian offsets in '
                'coordinates. Storing effective lat/lon for stations: %s' %
                ', '.join('.'.join(nsl) for nsl in sorted(have_offsets)))

        timestamp = time.time()
        network_list = []
        for k, station_list in network_dict.items():

            network_list.append(
                Network(
                    code=k, station_list=station_list,
                    total_number_stations=len(station_list)))

        sxml = FDSNStationXML(
            source='from pyrocko stations list', created=timestamp,
            network_list=network_list)

        sxml.validate()
        return sxml

    def iter_network_stations(
            self, net=None, sta=None, time=None, timespan=None):

        tt = ()
        if time is not None:
            tt = (time,)
        elif timespan is not None:
            tt = timespan

        for network in self.network_list:
            if not network.spans(*tt) or (
                    net is not None and network.code != net):
                continue

            for station in network.station_list:
                if not station.spans(*tt) or (
                        sta is not None and station.code != sta):
                    continue

                yield (network, station)

    def iter_network_station_channels(
            self, net=None, sta=None, loc=None, cha=None,
            time=None, timespan=None):

        if loc is not None:
            loc = loc.strip()

        tt = ()
        if time is not None:
            tt = (time,)
        elif timespan is not None:
            tt = timespan

        for network in self.network_list:
            if not network.spans(*tt) or (
                    net is not None and network.code != net):
                continue

            for station in network.station_list:
                if not station.spans(*tt) or (
                        sta is not None and station.code != sta):
                    continue

                if station.channel_list:
                    for channel in station.channel_list:
                        if (not channel.spans(*tt) or
                                (cha is not None and channel.code != cha) or
                                (loc is not None and
                                 channel.location_code.strip() != loc)):
                            continue

                        yield (network, station, channel)

    def get_channel_groups(self, net=None, sta=None, loc=None, cha=None,
                           time=None, timespan=None):

        groups = {}
        for network, station, channel in self.iter_network_station_channels(
                net, sta, loc, cha, time=time, timespan=timespan):

            net = network.code
            sta = station.code
            cha = channel.code
            loc = channel.location_code.strip()
            if len(cha) == 3:
                bic = cha[:2]  # band and intrument code according to SEED
            elif len(cha) == 1:
                bic = ''
            else:
                bic = cha

            if channel.response and \
                    channel.response.instrument_sensitivity and \
                    channel.response.instrument_sensitivity.input_units:

                unit = channel.response.instrument_sensitivity\
                    .input_units.name.upper()
            else:
                unit = None

            bic = (bic, unit)

            k = net, sta, loc
            if k not in groups:
                groups[k] = {}

            if bic not in groups[k]:
                groups[k][bic] = []

            groups[k][bic].append(channel)

        for nsl, bic_to_channels in groups.items():
            bad_bics = []
            for bic, channels in bic_to_channels.items():
                sample_rates = []
                for channel in channels:
                    sample_rates.append(channel.sample_rate.value)

                if not same(sample_rates):
                    scs = ','.join(channel.code for channel in channels)
                    srs = ', '.join('%e' % x for x in sample_rates)
                    err = 'ignoring channels with inconsistent sampling ' + \
                          'rates (%s.%s.%s.%s: %s)' % (nsl + (scs, srs))

                    logger.warn(err)
                    bad_bics.append(bic)

            for bic in bad_bics:
                del bic_to_channels[bic]

        return groups

    def choose_channels(
            self,
            target_sample_rate=None,
            priority_band_code=['H', 'B', 'M', 'L', 'V', 'E', 'S'],
            priority_units=['M/S', 'M/S**2'],
            priority_instrument_code=['H', 'L'],
            time=None,
            timespan=None):

        nslcs = {}
        for nsl, bic_to_channels in self.get_channel_groups(
                time=time, timespan=timespan).items():

            useful_bics = []
            for bic, channels in bic_to_channels.items():
                rate = channels[0].sample_rate.value

                if target_sample_rate is not None and \
                        rate < target_sample_rate*0.99999:
                    continue

                if len(bic[0]) == 2:
                    if bic[0][0] not in priority_band_code:
                        continue

                    if bic[0][1] not in priority_instrument_code:
                        continue

                unit = bic[1]

                prio_unit = len(priority_units)
                try:
                    prio_unit = priority_units.index(unit)
                except ValueError:
                    pass

                prio_inst = len(priority_instrument_code)
                prio_band = len(priority_band_code)
                if len(channels[0].code) == 3:
                    try:
                        prio_inst = priority_instrument_code.index(
                            channels[0].code[1])
                    except ValueError:
                        pass

                    try:
                        prio_band = priority_band_code.index(
                            channels[0].code[0])
                    except ValueError:
                        pass

                if target_sample_rate is None:
                    rate = -rate

                useful_bics.append((-len(channels), prio_band, rate, prio_unit,
                                    prio_inst, bic))

            useful_bics.sort()

            for _, _, rate, _, _, bic in useful_bics:
                channels = sorted(
                    bic_to_channels[bic],
                    key=lambda channel: channel.code)

                if channels:
                    for channel in channels:
                        nslcs[nsl + (channel.code,)] = channel

                    break

        return nslcs

    def get_pyrocko_response(
            self, nslc,
            time=None, timespan=None, fake_input_units=None, stages=(0, 1)):

        net, sta, loc, cha = nslc
        resps = []
        for _, _, channel in self.iter_network_station_channels(
                net, sta, loc, cha, time=time, timespan=timespan):
            resp = channel.response
            if resp:
                resp.check_sample_rates(channel)
                resp.check_units()
                resps.append(resp.get_pyrocko_response(
                    '.'.join(nslc),
                    fake_input_units=fake_input_units,
                    stages=stages))

        if not resps:
            raise NoResponseInformation('%s.%s.%s.%s' % nslc)
        elif len(resps) > 1:
            raise MultipleResponseInformation('%s.%s.%s.%s' % nslc)

        return resps[0]

    @property
    def n_code_list(self):
        return sorted(set(x.code for x in self.network_list))

    @property
    def ns_code_list(self):
        nss = set()
        for network in self.network_list:
            for station in network.station_list:
                nss.add((network.code, station.code))

        return sorted(nss)

    @property
    def nsl_code_list(self):
        nsls = set()
        for network in self.network_list:
            for station in network.station_list:
                for channel in station.channel_list:
                    nsls.add(
                        (network.code, station.code, channel.location_code))

        return sorted(nsls)

    @property
    def nslc_code_list(self):
        nslcs = set()
        for network in self.network_list:
            for station in network.station_list:
                for channel in station.channel_list:
                    nslcs.add(
                        (network.code, station.code, channel.location_code,
                            channel.code))

        return sorted(nslcs)

    def summary(self):
        lst = [
            'number of n codes: %i' % len(self.n_code_list),
            'number of ns codes: %i' % len(self.ns_code_list),
            'number of nsl codes: %i' % len(self.nsl_code_list),
            'number of nslc codes: %i' % len(self.nslc_code_list)
        ]
        return '\n'.join(lst)

    def summary_stages(self):
        data = []
        for network, station, channel in self.iter_network_station_channels():
            nslc = (network.code, station.code, channel.location_code,
                    channel.code)

            stages = []
            in_units = '?'
            out_units = '?'
            if channel.response:
                sens = channel.response.instrument_sensitivity
                if sens:
                    in_units = sens.input_units.name.upper()
                    out_units = sens.output_units.name.upper()

                for stage in channel.response.stage_list:
                    stages.append(stage.summary())

            data.append(
                (nslc, tts(channel.start_date), tts(channel.end_date),
                 in_units, out_units, stages))

        data.sort()

        lst = []
        for nslc, stmin, stmax, in_units, out_units, stages in data:
            lst.append(' %s: %s - %s, %s -> %s' % (
                '.'.join(nslc), stmin, stmax, in_units, out_units))
            for stage in stages:
                lst.append('   %s' % stage)

        return '\n'.join(lst)

    def _check_overlaps(self):
        by_nslc = {}
        for network in self.network_list:
            for station in network.station_list:
                for channel in station.channel_list:
                    nslc = (network.code, station.code, channel.location_code,
                            channel.code)
                    if nslc not in by_nslc:
                        by_nslc[nslc] = []

                    by_nslc[nslc].append(channel)

        errors = []
        for nslc, channels in by_nslc.items():
            for ia, a in enumerate(channels):
                for b in channels[ia+1:]:
                    if overlaps(a, b):
                        errors.append(
                            'Channel epochs overlap for %s:\n'
                            '    %s - %s\n    %s - %s' % (
                                '.'.join(nslc),
                                tts(a.start_date), tts(a.end_date),
                                tts(b.start_date), tts(b.end_date)))
        return errors

    def check(self):
        errors = []
        for meth in [self._check_overlaps]:
            errors.extend(meth())

        if errors:
            raise Inconsistencies(
                'Inconsistencies found in StationXML:\n  '
                + '\n  '.join(errors))


class InconsistentChannelLocations(Exception):
    pass


class InvalidRecord(Exception):
    def __init__(self, line):
        Exception.__init__(self)
        self._line = line

    def __str__(self):
        return 'Invalid record: "%s"' % self._line


def load_channel_table(stream):

    networks = {}
    stations = {}

    for line in stream:
        line = str(line.decode('ascii'))
        if line.startswith('#'):
            continue

        t = line.rstrip().split('|')

        if len(t) != 17:
            logger.warn('Invalid channel record: %s' % line)
            continue

        (net, sta, loc, cha, lat, lon, ele, dep, azi, dip, sens, scale,
            scale_freq, scale_units, sample_rate, start_date, end_date) = t

        try:
            scale = float(scale)
        except ValueError:
            scale = None

        try:
            scale_freq = float(scale_freq)
        except ValueError:
            scale_freq = None

        try:
            depth = float(dep)
        except ValueError:
            depth = 0.0

        try:
            azi = float(azi)
            dip = float(dip)
        except ValueError:
            azi = None
            dip = None

        try:
            if net not in networks:
                network = Network(code=net)
            else:
                network = networks[net]

            if (net, sta) not in stations:
                station = Station(
                    code=sta, latitude=lat, longitude=lon, elevation=ele)

                station.regularize()
            else:
                station = stations[net, sta]

            if scale:
                resp = Response(
                    instrument_sensitivity=Sensitivity(
                        value=scale,
                        frequency=scale_freq,
                        input_units=scale_units))
            else:
                resp = None

            channel = Channel(
                code=cha,
                location_code=loc.strip(),
                latitude=lat,
                longitude=lon,
                elevation=ele,
                depth=depth,
                azimuth=azi,
                dip=dip,
                sensor=Equipment(description=sens),
                response=resp,
                sample_rate=sample_rate,
                start_date=start_date,
                end_date=end_date or None)

            channel.regularize()

        except ValidationError:
            raise InvalidRecord(line)

        if net not in networks:
            networks[net] = network

        if (net, sta) not in stations:
            stations[net, sta] = station
            network.station_list.append(station)

        station.channel_list.append(channel)

    return FDSNStationXML(
        source='created from table input',
        created=time.time(),
        network_list=sorted(networks.values(), key=lambda x: x.code))


def primitive_merge(sxs):
    networks = []
    for sx in sxs:
        networks.extend(sx.network_list)

    return FDSNStationXML(
        source='merged from different sources',
        created=time.time(),
        network_list=copy.deepcopy(
            sorted(networks, key=lambda x: x.code)))


if __name__ == '__main__':
    from optparse import OptionParser

    util.setup_logging('pyrocko.io.stationxml', 'warning')

    usage = \
        'python -m pyrocko.io.stationxml check|stats|stages ' \
        '<filename> [options]'

    description = '''Torture StationXML file.'''

    parser = OptionParser(
        usage=usage,
        description=description,
        formatter=util.BetterHelpFormatter())

    (options, args) = parser.parse_args(sys.argv[1:])

    if len(args) != 2:
        parser.print_help()
        sys.exit(1)

    action, path = args

    sx = load_xml(filename=path)
    if action == 'check':
        try:
            sx.check()
        except Inconsistencies as e:
            logger.error(e)
            sys.exit(1)

    elif action == 'stats':
        print(sx.summary())

    elif action == 'stages':
        print(sx.summary_stages())

    else:
        parser.print_help()
        sys.exit('unknown action: %s' % action)
