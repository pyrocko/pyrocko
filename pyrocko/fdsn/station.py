from guts import *

class Nominal(StringChoice):
    choices = [
        'NOMINAL',
        'CALCULATED' ]


class Email(UnicodePattern):
    pattern = u'[\\w\\.\\-_]+@[\\w\\.\\-_]+'


class RestrictedStatus(StringChoice):
    choices = [
        'open',
        'closed',
        'partial' ]


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
        'BEAM' ]


class PzTransferFunction(StringChoice):
    choices = [
        'LAPLACE (RADIANS/SECOND)',
        'LAPLACE (HERTZ)',
        'DIGITAL (Z-TRANSFORM)' ]


class Symmetry(StringChoice):
    choices = [
        'NONE',
        'EVEN',
        'ODD' ]


class CfTransferFunction(StringChoice):
    choices = [
        'ANALOG (RADIANS/SECOND)',
        'ANALOG (HERTZ)',
        'DIGITAL' ]


class Approximation(StringChoice):
    choices = [
        'MACLAURIN' ]


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

    value = Float.T(optional=True, xmltagname='Value')
    frequency = Float.T(optional=True, xmltagname='Frequency')


class NumeratorCoefficient(Object):
    i = Int.T(optional=True, xmlstyle='attribute')
    value = Float.T(xmlstyle='content')


class FloatNoUnit(Object):
    plus_error = Float.T(optional=True, xmlstyle='attribute')
    minus_error = Float.T(optional=True, xmlstyle='attribute')
    value = Float.T(xmlstyle='content')


class FloatWithUnit(FloatNoUnit):
    unit = String.T(optional=True, xmlstyle='attribute')
    value = Float.T(xmlstyle='content')


class Equipment(Object):
    resource_id = String.T(optional=True, xmlstyle='attribute')
    type = String.T(optional=True, xmltagname='Type')
    description = Unicode.T(optional=True, xmltagname='Description')
    manufacturer = Unicode.T(optional=True, xmltagname='Manufacturer')
    vendor = Unicode.T(optional=True, xmltagname='Vendor')
    model = Unicode.T(optional=True, xmltagname='Model')
    serial_number = String.T(optional=True, xmltagname='SerialNumber')
    installation_date = Timestamp.T(optional=True, xmltagname='InstallationDate')
    removal_date = Timestamp.T(optional=True, xmltagname='RemovalDate')
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
    frequency_db_variation = Float.T(optional=True, xmltagname='FrequencyDBVariation')


class Coefficient(FloatNoUnit):
    number = Counter.T(optional=True, xmlstyle='attribute')


class PoleZero(Object):
    '''Complex numbers used as poles or zeros in channel response.'''

    number = Int.T(optional=True, xmlstyle='attribute')
    real = FloatNoUnit.T(xmltagname='Real')
    imaginary = FloatNoUnit.T(xmltagname='Imaginary')


class ClockDrift(FloatWithUnit):
    unit = String.T(default='SECONDS/SAMPLE', optional=True, xmlstyle='attribute') # fixed


class Second(FloatWithUnit):
    '''A time value in seconds.'''

    unit = String.T(default='SECONDS', optional=True, xmlstyle='attribute') # fixed


class Voltage(FloatWithUnit):
    unit = String.T(default='VOLTS', optional=True, xmlstyle='attribute') # fixed


class Angle(FloatWithUnit):
    unit = String.T(default='DEGREES', optional=True, xmlstyle='attribute') # fixed


class Azimuth(FloatWithUnit):
    '''Instrument azimuth, degrees clockwise from North.'''

    unit = String.T(default='DEGREES', optional=True, xmlstyle='attribute') # fixed


class Dip(FloatWithUnit):
    '''Instrument dip in degrees down from horizontal. Together
    azimuth and dip describe the direction of the sensitive axis of
    the instrument.'''

    unit = String.T(default='DEGREES', optional=True, xmlstyle='attribute') # fixed


class Distance(FloatWithUnit):
    '''Extension of FloatWithUnit for distances, elevations, and depths.'''

    unit = String.T(default='METERS', optional=True, xmlstyle='attribute') # NOT fixed!


class Frequency(FloatWithUnit):
    unit = String.T(default='HERTZ', optional=True, xmlstyle='attribute') # fixed


class SampleRate(FloatWithUnit):
    '''Sample rate in samples per second.'''

    unit = String.T(default='SAMPLES/S', optional=True, xmlstyle='attribute') # fixed


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
    numerator_coefficient_list = List.T(NumeratorCoefficient.T(xmltagname='NumeratorCoefficient'))


class Coefficients(BaseFilter):
    '''Response: coefficients for FIR filter. Laplace transforms or
    IIR filters can be expressed using type as well but the
    PolesAndZeros should be used instead. Corresponds to SEED
    blockette 54.'''

    cf_transfer_function_type = CfTransferFunction.T(xmltagname='CfTransferFunctionType')
    numerator_list = List.T(FloatWithUnit.T(xmltagname='Numerator'))
    denominator_list = List.T(FloatWithUnit.T(xmltagname='Denominator'))


class Latitude(FloatWithUnit):
    '''Type for latitude coordinate.'''

    unit = String.T(default='DEGREES', optional=True, xmlstyle='attribute') # fixed
    datum = String.T(default='WGS84', optional=True, xmlstyle='attribute')


class Longitude(FloatWithUnit):
    '''Type for longitude coordinate.'''

    unit = String.T(default='DEGREES', optional=True, xmlstyle='attribute') # fixed
    datum = String.T(default='WGS84', optional=True, xmlstyle='attribute')


class PolesZeros(BaseFilter):
    '''Response: complex poles and zeros. Corresponds to SEED
    blockette 53.'''

    pz_transfer_function_type = PzTransferFunction.T(xmltagname='PzTransferFunctionType')
    normalization_factor = Float.T(default=1.0, xmltagname='NormalizationFactor')
    normalization_frequency = Frequency.T(xmltagname='NormalizationFrequency')
    zero_list = List.T(PoleZero.T(xmltagname='Zero'))
    pole_list = List.T(PoleZero.T(xmltagname='Pole'))


class ResponseListElement(Object):
    frequency = Frequency.T(xmltagname='Frequency')
    amplitude = FloatWithUnit.T(xmltagname='Amplitude')
    phase = Angle.T(xmltagname='Phase')


class Polynomial(BaseFilter):
    '''Response: expressed as a polynomial (allows non-linear sensors
    to be described). Corresponds to SEED blockette 62. Can be used to
    describe a stage of acquisition or a complete system.'''

    approximation_type = Approximation.T(default='MACLAURIN', xmltagname='ApproximationType')
    frequency_lower_bound = Frequency.T(xmltagname='FrequencyLowerBound')
    frequency_upper_bound = Frequency.T(xmltagname='FrequencyUpperBound')
    approximation_lower_bound = Float.T(xmltagname='ApproximationLowerBound')
    approximation_upper_bound = Float.T(xmltagname='ApproximationUpperBound')
    maximum_error = Float.T(xmltagname='MaximumError')
    coefficient_list = List.T(Coefficient.T(xmltagname='Coefficient'))


class Decimation(Object):
    '''Corresponds to SEED blockette 57.'''

    input_sample_rate = Frequency.T(xmltagname='InputSampleRate')
    factor = Int.T(xmltagname='Factor')
    offset = Int.T(xmltagname='Offset')
    delay = FloatWithUnit.T(xmltagname='Delay')
    correction = FloatWithUnit.T(xmltagname='Correction')


class Operator(Object):
    agency_list = List.T(Unicode.T(xmltagname='Agency'))
    contact_list = List.T(Person.T(xmltagname='Contact'))
    web_site = String.T(optional=True, xmltagname='WebSite')


class Comment(Object):
    '''Container for a comment or log entry. Corresponds to SEED
    blockettes 31, 51 and 59.'''

    id = Counter.T(optional=True, xmlstyle='attribute')
    value = Unicode.T(xmltagname='Value')
    begin_effective_time = Timestamp.T(optional=True, xmltagname='BeginEffectiveTime')
    end_effective_time = Timestamp.T(optional=True, xmltagname='EndEffectiveTime')
    author_list = List.T(Person.T(xmltagname='Author'))


class ResponseList(BaseFilter):
    '''Response: list of frequency, amplitude and phase values.
    Corresponds to SEED blockette 55.'''

    response_list_element_list = List.T(ResponseListElement.T(xmltagname='ResponseListElement'))


class Log(Object):
    '''Container for log entries.'''

    entry_list = List.T(Comment.T(xmltagname='Entry'))


class ResponseStage(Object):
    '''This complex type represents channel response and covers SEED
    blockettes 53 to 56.'''

    number = Counter.T(xmlstyle='attribute')
    resource_id = String.T(optional=True, xmlstyle='attribute')
    poles_zeros = PolesZeros.T(optional=True, xmltagname='PolesZeros')
    coefficients = Coefficients.T(optional=True, xmltagname='Coefficients')
    response_list = ResponseList.T(optional=True, xmltagname='ResponseList')
    fir = FIR.T(optional=True, xmltagname='FIR')
    polynomial = Polynomial.T(optional=True, xmltagname='Polynomial')
    decimation = Decimation.T(optional=True, xmltagname='Decimation')
    stage_gain = Gain.T(optional=True, xmltagname='StageGain')


class Response(Object):
    resource_id = String.T(optional=True, xmlstyle='attribute')
    instrument_sensitivity = Sensitivity.T(optional=True, xmltagname='InstrumentSensitivity')
    instrument_polynomial = Polynomial.T(optional=True, xmltagname='InstrumentPolynomial')
    stage_list = List.T(ResponseStage.T(xmltagname='Stage'))


class BaseNode(Object):
    '''A base node type for derivation from: Network, Station and
    Channel types.'''

    code = String.T(xmlstyle='attribute')
    start_date = Timestamp.T(optional=True, xmlstyle='attribute')
    end_date = Timestamp.T(optional=True, xmlstyle='attribute')
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


class Channel(BaseNode):
    '''Equivalent to SEED blockette 52 and parent element for the
    related the response blockettes.'''

    location_code = String.T(xmlstyle='attribute')
    external_reference_list = List.T(ExternalReference.T(xmltagname='ExternalReference'))
    latitude = Latitude.T(xmltagname='Latitude')
    longitude = Longitude.T(xmltagname='Longitude')
    elevation = Distance.T(xmltagname='Elevation')
    depth = Distance.T(xmltagname='Depth')
    azimuth = Azimuth.T(optional=True, xmltagname='Azimuth')
    dip = Dip.T(optional=True, xmltagname='Dip')
    type_list = List.T(Type.T(xmltagname='Type'))
    sample_rate = SampleRate.T(optional=True, xmltagname='SampleRate')
    sample_rate_ratio = SampleRateRatio.T(optional=True, xmltagname='SampleRateRatio')
    storage_format = String.T(optional=True, xmltagname='StorageFormat')
    clock_drift = ClockDrift.T(optional=True, xmltagname='ClockDrift')
    calibration_units = Units.T(optional=True, xmltagname='CalibrationUnits')
    sensor = Equipment.T(optional=True, xmltagname='Sensor')
    pre_amplifier = Equipment.T(optional=True, xmltagname='PreAmplifier')
    data_logger = Equipment.T(optional=True, xmltagname='DataLogger')
    equipment = Equipment.T(optional=True, xmltagname='Equipment')
    response = Response.T(optional=True, xmltagname='Response')


class Station(BaseNode):
    '''This type represents a Station epoch. It is common to only have
    a single station epoch with the station's creation and termination
    dates as the epoch start and end dates.'''

    latitude = Latitude.T(xmltagname='Latitude')
    longitude = Longitude.T(xmltagname='Longitude')
    elevation = Distance.T(xmltagname='Elevation')
    site = Site.T(xmltagname='Site')
    vault = Unicode.T(optional=True, xmltagname='Vault')
    geology = Unicode.T(optional=True, xmltagname='Geology')
    equipment_list = List.T(Equipment.T(xmltagname='Equipment'))
    operator_list = List.T(Operator.T(xmltagname='Operator'))
    creation_date = Timestamp.T(xmltagname='CreationDate')
    termination_date = Timestamp.T(optional=True, xmltagname='TerminationDate')
    total_number_channels = Counter.T(optional=True, xmltagname='TotalNumberChannels')
    selected_number_channels = Counter.T(optional=True, xmltagname='SelectedNumberChannels')
    external_reference_list = List.T(ExternalReference.T(xmltagname='ExternalReference'))
    channel_list = List.T(Channel.T(xmltagname='Channel'))


class Network(BaseNode):
    '''This type represents the Network layer, all station metadata is
    contained within this element. The official name of the network or
    other descriptive information can be included in the Description
    element. The Network can contain 0 or more Stations.'''

    total_number_stations = Counter.T(optional=True, xmltagname='TotalNumberStations')
    selected_number_stations = Counter.T(optional=True, xmltagname='SelectedNumberStations')
    station_list = List.T(Station.T(xmltagname='Station'))


def value_or_none(x):
    if x is not None:
        return x.value
    else:
        return None

class FDSNStationXML(Object):
    '''Top-level type for Station XML. Required field are Source
    (network ID of the institution sending the message) and one or
    more Network containers or one or more Station containers.'''

    schema_version = Float.T(xmlstyle='attribute')
    source = String.T(xmltagname='Source')
    sender = String.T(optional=True, xmltagname='Sender')
    module = String.T(optional=True, xmltagname='Module')
    module_uri = String.T(optional=True, xmltagname='ModuleURI')
    created = Timestamp.T(xmltagname='Created')
    network_list = List.T(Network.T(xmltagname='Network'))

    xmltagname = 'FDSNStationXML'

    def get_pyrocko_stations(self, time=None, timespan=None):
        tt = ()
        if time is not None:
            tt = (time,)
        elif timespan is not None:
            tt = timespan

        from pyrocko import model
        pstations = []
        for network in self.network_list:
            if not network.spans(*tt):
                continue

            for station in network.station_list:
                if not station.spans(*tt):
                    continue

                if station.channel_list:

                    loc_to_channels = {}
                    loc_to_depth = {}
                    for channel in station.channel_list:
                        if not channel.spans(*tt):
                            continue

                        c = model.Channel(channel.code, 
                                azimuth=value_or_none(channel.azimuth),
                                dip=value_or_none(channel.dip))

                        loc = channel.location_code
                        if loc not in loc_to_channels:
                            loc_to_channels[loc] = []

                        loc_to_channels[loc].append(c)
                        loc_to_depth[loc] = value_or_none(channel.depth)

                    for loc in sorted(loc_to_channels.keys()):
                        pstation = model.Station(network.code, station.code, loc,
                                lat=station.latitude.value,
                                lon=station.longitude.value,
                                elevation=value_or_none(station.elevation),
                                depth=loc_to_depth[loc],
                                name=station.description, 
                                channels=loc_to_channels[loc])

                        pstations.append(pstation)

                else:
                    pstation = model.Station(network.code, station.code, '',
                        lat=station.latitude.value,
                        lon=station.longitude.value,
                        elevation=value_or_none(station.elevation),
                        name=station.description)
                    pstations.append(pstation)
                    
        return pstations

