# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
from __future__ import absolute_import
import logging
from pyrocko.guts import StringPattern, StringChoice, String, Float, Int,\
    Timestamp, Object, List, Union, Bool, Unicode
from pyrocko.model import event
from pyrocko.gui import marker
from pyrocko import moment_tensor
import numpy as num

logger = logging.getLogger('pyrocko.io.quakeml')

guts_prefix = 'quakeml'
guts_xmlns = 'http://quakeml.org/xmlns/bed/1.2'
polarity_choices = {'positive': 1, 'negative': -1, 'undecidable': None}


class QuakeMLError(Exception):
    pass


class NoPreferredOriginSet(QuakeMLError):
    pass


def one_element_or_none(li):
    if len(li) == 1:
        return li[0]
    elif len(li) == 0:
        return None
    else:
        logger.warning('More than one element in list: {}'.format(li))
        return None


class ResourceIdentifier(StringPattern):
    pattern = "^(smi|quakeml):[\\w\\d][\\w\\d\\-\\.\\*\\(\\)_~']{2,}/[\\w" +\
        "\\d\\-\\.\\*\\(\\)_~'][\\w\\d\\-\\.\\*\\(\\)\\+\\?_~'=,;#/&]*$"


class WhitespaceOrEmptyStringType(StringPattern):
    pattern = '^\\s*$'


class OriginUncertaintyDescription(StringChoice):
    choices = [
        'horizontal uncertainty',
        'uncertainty ellipse',
        'confidence ellipsoid']


class AmplitudeCategory(StringChoice):
    choices = ['point', 'mean', 'duration', 'period', 'integral', 'other']


class OriginDepthType(StringChoice):
    choices = [
        'from location',
        'from moment tensor inversion',
        'from modeling of broad-band P waveforms',
        'constrained by depth phases',
        'constrained by direct phases',
        'constrained by depth and direct phases',
        'operator assigned',
        'other']


class OriginType(StringChoice):
    choices = [
        'hypocenter',
        'centroid',
        'amplitude',
        'macroseismic',
        'rupture start',
        'rupture end']


class MTInversionType(StringChoice):
    choices = ['general', 'zero trace', 'double couple']


class EvaluationMode(StringChoice):
    choices = ['manual', 'automatic']


class EvaluationStatus(StringChoice):
    choices = ['preliminary', 'confirmed', 'reviewed', 'final', 'rejected']


class PickOnset(StringChoice):
    choices = ['emergent', 'impulsive', 'questionable']


class EventType(StringChoice):
    choices = [
        'not existing',
        'not reported',
        'earthquake',
        'anthropogenic event',
        'collapse',
        'cavity collapse',
        'mine collapse',
        'building collapse',
        'explosion',
        'accidental explosion',
        'chemical explosion',
        'controlled explosion',
        'experimental explosion',
        'industrial explosion',
        'mining explosion',
        'quarry blast',
        'road cut',
        'blasting levee',
        'nuclear explosion',
        'induced or triggered event',
        'rock burst',
        'reservoir loading',
        'fluid injection',
        'fluid extraction',
        'crash',
        'plane crash',
        'train crash',
        'boat crash',
        'other event',
        'atmospheric event',
        'sonic boom',
        'sonic blast',
        'acoustic noise',
        'thunder',
        'avalanche',
        'snow avalanche',
        'debris avalanche',
        'hydroacoustic event',
        'ice quake',
        'slide',
        'landslide',
        'rockslide',
        'meteorite',
        'volcanic eruption',
        'duplicate earthquake',
        'rockburst']


class DataUsedWaveType(StringChoice):
    choices = [
        'P waves',
        'body waves',
        'surface waves',
        'mantle waves',
        'combined',
        'unknown']


class AmplitudeUnit(StringChoice):
    choices = ['m', 's', 'm/s', 'm/(s*s)', 'm*s', 'dimensionless', 'other']


class EventDescriptionType(StringChoice):
    choices = [
        'felt report',
        'Flinn-Engdahl region',
        'local time',
        'tectonic summary',
        'nearest cities',
        'earthquake name',
        'region name']


class MomentTensorCategory(StringChoice):
    choices = ['teleseismic', 'regional']


class EventTypeCertainty(StringChoice):
    choices = ['known', 'suspected']


class SourceTimeFunctionType(StringChoice):
    choices = ['box car', 'triangle', 'trapezoid', 'unknown']


class PickPolarity(StringChoice):
    choices = list(polarity_choices.keys())


class AgencyID(String):
    pass


class Author(Unicode):
    pass


class Version(String):
    pass


class Phase(Object):
    value = String.T(xmlstyle='content')


class GroundTruthLevel(String):
    pass


class AnonymousNetworkCode(String):
    pass


class AnonymousStationCode(String):
    pass


class AnonymousChannelCode(String):
    pass


class AnonymousLocationCode(String):
    pass


class Type(String):
    pass


class MagnitudeHint(String):
    pass


class Region(Unicode):
    pass


class RealQuantity(Object):
    value = Float.T()
    uncertainty = Float.T(optional=True)
    lower_uncertainty = Float.T(optional=True)
    upper_uncertainty = Float.T(optional=True)
    confidence_level = Float.T(optional=True)


class IntegerQuantity(Object):
    value = Int.T()
    uncertainty = Int.T(optional=True)
    lower_uncertainty = Int.T(optional=True)
    upper_uncertainty = Int.T(optional=True)
    confidence_level = Float.T(optional=True)


class ConfidenceEllipsoid(Object):
    semi_major_axis_length = Float.T()
    semi_minor_axis_length = Float.T()
    semi_intermediate_axis_length = Float.T()
    major_axis_plunge = Float.T()
    major_axis_azimuth = Float.T()
    major_axis_rotation = Float.T()


class TimeQuantity(Object):
    value = Timestamp.T()
    uncertainty = Float.T(optional=True)
    lower_uncertainty = Float.T(optional=True)
    upper_uncertainty = Float.T(optional=True)
    confidence_level = Float.T(optional=True)


class TimeWindow(Object):
    begin = Float.T()
    end = Float.T()
    reference = Timestamp.T()


class ResourceReference(ResourceIdentifier):
    pass


class DataUsed(Object):
    wave_type = DataUsedWaveType.T()
    station_count = Int.T(optional=True)
    component_count = Int.T(optional=True)
    shortest_period = Float.T(optional=True)
    longest_period = Float.T(optional=True)


class EventDescription(Object):
    text = Unicode.T()
    type = EventDescriptionType.T(optional=True)


class SourceTimeFunction(Object):
    type = SourceTimeFunctionType.T()
    duration = Float.T()
    rise_time = Float.T(optional=True)
    decay_time = Float.T(optional=True)


class OriginQuality(Object):
    associated_phase_count = Int.T(optional=True)
    used_phase_count = Int.T(optional=True)
    associated_station_count = Int.T(optional=True)
    used_station_count = Int.T(optional=True)
    depth_phase_count = Int.T(optional=True)
    standard_error = Float.T(optional=True)
    azimuthal_gap = Float.T(optional=True)
    secondary_azimuthal_gap = Float.T(optional=True)
    ground_truth_level = GroundTruthLevel.T(optional=True)
    maximum_distance = Float.T(optional=True)
    minimum_distance = Float.T(optional=True)
    median_distance = Float.T(optional=True)


class Axis(Object):
    azimuth = RealQuantity.T()
    plunge = RealQuantity.T()
    length = RealQuantity.T()


class Tensor(Object):
    mrr = RealQuantity.T(xmltagname='Mrr')
    mtt = RealQuantity.T(xmltagname='Mtt')
    mpp = RealQuantity.T(xmltagname='Mpp')
    mrt = RealQuantity.T(xmltagname='Mrt')
    mrp = RealQuantity.T(xmltagname='Mrp')
    mtp = RealQuantity.T(xmltagname='Mtp')


class NodalPlane(Object):
    strike = RealQuantity.T()
    dip = RealQuantity.T()
    rake = RealQuantity.T()


class CompositeTime(Object):
    year = IntegerQuantity.T(optional=True)
    month = IntegerQuantity.T(optional=True)
    day = IntegerQuantity.T(optional=True)
    hour = IntegerQuantity.T(optional=True)
    minute = IntegerQuantity.T(optional=True)
    second = RealQuantity.T(optional=True)


class OriginUncertainty(Object):
    horizontal_uncertainty = Float.T(optional=True)
    min_horizontal_uncertainty = Float.T(optional=True)
    max_horizontal_uncertainty = Float.T(optional=True)
    azimuth_max_horizontal_uncertainty = Float.T(optional=True)
    confidence_ellipsoid = ConfidenceEllipsoid.T(optional=True)
    preferred_description = OriginUncertaintyDescription.T(optional=True)
    confidence_level = Float.T(optional=True)


class ResourceReferenceOptional(Union):
    members = [ResourceReference.T(), WhitespaceOrEmptyStringType.T()]


class CreationInfo(Object):
    agency_id = AgencyID.T(optional=True, xmltagname='agencyID')
    agency_uri = ResourceReference.T(optional=True, xmltagname='agencyURI')
    author = Author.T(optional=True)
    author_uri = ResourceReference.T(optional=True, xmltagname='authorURI')
    creation_time = Timestamp.T(optional=True)
    version = Version.T(optional=True)


class StationMagnitudeContribution(Object):
    station_magnitude_id = ResourceReference.T(xmltagname='stationMagnitudeID')
    residual = Float.T(optional=True)
    weight = Float.T(optional=True)


class PrincipalAxes(Object):
    t_axis = Axis.T()
    p_axis = Axis.T()
    n_axis = Axis.T(optional=True)


class NodalPlanes(Object):
    preferred_plane = Int.T(optional=True, xmlstyle='attribute')
    nodal_plane1 = NodalPlane.T(optional=True)
    nodal_plane2 = NodalPlane.T(optional=True)


class WaveformStreamID(Object):
    value = ResourceReferenceOptional.T(xmlstyle='content')
    network_code = AnonymousNetworkCode.T(xmlstyle='attribute')
    station_code = AnonymousStationCode.T(xmlstyle='attribute')
    channel_code = AnonymousChannelCode.T(optional=True, xmlstyle='attribute')
    location_code = AnonymousLocationCode.T(
        optional=True, xmlstyle='attribute')

    @property
    def nslc_id(self):
        return (self.network_code, self.station_code, self.location_code,
                self.channel_code)


class Comment(Object):
    id = ResourceReference.T(optional=True, xmlstyle='attribute')
    text = Unicode.T()
    creation_info = CreationInfo.T(optional=True)


class MomentTensor(Object):
    public_id = ResourceReference.T(
        xmlstyle='attribute', xmltagname='publicID')
    data_used_list = List.T(DataUsed.T())
    comment_list = List.T(Comment.T())
    derived_origin_id = ResourceReference.T(
        optional=True, xmltagname='derivedOriginID')
    moment_magnitude_id = ResourceReference.T(
        optional=True, xmltagname='momentMagnitudeID')
    scalar_moment = RealQuantity.T(optional=True)
    tensor = Tensor.T(optional=True)
    variance = Float.T(optional=True)
    variance_reduction = Float.T(optional=True)
    double_couple = Float.T(optional=True)
    clvd = Float.T(optional=True)
    iso = Float.T(optional=True)
    greens_function_id = ResourceReference.T(
        optional=True, xmltagname='greensFunctionID')
    filter_id = ResourceReference.T(optional=True, xmltagname='filterID')
    source_time_function = SourceTimeFunction.T(optional=True)
    method_id = ResourceReference.T(optional=True, xmltagname='methodID')
    category = MomentTensorCategory.T(optional=True)
    inversion_type = MTInversionType.T(optional=True)
    creation_info = CreationInfo.T(optional=True)

    def pyrocko_moment_tensor(self):
        mrr = self.tensor.mrr.value
        mtt = self.tensor.mtt.value
        mpp = self.tensor.mpp.value
        mrt = self.tensor.mrt.value
        mrp = self.tensor.mrp.value
        mtp = self.tensor.mtp.value
        mt = moment_tensor.MomentTensor(m_up_south_east=num.matrix([
             [mrr, mrt, mrp], [mrt, mtt, mtp], [mrp, mtp, mpp]]))

        return mt


class Amplitude(Object):
    public_id = ResourceReference.T(
        xmlstyle='attribute', xmltagname='publicID')
    comment_list = List.T(Comment.T())
    generic_amplitude = RealQuantity.T()
    type = Type.T(optional=True)
    category = AmplitudeCategory.T(optional=True)
    unit = AmplitudeUnit.T(optional=True)
    method_id = ResourceReference.T(optional=True, xmltagname='methodID')
    period = RealQuantity.T(optional=True)
    snr = Float.T(optional=True)
    time_window = TimeWindow.T(optional=True)
    pick_id = ResourceReference.T(optional=True, xmltagname='pickID')
    waveform_id = WaveformStreamID.T(optional=True, xmltagname='waveformID')
    filter_id = ResourceReference.T(optional=True, xmltagname='filterID')
    scaling_time = TimeQuantity.T(optional=True)
    magnitude_hint = MagnitudeHint.T(optional=True)
    evaluation_mode = EvaluationMode.T(optional=True)
    evaluation_status = EvaluationStatus.T(optional=True)
    creation_info = CreationInfo.T(optional=True)


class Magnitude(Object):
    public_id = ResourceReference.T(
        xmlstyle='attribute', xmltagname='publicID')
    comment_list = List.T(Comment.T())
    station_magnitude_contribution_list = List.T(
        StationMagnitudeContribution.T())
    mag = RealQuantity.T()
    type = Type.T(optional=True)
    origin_id = ResourceReference.T(optional=True, xmltagname='originID')
    method_id = ResourceReference.T(optional=True, xmltagname='methodID')
    station_count = Int.T(optional=True)
    azimuthal_gap = Float.T(optional=True)
    evaluation_mode = EvaluationMode.T(optional=True)
    evaluation_status = EvaluationStatus.T(optional=True)
    creation_info = CreationInfo.T(optional=True)


class StationMagnitude(Object):
    public_id = ResourceReference.T(
        xmlstyle='attribute', xmltagname='publicID')
    comment_list = List.T(Comment.T())
    origin_id = ResourceReference.T(optional=True, xmltagname='originID')
    mag = RealQuantity.T()
    type = Type.T(optional=True)
    amplitude_id = ResourceReference.T(optional=True, xmltagname='amplitudeID')
    method_id = ResourceReference.T(optional=True, xmltagname='methodID')
    waveform_id = WaveformStreamID.T(optional=True, xmltagname='waveformID')
    creation_info = CreationInfo.T(optional=True)


class Arrival(Object):
    public_id = ResourceReference.T(
        xmlstyle='attribute', xmltagname='publicID')
    comment_list = List.T(Comment.T())
    pick_id = ResourceReference.T(xmltagname='pickID')
    phase = Phase.T()
    time_correction = Float.T(optional=True)
    azimuth = Float.T(optional=True)
    distance = Float.T(optional=True)
    takeoff_angle = RealQuantity.T(optional=True)
    time_residual = Float.T(optional=True)
    horizontal_slowness_residual = Float.T(optional=True)
    backazimuth_residual = Float.T(optional=True)
    time_weight = Float.T(optional=True)
    time_used = Int.T(optional=True)
    horizontal_slowness_weight = Float.T(optional=True)
    backazimuth_weight = Float.T(optional=True)
    earth_model_id = ResourceReference.T(
        optional=True, xmltagname='earthModelID')
    creation_info = CreationInfo.T(optional=True)


class Pick(Object):
    public_id = ResourceReference.T(
        xmlstyle='attribute', xmltagname='publicID')
    comment_list = List.T(Comment.T())
    time = TimeQuantity.T()
    waveform_id = WaveformStreamID.T(xmltagname='waveformID')
    filter_id = ResourceReference.T(optional=True, xmltagname='filterID')
    method_id = ResourceReference.T(optional=True, xmltagname='methodID')
    horizontal_slowness = RealQuantity.T(optional=True)
    backazimuth = RealQuantity.T(optional=True)
    slowness_method_id = ResourceReference.T(
        optional=True, xmltagname='slownessMethodID')
    onset = PickOnset.T(optional=True)
    phase_hint = Phase.T(optional=True)
    polarity = PickPolarity.T(optional=True)
    evaluation_mode = EvaluationMode.T(optional=True)
    evaluation_status = EvaluationStatus.T(optional=True)
    creation_info = CreationInfo.T(optional=True)

    @property
    def pyrocko_polarity(self):
        return polarity_choices.get(self.polarity, None)

    def get_pyrocko_phase_marker(self, event=None):
        if not self.phase_hint:
            logger.warn('Pick %s: phase_hint undefined' % self.public_id)
            phasename = 'undefined'
        else:
            phasename = self.phase_hint.value

        return marker.PhaseMarker(
            event=event, nslc_ids=[self.waveform_id.nslc_id],
            tmin=self.time.value, tmax=self.time.value,
            phasename=phasename,
            polarity=self.pyrocko_polarity,
            automatic=self.evaluation_mode)


class FocalMechanism(Object):
    public_id = ResourceReference.T(
        xmlstyle='attribute', xmltagname='publicID')
    waveform_id_list = List.T(WaveformStreamID.T(xmltagname='waveformID'))
    comment_list = List.T(Comment.T())
    moment_tensor_list = List.T(MomentTensor.T())
    triggering_origin_id = ResourceReference.T(
        optional=True, xmltagname='triggeringOriginID')
    nodal_planes = NodalPlanes.T(optional=True)
    principal_axes = PrincipalAxes.T(optional=True)
    azimuthal_gap = Float.T(optional=True)
    station_polarity_count = Int.T(optional=True)
    misfit = Float.T(optional=True)
    station_distribution_ratio = Float.T(optional=True)
    method_id = ResourceReference.T(optional=True, xmltagname='methodID')
    evaluation_mode = EvaluationMode.T(optional=True)
    evaluation_status = EvaluationStatus.T(optional=True)
    creation_info = CreationInfo.T(optional=True)


class Origin(Object):
    public_id = ResourceReference.T(
        xmlstyle='attribute', xmltagname='publicID')
    composite_time_list = List.T(CompositeTime.T())
    comment_list = List.T(Comment.T())
    origin_uncertainty_list = List.T(OriginUncertainty.T())
    arrival_list = List.T(Arrival.T())
    time = TimeQuantity.T()
    longitude = RealQuantity.T()
    latitude = RealQuantity.T()
    depth = RealQuantity.T(optional=True)
    depth_type = OriginDepthType.T(optional=True)
    time_fixed = Bool.T(optional=True)
    epicenter_fixed = Bool.T(optional=True)
    reference_system_id = ResourceReference.T(
        optional=True, xmltagname='referenceSystemID')
    method_id = ResourceReference.T(optional=True, xmltagname='methodID')
    earth_model_id = ResourceReference.T(
        optional=True, xmltagname='earthModelID')
    quality = OriginQuality.T(optional=True)
    type = OriginType.T(optional=True)
    region = Region.T(optional=True)
    evaluation_mode = EvaluationMode.T(optional=True)
    evaluation_status = EvaluationStatus.T(optional=True)
    creation_info = CreationInfo.T(optional=True)

    def position_values(self):
        lat = self.latitude.value
        lon = self.longitude.value
        if not self.depth:
            logger.warn(
                'Origin %s: Depth is undefined. Set to depth=0.' %
                self.public_id)
            depth = 0.
        else:
            depth = self.depth.value

        return lat, lon, depth

    def get_pyrocko_event(self):
        lat, lon, depth = self.position_values()
        otime = self.time.value
        if self.creation_info:
            cat = self.creation_info.agency_id
        else:
            cat = None

        return event.Event(
            name=self.public_id,
            lat=lat,
            lon=lon,
            time=otime,
            depth=depth,
            catalog=cat,
            region=self.region)


class Event(Object):
    public_id = ResourceReference.T(
        xmlstyle='attribute', xmltagname='publicID')
    description_list = List.T(EventDescription.T())
    comment_list = List.T(Comment.T())
    focal_mechanism_list = List.T(FocalMechanism.T())
    amplitude_list = List.T(Amplitude.T())
    magnitude_list = List.T(Magnitude.T())
    station_magnitude_list = List.T(StationMagnitude.T())
    origin_list = List.T(Origin.T())
    pick_list = List.T(Pick.T())
    preferred_origin_id = ResourceReference.T(
        optional=True, xmltagname='preferredOriginID')
    preferred_magnitude_id = ResourceReference.T(
        optional=True, xmltagname='preferredMagnitudeID')
    preferred_focal_mechanism_id = ResourceReference.T(
        optional=True, xmltagname='preferredFocalMechanismID')
    type = EventType.T(
        optional=True)
    type_certainty = EventTypeCertainty.T(
        optional=True)
    creation_info = CreationInfo.T(
        optional=True)
    region = Region.T(
        optional=True)

    def describe(self):
        return '''%s:
    origins: %i %s
    magnitudes: %i %s
    focal_machanisms: %i %s
    picks: %i
    amplitudes: %i
    station_magnitudes: %i''' % (
            self.public_id,
            len(self.origin_list),
            '@' if self.preferred_origin_id else '-',
            len(self.magnitude_list),
            '@' if self.preferred_magnitude_id else '-',
            len(self.focal_mechanism_list),
            '@' if self.preferred_focal_mechanism_id else '-',
            len(self.pick_list),
            len(self.amplitude_list),
            len(self.station_magnitude_list))

    def get_pyrocko_phase_markers(self):
        event = self.get_pyrocko_event()
        return [
            p.get_pyrocko_phase_marker(event=event) for p in self.pick_list]

    def get_pyrocko_event(self):
        '''
        Convert into Pyrocko event object.

        Uses *preferred* origin, magnitude, and moment tensor. If no preferred
        item is specified, it picks the first from the list and emits a
        warning.
        '''

        origin = self.preferred_origin
        if not origin and self.origin_list:
            origin = self.origin_list[0]
            if len(self.origin_list) > 1:
                logger.warn(
                    'Event %s: No preferred origin set, '
                    'more than one available, using first' % self.public_id)

        if not origin:
            raise QuakeMLError(
                'No origin available for event: %s' % self.public_id)

        ev = origin.get_pyrocko_event()

        foc_mech = self.preferred_focal_mechanism
        if not foc_mech and self.focal_mechanism_list:
            foc_mech = self.focal_mechanism_list[0]
            if len(self.focal_mechanism_list) > 1:
                logger.warn(
                    'Event %s: No preferred focal mechanism set, '
                    'more than one available, using first' % ev.name)

        if foc_mech and foc_mech.moment_tensor_list:
            ev.moment_tensor = \
                foc_mech.moment_tensor_list[0].pyrocko_moment_tensor()

            if len(foc_mech.moment_tensor_list) > 1:
                logger.warn(
                    'more than one moment tensor available, using first')

        mag = None
        pref_mag = self.preferred_magnitude
        if pref_mag:
            mag = pref_mag
        elif self.magnitude_list:
            mag = self.magnitude_list[0]
            if len(self.magnitude_list) > 1:
                logger.warn(
                    'Event %s: No preferred magnitude set, '
                    'more than one available, using first' % ev.name)

        if mag:
            ev.magnitude = mag.mag.value
            ev.magnitude_type = mag.type

        ev.region = self.get_effective_region()

        return ev

    def get_effective_region(self):
        if self.region:
            return self.region

        for desc in self.description_list:
            if desc.type in ('Flinn-Engdahl region', 'region name'):
                return desc.text

        return None

    @property
    def preferred_origin(self):
        return one_element_or_none(
            [x for x in self.origin_list
             if x.public_id == self.preferred_origin_id])

    @property
    def preferred_magnitude(self):
        return one_element_or_none(
            [x for x in self.magnitude_list
             if x.public_id == self.preferred_magnitude_id])

    @property
    def preferred_focal_mechanism(self):
        return one_element_or_none(
            [x for x in self.focal_mechanism_list
             if x.public_id == self.preferred_focal_mechanism_id])


class EventParameters(Object):
    public_id = ResourceReference.T(
        xmlstyle='attribute', xmltagname='publicID')
    comment_list = List.T(Comment.T())
    event_list = List.T(Event.T(xmltagname='event'))
    description = Unicode.T(optional=True)
    creation_info = CreationInfo.T(optional=True)


class QuakeML(Object):
    xmltagname = 'quakeml'
    xmlns = 'http://quakeml.org/xmlns/quakeml/1.2'
    guessable_xmlns = [xmlns, guts_xmlns]

    event_parameters = EventParameters.T(optional=True)

    def get_events(self):
        return self.event_parameters.event_list

    def get_pyrocko_events(self):
        '''Extract a list of :py:class:`pyrocko.model.Event` instances'''
        events = []
        for e in self.event_parameters.event_list:
            events.append(e.get_pyrocko_event())

        return events

    def get_pyrocko_phase_markers(self):
        markers = []
        for e in self.event_parameters.event_list:
            markers.extend(e.get_pyrocko_phase_markers())

        return markers

    @classmethod
    def load_xml(cls, *args, **kwargs):
        kwargs['ns_hints'] = [
            'http://quakeml.org/xmlns/quakeml/1.2',
            'http://quakeml.org/xmlns/bed/1.2']

        kwargs['ns_ignore'] = True

        return super(QuakeML, cls).load_xml(*args, **kwargs)
