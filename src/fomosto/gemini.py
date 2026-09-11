import asyncio
import logging
import os
import signal
import subprocess
from pathlib import Path
from subprocess import PIPE, Popen

import numpy as np
from pyrocko import gf, io, model, trace, util
from pyrocko.guts import Float, Int, Object, String
from pyrocko.moment_tensor import MomentTensor, symmat6

logger = logging.getLogger("pyrocko.fomosto.gemini")
logging.basicConfig(level=logging.DEBUG)


# Data files (green/, sources/, stations/, iasp91, lw200mhz, spec3k,
# seismogr). The programs run in here, so the file names in the configs stay
# relative to it, just like in gemini.sc, disp.sc and to.sc.
GEMINI_DIRECTORY = Path(
    "~/GeminiInhalt/FunktionierendeVersion/gemini-2.2.1"
).expanduser()

# The Fortran binaries, installed outside the data directory.
BIN_DIRECTORY = Path("~/.local/bin").expanduser()

guts_prefix = "gemini"

program_bins = {
    "gemini": BIN_DIRECTORY / "gemini_inputfile",
    "dispec": BIN_DIRECTORY / "dispec_inputfile",
    "totido": BIN_DIRECTORY / "totido_inputfile",
}


# Tests the existence of the binaries.
def have_backend():
    return all(binary.is_file() for binary in program_bins.values())


# Here, the input parameters for the simulated source, Gemini, Dispec and Totido are defined.
# Later on it could be useful to define the Earthmodel (like iasp91 (not anisotropic), or other anisotropic models) here too.
# Also the maximum degrees depending on the frequency could be defined here, but for now it is just the default lw200mhz file.
class GeminiSource(Object):
    """
    Tags:   [phys] enters the calculation,
            [label] only reaches the seismogr header,
            [pad] is read by readcmt.f and discarded.
    A [pad] value is irrelevant, but its column width is not.
    """

    # Line 1: basic event information.
    event_id: str = String.T(default="ICHteste")  # [label] 8 characters
    date: str = String.T(default="01/01/26")  # [label] dd/mm/yy
    origin_time: str = String.T(default="00:00:00.0")  # [label] hh:mm:ss.s
    latitude: float = Float.T(default=40.64)  # [pad] use centroid_latitude
    longitude: float = Float.T(default=29.83)  # [pad] use centroid_longitude
    depth: float = Float.T(default=90.0)  # [pad] use centroid_depth
    mb: float = Float.T(default=2.0)  # [pad] body-wave magnitude
    MS: float = Float.T(default=4.0)  # [pad] surface-wave magnitude
    region: str = String.T(
        default="KARAMURSEL TURKEI"
    )  # [pad] max 24 characters

    # Line 2: CMT header and inversion errors.
    # Origin of the epicentre data: 'PDE' preliminary, 'MLI' NEIC monthly
    # listings, 'ISC' ISC catalogue.
    source_type: str = String.T(default="PDE")  # [pad]
    bw_stations: int = Int.T(default=0)  # [pad] body-wave stations used
    bw_records: int = Int.T(default=0)  # [pad] body-wave records used
    bw_cutoff: int = Int.T(default=0)  # [pad] cut-off in s, i4 in readcmt.f
    mw_stations: int = Int.T(default=0)  # [pad] mantle-wave stations used
    mw_records: int = Int.T(default=0)  # [pad] mantle-wave records used
    mw_cutoff: int = Int.T(default=0)  # [pad] cut-off in s, i4 in readcmt.f
    centroid_time: float = Float.T(
        default=0.0
    )  # [phys] offset from origin_time in s
    centroid_time_error: float = Float.T(
        default=0.0
    )  # [pad] standard error in s
    centroid_latitude: float = Float.T(
        default=40.64
    )  # [phys] source latitude in degrees
    centroid_latitude_error: float = Float.T(
        default=0.0
    )  # [pad] standard error
    centroid_longitude: float = Float.T(
        default=29.83
    )  # [phys] source longitude in degrees
    centroid_longitude_error: float = Float.T(
        default=0.0
    )  # [pad] standard error
    centroid_depth: float = Float.T(default=10.0)  # [phys] source depth in km
    centroid_depth_error: float = Float.T(
        default=0.0
    )  # [pad] standard error in km

    # Line 3: moment tensor. Every component is scaled by 10**exponent and
    # carries the standard error of the inversion.
    duration: float = Float.T(default=0.0)  # [pad] read but never returned
    exponent: int = Int.T(
        default=27
    )  # [phys] common exponent of all moment values
    mrr: float = Float.T(default=2.02)  # [phys] radial-radial component
    mrr_error: float = Float.T(default=0.0)  # [pad] standard error of mrr
    mss: float = Float.T(default=-0.07)  # [phys] south-south component
    mss_error: float = Float.T(default=0.0)  # [pad] standard error of mss
    mee: float = Float.T(default=5.80)  # [phys] east-east component
    mee_error: float = Float.T(default=0.0)  # [pad] standard error of mee
    mrs: float = Float.T(default=0.02)  # [phys] radial-south component
    mrs_error: float = Float.T(default=0.0)  # [pad] standard error of mrs
    mre: float = Float.T(default=-0.34)  # [phys] radial-east component
    mre_error: float = Float.T(default=0.0)  # [pad] standard error of mre
    mse: float = Float.T(default=-1.34)  # [phys] south-east component
    mse_error: float = Float.T(default=0.0)  # [pad] standard error of mse

    # Line 4: principal axes and fault-plane solution. Entirely [pad] -
    # readcmt.f returns none of it, and it is derivable from the tensor.
    eigenvalue1: float = Float.T(default=0.0)  # [pad] principal axis 1
    plunge1: int = Int.T(default=0)  # [pad] 0-90 degrees, i3 in readcmt.f
    azimuth1: int = Int.T(default=0)  # [pad] 0-360 degrees, i4 in readcmt.f
    eigenvalue2: float = Float.T(default=0.0)  # [pad] principal axis 2
    plunge2: int = Int.T(default=0)  # [pad] 0-90 degrees, i3 in readcmt.f
    azimuth2: int = Int.T(default=0)  # [pad] 0-360 degrees, i4 in readcmt.f
    eigenvalue3: float = Float.T(default=0.0)  # [pad] principal axis 3
    plunge3: int = Int.T(default=0)  # [pad] 0-90 degrees, i3 in readcmt.f
    azimuth3: int = Int.T(default=0)  # [pad] 0-360 degrees, i4 in readcmt.f
    scalar_moment: float = Float.T(
        default=1.38
    )  # [pad] scaled by 10**exponent
    strike1: int = Int.T(default=0)  # [pad] 0-360 degrees, i4 in readcmt.f
    dip1: int = Int.T(default=0)  # [pad] 0-90 degrees, i3 in readcmt.f
    rake1: int = Int.T(
        default=0
    )  # [pad] -180 to +180 degrees, i5 in readcmt.f
    strike2: int = Int.T(default=0)  # [pad] auxiliary plane, i4 in readcmt.f
    dip2: int = Int.T(default=0)  # [pad] auxiliary plane, i3 in readcmt.f
    rake2: int = Int.T(default=0)  # [pad] auxiliary plane, i5 in readcmt.f


# This class converts the GeminiSource into the CMT file that Dispec reads.
class CMTBuilder(object):
    """Turn one GeminiSource into the CMT file that Dispec reads."""

    text_widths = {"event_id": 8, "source_type": 3, "region": 24}

    # readcmt.f reads with:
    # line 1: a8,5(1x,a2),1x,f4.1,f7.2,f8.2,f6.1,2f3.1,a24
    # line 2: a3,2(4x,i2,i3,i4),4x,f6.1,f4.1,f7.2,f5.2,f8.2,f5.2,f6.1,f5.1
    # line 3: 4x,f4.1,4x,i2,6(f6.2,f5.2)
    # line 4: 3(f7.2,i3,i4),f7.2,2(i4,i3,i5)
    line_widths = (79, 79, 80, 73)

    def __init__(
        self,
        source,
        filepath=GEMINI_DIRECTORY / "sources" / "Quelle",
    ):
        self.source = source
        self.filepath = Path(filepath)

    @staticmethod
    def map_value_to_column_type(value, kind, length=None):
        if value is None:
            if kind == "int":
                return 0
            elif kind == "float":
                return 0.0
            elif kind == "date":
                return "00/00/00"
            elif kind == "time":
                return "00:00:00.0"
            else:
                return " " * (length or 1)

        if kind == "int":
            return round(float(value))
        elif kind == "float":
            return float(value)
        elif kind == "str":
            return str(value)[:length] if length else str(value)
        return value

    def extract_column_types(self):
        source_column_values = {}
        for data_attribute in self.source.T.properties:
            if data_attribute.name == "date":
                kind = "date"
            elif data_attribute.name == "origin_time":
                kind = "time"
            elif isinstance(data_attribute, Int.T):
                kind = "int"
            elif isinstance(data_attribute, Float.T):
                kind = "float"
            else:
                kind = "str"

            source_column_values[data_attribute.name] = (
                self.map_value_to_column_type(
                    getattr(self.source, data_attribute.name),
                    kind,
                    self.text_widths.get(data_attribute.name),
                )
            )

        return source_column_values

    def lines(self):
        v = self.extract_column_types()
        line1 = f"{v['event_id']:8s} {v['date'][0:2]}/{v['date'][3:5]}/{v['date'][6:8]} {v['origin_time']}{v['latitude']:7.2f}{v['longitude']:8.2f}{v['depth']:6.1f}{v['mb']:3.1f}{v['MS']:3.1f}{v['region']:<24}"
        line2 = f"{v['source_type']:3s} BW:{v['bw_stations']:2d}{v['bw_records']:3d}{v['bw_cutoff']:4d} MW:{v['mw_stations']:2d}{v['mw_records']:3d}{v['mw_cutoff']:4d} DT={v['centroid_time']:6.1f}{v['centroid_time_error']:4.1f}{v['centroid_latitude']:7.2f}{v['centroid_latitude_error']:5.2f}{v['centroid_longitude']:8.2f}{v['centroid_longitude_error']:5.2f}{v['centroid_depth']:6.1f}{v['centroid_depth_error']:5.1f}"
        line3 = f" DUR{v['duration']:4.1f} EX{v['exponent']:3d}{v['mrr']:6.2f}{v['mrr_error']:5.2f}{v['mss']:6.2f}{v['mss_error']:5.2f}{v['mee']:6.2f}{v['mee_error']:5.2f}{v['mrs']:6.2f}{v['mrs_error']:5.2f}{v['mre']:6.2f}{v['mre_error']:5.2f}{v['mse']:6.2f}{v['mse_error']:5.2f}"
        line4 = f"{v['eigenvalue1']:7.2f}{v['plunge1']:3d}{v['azimuth1']:4d}{v['eigenvalue2']:7.2f}{v['plunge2']:3d}{v['azimuth2']:4d}{v['eigenvalue3']:7.2f}{v['plunge3']:3d}{v['azimuth3']:4d}{v['scalar_moment']:7.2f}{v['strike1']:4d}{v['dip1']:3d}{v['rake1']:5d}{v['strike2']:4d}{v['dip2']:3d}{v['rake2']:5d}"
        return [line1, line2, line3, line4]

    def check(self, lines):
        for i, line in enumerate(lines):
            if len(line) > self.line_widths[i]:
                raise ValueError(
                    f"Line {i + 1} is too long: {len(line)} characters, max {self.line_widths[i]}"
                )
            elif len(line) < self.line_widths[i]:
                raise ValueError(
                    f"Line {i + 1} is too short: {len(line)} characters, min {self.line_widths[i]}"
                )
        return lines

    def write(self, filepath=None):
        filepath = Path(filepath or self.filepath)
        lines = self.check(self.lines())
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text("\n".join(lines) + "\n")

        logger.info(f"source written to {filepath}")
        return filepath


# SourceDescriptionConverter(GeminiSource()).write(filename='sources/QuellenTest')


class GeminiStation(Object):
    # also read by Dispec, in the order of the columns of stations/GRSN_2003. for a greensfunction store, this would be the area to define the stations.
    filepath: str = String.T(default="stations/GRSN_2003")


class GeminiEarthModel(Object):
    # Right now, the earth model is written in a text file and read by Gemini.
    filepath: str = String.T(default="iasp91")


class GeminiMaximumDegreeWindow(Object):
    filepath: str = String.T(default="lw200mhz")


class GeminiConfig(Object):
    """Input parameters for GEMINI, in the order gemini.sc feeds them to the
    binary.
    GEMINI computes the basis solutions (expansion coefficients) in
    the frequency-degree domain."""

    # Kind of motion to calculate: 1 for P-SV only, 2 for SH only, 3 for both.
    what_motion: int = Int.T(default=3)
    # Verbose level of the monitor output: 0 reports every frequency, n
    # reports every degree l with mod(l, n) == 0.
    print_level: int = Int.T(default=0)
    # Length of the seismogram in seconds.
    seismo_lenght: int = Int.T(default=5400)
    # Damping time of the complex frequency (Laplace transform). A fifth of
    # seismo_lenght is a good choice.
    damping_time: int = Int.T(default=2000)
    # Minimum frequency in millihertz. May be 0, GEMINI then raises it to at
    # least 1/seismo_lenght.
    minimum_frequency: int = Int.T(default=0)
    # Maximum frequency in millihertz.
    maximum_frequency: int = Int.T(default=50)
    # Take dispersion (attenuation) into account, which makes the elastic
    # moduli frequency-dependent: 1 for yes, 0 for no.
    dispersion_switch: int = Int.T(default=1)
    # Minimum degree of the spherical harmonics, 0 is recommended.
    minimum_degree: int = Int.T(default=0)
    # Maximum degree of the spherical harmonics. GEMINI does not compute
    # beyond this limit.
    maximum_degree: int = Int.T(default=5000)
    # Step in the degree domain, normally 1 because of the 2*Pi periodicity.
    # Larger values speed up the calculation but make the Earth
    # 2*Pi/degree_step-periodic and thus produce alias effects.
    degree_step: int = Int.T(default=1)
    # Depth of the source in km, may be 0.
    source_depth: int = Int.T(default=10)
    # Accuracy ruling the Bulirsch-Stoer integrator. Do not be too greedy,
    # 1.e-4 is sufficient.
    accuracy: float = Float.T(default=1.0e-4)
    # File name of the earth model.
    earth_model: str = String.T(default="iasp91")
    # File name of the window in the frequency-degree domain, holding
    # tabulated maximum degrees for selected frequencies.
    omega_ell_window: str = String.T(default="lw200mhz")
    # Name of the output file with the expansion coefficients.
    output_filepath: str = String.T(default="green/bas.f50.d100.3.out")
    # Confirm the input.
    confirmation: int = Int.T(default=1)


class GeminiRecieverSection(Object):
    distance_min: float = Float.T(default=-30.0)
    distance_max: float = Float.T(default=-71.0)
    distance_delta: float = Float.T(default=10.0)
    reciever_type: int = Int.T(default=4)


class DispecConfig(Object):
    """Input parameters for DISPEC, in the order disp.sc feeds them to the
    binary.
    DISPEC combines the GEMINI basis solutions with source and
    stations and writes the spectra to 'spec3k' in the current directory."""

    # File with the basis solutions calculated by GEMINI.
    basis_solutions: str = String.T(default="green/bas.f50.d100.3.out")
    # Source file holding exactly ONE set of earthquake parameters in
    # Harvard-CMT format. dispec.f reads this name with '(a70)', so it has to
    # stay short.
    source_file: str = String.T(default="sources/Quelle")
    # Source mechanism: 'm' for moment tensor, 'f' for single force.
    source_mechanism: str = String.T(default="m")
    # Maximum order 'm' in the sum over the spherical harmonics. Set 0 for an
    # explosion-type source.
    maximum_order: int = Int.T(default=2)
    # Window in the omega-l domain, the same file GEMINI needs.
    omega_ell_window: str = String.T(default="lw200mhz")
    # Length of the taper applied on the l-range to avoid cut-off effects. An
    # empirical number, do not choose it too large or it cuts into the
    # surface wave branch.
    ell_taper: int = Int.T(default=40)
    # File with station names and parameters in IRIS-DMC format.
    stations_file: str = String.T(default="stations/GRSN_2003")
    # Receiver type: 1 station by latitude and longitude in degrees, 2 station
    # by its abbreviation (e.g. PFO), 3 all stations in stations_file,
    # 4 section along the great circle between two points on the sphere.
    reciever_type: int = Int.T(default=3)
    # Receiver selection matching reciever_type. Examples: type 1 '48.3 8.3',
    # type 2 'BFO', type 3 'all' (a dummy, the program only reads format
    # '(1x)'), type 4 '-30. -71. 48.3 8.3 10'.
    recievers_file: str = String.T(default="all")
    # Horizontal displacement is calculated in source centered coordinates by
    # default. A number greater than zero yields station centered
    # coordinates, i.e. north-south and east-west.
    nsew_coordinates: int = Int.T(default=1)


class TotidoConfig(Object):
    """Input parameters for TOTIDO, in the order to.sc feeds them to the
    binary.
    TOTIDO turns the spectra from DISPEC into time series."""

    # File with the spectra generated by DISPEC. In to.sc this is the only
    # value without a default, it has to be passed as '-f <spectrumfile>'.
    spectrum_filepath: str = String.T(default="spec3k")
    # File with real and imaginary part of the instrument transfer function at
    # the frequencies used in GEMINI and DISPEC, one line per frequency:
    # frequency, real part, imaginary part.
    response_file: str = String.T(default="")
    # Time by which the beginning of the time series is delayed.
    time_shift: float = Float.T(default=0.0)
    # Number of Butterworth low pass filters applied to the seismogram.
    lowpass_number: int = Int.T(default=0)
    # Order of the low pass filter, e.g. 7.
    lowpass_order: int = Int.T(default=0)
    # Corner frequency of the low pass filter in Hz, e.g. 0.01.
    lowpass_cutoff: float = Float.T(default=0.025)
    # Number of Butterworth high pass filters applied to the seismogram.
    highpass_number: int = Int.T(default=0)
    # Order of the high pass filter, e.g. 3.
    highpass_order: int = Int.T(default=0)
    # Corner frequency of the high pass filter in Hz, e.g. 0.008.
    highpass_cutoff: float = Float.T(default=0.025)
    # Output format: 'a' for ASCII, 's' for SFF.
    output_format: str = String.T(default="a")
    # The FFT needs 2**n samples and zeros are appended to reach such a
    # length. A number n greater than zero multiplies the number of samples
    # by 2**n on top of that, interpolating and smoothing the time series.
    zero_padding: int = Int.T(default=2)
    # Type of seismogram: 'd' displacement, 'v' velocity, 'a' acceleration,
    # 'g' accelerometer response.
    seismo_type: str = String.T(default="v")
    # Length of the output time series in seconds.
    seconds_out: int = Int.T(default=5400)
    # getopts option string of to.sc, through which the values above are
    # overridden on the command line.
    opts: str = String.T(default="L:l:H:h:s:o:O:p:r:f:")


class GeminiConfigFull(Object):
    gemini_config: GeminiConfig = GeminiConfig.T(default=GeminiConfig.D())
    dispec_config: DispecConfig = DispecConfig.T(default=DispecConfig.D())
    totido_config: TotidoConfig = TotidoConfig.T(default=TotidoConfig.D())
    source: GeminiSource = GeminiSource.T(default=GeminiSource.D())
    station: GeminiStation = GeminiStation.T(default=GeminiStation.D())
    maximum_degree_based_on_frequency: GeminiMaximumDegreeWindow = (
        GeminiMaximumDegreeWindow.T(default=GeminiMaximumDegreeWindow.D())
    )


config_filepath = (
    GEMINI_DIRECTORY / "Configurations" / "gemini_config_full.yaml"
)


def dump_config(config, filepath=config_filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    config.dump(filename=str(filepath))
    logger.info(f"config dumped to {filepath}")
    return filepath


def load_config(filepath=config_filepath):
    return GeminiConfigFull.load(filename=str(filepath))


def gemini_input(conf):
    """Build the stdin block for GEMINI, like the heredoc in gemini.sc."""

    return (
        "\n".join(
            str(value)
            for value in [
                conf.what_motion,
                conf.print_level,
                conf.seismo_lenght,
                conf.damping_time,
                conf.minimum_frequency,
                conf.maximum_frequency,
                conf.dispersion_switch,
                conf.minimum_degree,
                conf.maximum_degree,
                conf.degree_step,
                conf.source_depth,
                conf.accuracy,
                conf.earth_model,
                conf.omega_ell_window,
                conf.output_filepath,
                conf.confirmation,
            ]
        )
        + "\n"
    )


def dispec_input(conf):
    """Build the stdin block for DISPEC, like the heredoc in disp.sc."""

    return (
        "\n".join(
            str(value)
            for value in [
                conf.basis_solutions,
                conf.source_file,
                conf.source_mechanism,
                conf.maximum_order,
                conf.omega_ell_window,
                conf.ell_taper,
                conf.stations_file,
                conf.reciever_type,
                conf.recievers_file,
                conf.nsew_coordinates,
            ]
        )
        + "\n"
    )


def totido_input(conf):
    """Build the stdin block for TOTIDO, like the heredoc in to.sc."""

    # totido.f reads each filter as 'number, (order, corner frequency)*number',
    # so the three values share one line. With number 0 the rest of the line is
    # ignored, which is why to.sc always writes all three.
    return (
        "\n".join(
            str(value)
            for value in [
                conf.spectrum_filepath,
                conf.response_file,
                conf.time_shift,
                f"{conf.lowpass_number} {conf.lowpass_order} {conf.lowpass_cutoff}",
                f"{conf.highpass_number} {conf.highpass_order} {conf.highpass_cutoff}",
                conf.zero_padding,
                conf.seismo_type,
                conf.seconds_out,
                conf.output_format,
            ]
        )
        + "\n"
    )


# run_program('dispec', dispec_input(load_config().dispec_config), base_directory)
class MseedConverter:
    def __init__(
        self,
        config,
        ascii_filepath=GEMINI_DIRECTORY / "seismogr",
        stations_filepath=GEMINI_DIRECTORY / "demo_stations.txt",
        event_filepath=GEMINI_DIRECTORY / "demo_event.txt",
        mseeds_dir=GEMINI_DIRECTORY / "mseed",
    ):
        self.config = config
        self.ascii_filepath = Path(ascii_filepath)
        self.stations_filepath = Path(stations_filepath)
        self.event_filepath = Path(event_filepath)
        self.mseeds_dir = Path(mseeds_dir)

        station_coords = []
        blocks_together = []
        block_number = 0

        with open(self.ascii_filepath) as input_file:
            for line in input_file:
                fields = line.split()
                if not fields:
                    continue
                try:
                    values = [float(value) for value in fields]
                except ValueError:
                    if fields[0] == "RECLat":
                        station_coords.append(
                            [float(fields[1]), float(fields[3])]
                        )
                        block_number += 1
                        blocks_together.append([])
                else:
                    if block_number:
                        blocks_together[block_number - 1].append(values)

        # station position
        station_latitudes = [coord[0] for coord in station_coords]
        station_longitudes = [coord[1] for coord in station_coords]

        location_tolerance = 1e-3

        def find_station(latitude, longitude):
            for catalog in (
                GEMINI_DIRECTORY / self.config.station.filepath,
                # GEMINI_DIRECTORY / 'stations/GRSN_2003',
                # GEMINI_DIRECTORY / 'stations/IRIS_1996'
            ):
                with open(catalog) as catalog_file:
                    for line in catalog_file:
                        fields = line.split()
                        if len(fields) < 6 or fields[1] == "s_station":
                            continue
                        try:
                            catalog_latitude = float(fields[3])
                            catalog_longitude = float(fields[4])
                            catalog_elevation = float(fields[5])
                        except ValueError:
                            continue
                        if (
                            abs(catalog_latitude - latitude)
                            < location_tolerance
                            and abs(catalog_longitude - longitude)
                            < location_tolerance
                        ):
                            return fields[2], fields[1], catalog_elevation, ""

            raise LookupError(
                f"no station at lat {latitude}, lon {longitude} in {self.config.station.filepath}"
            )

        stations = []
        for i in range(len(station_coords)):
            network, station_code, elevation, location = find_station(
                station_latitudes[i], station_longitudes[i]
            )
            stations.append(
                model.Station(
                    network=network,
                    station=station_code,
                    location=location,
                    lat=station_latitudes[i],
                    lon=station_longitudes[i],
                    elevation=elevation,
                    depth=0.0,
                )
            )

        model.dump_stations(stations, filename=str(self.stations_filepath))

        source = self.config.source

        day, month, year = source.date.split("/")
        start_time_string = f"20{year}-{month}-{day} {source.origin_time}"
        formatted_start_time = float(util.str_to_time(start_time_string))

        event = model.Event(
            lat=source.latitude,
            lon=source.longitude,
            depth=source.depth * 1000,
            time=formatted_start_time,
            name=source.event_id,
        )

        model.dump_events([event], filename=str(self.event_filepath))

        ## build traces
        traces = []
        for i in range(len(blocks_together)):
            data = np.array(blocks_together[i])
            times = data[:, 0]
            delta_time = times[1] - times[0]
            station = stations[i]
            # seismogr holds the columns time, Z, NS, EW (writeAscii.f). Z is
            # the radial component of dispec.f, positive up; NS and EW are
            # already rotated so that positive points north and east
            # (dispec.f, 'Transform to north-south and east-west components').
            # pyrocko and fomosto use NED with z downward, so north and east
            # pass through unchanged and only the vertical is flipped.
            for direction, column, sign in (
                ("N", 2, 1),
                ("E", 3, 1.0),
                ("D", 1, -1),
            ):
                traces.append(
                    trace.Trace(
                        network=station.network,
                        station=station.station,
                        location=station.location,
                        channel=direction,
                        tmin=formatted_start_time + times[0],
                        deltat=delta_time,
                        ydata=sign * data[:, column],
                    )
                )

        # save
        self.mseeds_dir.mkdir(parents=True, exist_ok=True)

        io.save(
            traces,
            filename_template=str(
                self.mseeds_dir
                / f"{{network}}.{{station}}.{{location}}.{{channel}}.mseed"
            ),
            format="mseed",
        )
        logger.info(f"Saved {len(traces)} traces to {self.mseeds_dir}")


async def run_program(
    program: str, input_string: str, gemini_directory: Path = GEMINI_DIRECTORY
) -> str:
    """Feed one input block into one of the Fortran programs"""
    binary = program_bins[program]
    logger.info(f"running {binary} in {gemini_directory}")
    logger.debug(f"input string:\n{input_string}")

    prog = await asyncio.create_subprocess_shell(
        str(binary),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=gemini_directory,
    )

    try:
        out, err = await prog.communicate(input=input_string.encode())
    except BaseException:
        prog.kill()
        await prog.wait()
        raise

    try:
        ret_code = await prog.wait()
    except asyncio.CancelledError:
        prog.terminate()
        raise

    if ret_code != 0:
        raise RuntimeError(
            f"{program} failed with return code {ret_code}\n"
            f"Error: {err.decode()}\n"
            f"Output: {out.decode()}"
        )
    logger.debug(f"{program} output:\n{out.decode()}")
    return out.decode()


async def run(
    config,
    gemini_directory=GEMINI_DIRECTORY,
    cmt_build=True,
    cmt_filepath=None,
    gemini_run=True,
    dispec_run=True,
    totido_run=True,
    mseed_convert=True,
    mseeds_dir=GEMINI_DIRECTORY / "mseed",
    stations_filepath=GEMINI_DIRECTORY / "demo_stations.txt",
    event_filepath=GEMINI_DIRECTORY / "demo_event.txt",
    snuffler_run=False,
):
    print(f"Running GEMINI pipeline with config: {config}")
    if cmt_build:
        if cmt_filepath is None:
            cmt_filepath = gemini_directory / config.dispec_config.source_file
        CMTBuilder(config.source).write(cmt_filepath)
        logger.info(f"CMT file written to {cmt_filepath}")
    if gemini_run:
        await run_program(
            "gemini", gemini_input(config.gemini_config), gemini_directory
        )
        logger.info("GEMINI run completed")
    if dispec_run:
        await run_program(
            "dispec", dispec_input(config.dispec_config), gemini_directory
        )
        logger.info("DISPEC run completed")
    if totido_run:
        await run_program(
            "totido", totido_input(config.totido_config), gemini_directory
        )
        logger.info("TOTIDO run completed")
    if mseed_convert:
        MseedConverter(
            config=config,
            ascii_filepath=gemini_directory / "seismogr",
            stations_filepath=stations_filepath,
            event_filepath=event_filepath,
            mseeds_dir=mseeds_dir,
        )
        logger.info("MSEED conversion completed")
    if snuffler_run:
        run_snuffler(
            gemini_directory=gemini_directory,
            mseeds_dir=mseeds_dir,
            stations_filepath=stations_filepath,
            event_filepath=event_filepath,
        )
        logger.info("Snuffler opened")


configlong = load_config()


async def run_parallel():
    await asyncio.gather(
        run(
            config=configlong,
            cmt_build=False,
            gemini_run=False,
            dispec_run=True,
            totido_run=False,
            mseed_convert=False,
            snuffler_run=False,
        ),
        run(
            config=configlong,
            cmt_build=False,
            gemini_run=False,
            dispec_run=True,
            totido_run=False,
            mseed_convert=False,
            snuffler_run=False,
        ),
    )


# asyncio.run(run_parallel())


def run_snuffler(
    gemini_directory=GEMINI_DIRECTORY,
    wait=True,
    mseeds_dir=GEMINI_DIRECTORY / "mseed",
    stations_filepath=GEMINI_DIRECTORY / "demo_stations.txt",
    event_filepath=GEMINI_DIRECTORY / "demo_event.txt",
):

    snuffler = Popen(
        [
            "snuffler",
            str(mseeds_dir),
            f"--stations={stations_filepath}",
            f"--events={event_filepath}",
        ],
        cwd=gemini_directory,
    )

    if not wait:
        return snuffler

    snuffler.wait()
    return snuffler.returncode


"""das init ding ist noch garnichts"""


def init(store_dir, variant, config_params=None):
    print("Init called")


class GeminiMomentTensor(object):
    """One moment tensor, given in NED and handed on in USE.

    NED (Mnn, Mee, Mdd, Mne, Mnd, Med) is what fomosto uses for the elastic10
    elementary sources, USE (Mrr, Mss, Mee, Mrs, Mre, Mse) is what line 3 of
    the CMT file expects. pyrocko does the basis change, this class only
    fixes the direction and the scaling.
    """

    def __init__(self, mnn, mee, mdd, mne, mnd, med, moment=1.0):
        # symmat6 takes the NED components in exactly this order, so the
        # arguments can be passed straight through. moment is in N m.
        self.moment_tensor = MomentTensor(
            m=symmat6(mnn, mee, mdd, mne, mnd, med) * moment
        )

    def m6_ned(self):
        return self.moment_tensor.m6()

    def m6_use(self):
        return self.moment_tensor.m6_up_south_east()

    def cmt_values(self):
        scaled_values = np.array(self.m6_use()) * 1e7  # N m -> dyne cm
        largest_scaled_value = np.abs(scaled_values).max()
        if largest_scaled_value == 0.0:
            return 0, scaled_values
        exponent = int(np.floor(np.log10(largest_scaled_value)))
        return exponent, scaled_values / 10.0**exponent


elastic10_tensors = [
    ("mmt1", GeminiMomentTensor(1, 0, 0, 1, 0, 0)),  # Mnn, Mne
    ("mmt2", GeminiMomentTensor(0, 0, 0, 0, 1, 1)),  # Mnd, Med
    ("mmt3", GeminiMomentTensor(0, 0, 1, 0, 0, 0)),  # Mdd
    ("mmt4", GeminiMomentTensor(0, 1, 0, 0, 0, 0)),  # Mee
]


def print_elastic10():
    """Show the four elementary sources in both conventions."""
    print(
        "%-6s %-30s %s"
        % (
            "id",
            "NED (nn, ee, dd, ne, nd, ed)",
            "USE (rr, ss, ee, rs, re, se)",
        )
    )
    for name, tensor in elastic10_tensors:
        print(
            "%-6s %-30s %s"
            % (
                name,
                " ".join("%4.0f" % v for v in tensor.m6_ned()),
                " ".join("%4.0f" % v for v in tensor.m6_use()),
            )
        )


def source_with_tensor(tensor, source):
    """Set the moment tensor of a GeminiSource, leave everything else alone."""

    source.exponent, komponenten = tensor.cmt_values()
    (
        source.mrr,
        source.mss,
        source.mee,
        source.mrs,
        source.mre,
        source.mse,
    ) = komponenten
    return source


def run_tensor(tensor, gemini_directory=GEMINI_DIRECTORY):
    """Send one moment tensor through Dispec, Totido, Converter and snuffler."""

    config = load_config()
    config.source = source_with_tensor(tensor, config.source)

    # source_converter laeuft als Subprozess und liest die YAML von der
    # Platte, deshalb muss der neue Tensor erst dorthin.
    dump_config(config)
    asyncio.run(
        run(config, gemini_directory, gemini_run=False, snuffler_run=True)
    )


# run_tensor(elastic10_tensors[0][1])
# run_tensor(elastic10_tensors[1][1])
# run_tensor(elastic10_tensors[2][1])
# run_tensor(elastic10_tensors[3][1])
class GeminiStoreConfig(Object):
    km = 1000
    d = dict(
        id="Store_Dir",
        ncomponents=10,
        component_scheme="elastic10",
        stored_quantity=100,
        sample_rate=10,
        receiver_depth=0 * km,
        source_depth_min=10 * km,
        source_depth_max=20 * km,
        source_depth_delta=10 * km,
        distance_min=100 * km,
        distance_max=1000 * km,
        distance_delta=10 * km,
        earthmodel_1d="cake.load_model()",
        modelling_code_id="gemini",
        tabulated_phases=[
            gf.meta.TPDef(id="begin", definition="p,P,p\\,P\\,Pv_(cmb)p"),
            gf.meta.TPDef(id="end", definition="2.5"),
            gf.meta.TPDef(id="P", definition="!P"),
            gf.meta.TPDef(id="S", definition="!S"),
            gf.meta.TPDef(id="p", definition="!p"),
            gf.meta.TPDef(id="s", definition="!s"),
        ],
    )
    # if config_params is not None:
    #        d.update(config_params)

    configA = gf.meta.ConfigTypeA()
    dump_config(
        configA, filepath=Path(__file__).parent / "gemini_store_configA.yaml"
    )


GeminiStoreConfig.dump_config()  # create default config file for GEMINI store
