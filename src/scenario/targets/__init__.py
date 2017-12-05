from .base import *  # noqa
from .station import *  # noqa
from .waveform import *  # noqa
from .insar import *  # noqa
from .gnss_campaign import *  # noqa

AVAILABLE_TARGETS =\
    [RandomStationGenerator, WaveformGenerator, InSARGenerator,  # noqa
     GNSSCampaignGenerator]  # noqa
