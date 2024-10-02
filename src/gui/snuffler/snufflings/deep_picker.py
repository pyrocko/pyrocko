# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from typing import Any, Dict, Tuple
import logging
from pyrocko.trace import Trace, NoData
from pyrocko import obspy_compat as compat

from ..snuffling import Param, Snuffling, Choice, Switch
from ..marker import PhaseMarker

try:
    import seisbench.models as sbm
    import torch
    from obspy import Stream
except ImportError:
    sbm = None
    Stream = None


h = 3600.0
logger = logging.getLogger(__name__)


detectionmethods = (
    'original', 'ethz', 'instance', 'scedc', 'stead', 'geofon',
    'neic', 'cascadia', 'cms', 'jcms', 'jcs', 'jms', 'mexico',
    'nankai', 'san_andreas')

networks = (
    'PhaseNet', 'EQTransformer', 'GPD', 'LFEDetect'
)


MODEL_CACHE: Dict[str, Any] = {}


def get_blinding_samples(model: "sbm.WaveformModel") -> Tuple[int, int]:
    try:
        return model.default_args["blinding"]
    except KeyError:
        return model._annotate_args["blinding"][1]


class DeepDetector(Snuffling):

    def __init__(self, *args, **kwargs):
        Snuffling.__init__(self, *args, **kwargs)
        self.training_model = 'original'
        try:
            from seisbench.util.annotations import PickList
            self.old_method: str = None
            self.old_network: str = None
            self.pick_list: PickList | None = PickList()

        except ImportError:
            self.old_method = None
            self.old_network = None
            self.pick_list = None

    def help(self) -> str:
        return '''
<html>
<head>
<style type="text/css">
    body { margin-left:10px };
</style>
</head>
<body>
<h1 align="center">SeisBench: ML Picker</h1>
<p>
    Automatic detection of P- and S-Phases in the given traces, using various
    pre-trained ML models from SeisBench.<br/>
<p>
<b>Parameters:</b><br />
    <b>&middot; P threshold</b>
    -  Define a trigger threshold for the P-Phase detection <br />
    <b>&middot; S threshold</b>
    -  Define a trigger threshold for the S-Phase detection <br />
    <b>&middot; Detection method</b>
    -  Choose the pretrained model, used for detection. <br />
</p>
<p>
    <span style="color:red">P-Phases</span> are marked with red markers, <span
    style="color:green>S-Phases</span>  with green markers.
<p>
    More information about SeisBench can be found <a
    href="https://seisbench.readthedocs.io/en/stable/index.html">on the
    seisbench website</a>.
</p>
</body>
</html>
        '''

    def setup(self) -> None:
        self.set_name('SeisBench: ML Picker')
        self.add_parameter(
                    Choice(
                        'Network',
                        'network',
                        default='PhaseNet',
                        choices=networks,
                    )
                )

        self.add_parameter(
            Choice(
                'Training model',
                'training_model',
                default='original',
                choices=detectionmethods,
            )
        )

        self.add_parameter(
            Param(
                'P threshold',
                'p_threshold',
                default=self.get_default_threshold('P'),
                minimum=0.0,
                maximum=1.0,
            )
        )
        self.add_parameter(
            Param(
                'S threshold', 's_threshold',
                default=self.get_default_threshold('S'),
                minimum=0.0,
                maximum=1.0,
            )
        )

        self.add_parameter(
            Switch(
                'Show annotation traces', 'show_annotation_traces',
                default=False
            )
        )
        self.add_parameter(
            Switch(
                'Use predefined filters', 'use_predefined_filters',
                default=True
            )
        )
        self.add_parameter(
            Param(
                'Rescaling factor', 'scale_factor',
                default=1.0, minimum=0.1, maximum=10.0
            )
        )

        self.set_live_update(True)

    def get_default_threshold(self, phase: str) -> float:
        if sbm is None or self.training_model == 'original':
            return 0.3

        else:
            model = self.get_model(self.network, self.training_model)
            if phase == 'S':
                return model.default_args['S_threshold']
            elif phase == 'P':
                return model.default_args['P_threshold']

    def get_model(self, network: str, model: str) -> Any:
        if sbm is None:
            raise ImportError(
                'SeisBench is not installed. Install to use this plugin.')

        if model in MODEL_CACHE:
            return MODEL_CACHE[(network, model)]
        seisbench_model = eval(f'sbm.{network}.from_pretrained("{model}")')
        try:
            seisbench_model = seisbench_model.to('cuda')
        except (RuntimeError, AssertionError):
            logger.info('CUDA not available, using CPU')
            pass
        seisbench_model.eval()
        try:
            seisbench_model = torch.compile(
                seisbench_model, mode='max-autotune')
        except RuntimeError:
            logger.info('Torch compile failed')
            pass
        MODEL_CACHE[(network, model)] = seisbench_model
        return seisbench_model

    def set_default_thresholds(self) -> None:
        if self.training_model == 'original':
            self.set_parameter('p_threshold', 0.3)
            self.set_parameter('s_threshold', 0.3)
        elif self.training_model in detectionmethods[-8:]:
            self.set_parameter('p_threshold', 0.3)
            self.set_parameter('s_threshold', 0.3)
        else:
            self.set_parameter('p_threshold', self.get_default_threshold('P'))
            self.set_parameter('s_threshold', self.get_default_threshold('S'))

    def content_changed(self) -> None:
        if self._live_update:
            self.call()

    def panel_visibility_changed(self, visible: bool) -> None:
        if visible:
            self._connect_signals()
        else:
            self._disconnect_signals()

    def _connect_signals(self) -> None:
        viewer = self.get_viewer()
        viewer.pile_has_changed_signal.connect(self.content_changed)
        viewer.frequency_filter_changed.connect(self.content_changed)

    def _disconnect_signals(self) -> None:
        viewer = self.get_viewer()
        viewer.pile_has_changed_signal.disconnect(self.content_changed)
        viewer.frequency_filter_changed.disconnect(self.content_changed)

    def call(self) -> None:
        self.cleanup()
        self.adjust_thresholds()
        model = self.get_model(self.network, self.training_model)

        tinc = 300.0
        tpad = 1.0

        tpad_filter = 0.0
        if self.use_predefined_filters:
            fmin = self.get_viewer().highpass
            tpad_filter = 0.0 if fmin is None else 2.0/fmin

        for batch in self.chopper_selected_traces(
            tinc=tinc,
            tpad=tpad + tpad_filter,
            fallback=True,
            mode='visible',
            progress='Calculating SeisBench detections...',
            responsive=True,
            style='batch',
        ):
            traces = batch.traces

            if not traces:
                continue

            wmin, wmax = batch.tmin, batch.tmax

            for tr in traces:
                tr.meta = {'tabu': True}

            if self.use_predefined_filters:
                traces = [self.apply_filter(tr, tpad_filter) for tr in traces]

            stream = Stream([compat.to_obspy_trace(tr) for tr in traces])
            tr_starttimes: Dict[str, float] = {}
            if self.scale_factor != 1:
                for tr in stream:
                    tr.stats.sampling_rate /= self.scale_factor
                    s = tr.stats
                    tr_nsl = '.'.join((s.network, s.station, s.location))
                    tr_starttimes[tr_nsl] = s.starttime.timestamp

            output_classify = model.classify(
                stream,
                P_threshold=self.p_threshold,
                S_threshold=self.s_threshold,
            )

            if self.show_annotation_traces:
                output_annotation = model.annotate(
                    stream,
                    P_threshold=self.p_threshold,
                    S_threshold=self.s_threshold,
                )

                if self.scale_factor != 1:
                    for tr in output_annotation:
                        tr.stats.sampling_rate *= self.scale_factor
                        blinding_samples = max(get_blinding_samples(model))
                        blinding_seconds = (blinding_samples / 100.0) * \
                            (1.0 - 1 / self.scale_factor)
                        tr.stats.starttime -= blinding_seconds

                traces_raw = compat.to_pyrocko_traces(output_annotation)
                ano_traces = []
                for tr in traces_raw:
                    if tr.channel[-1] != 'N':
                        tr = tr.copy()
                        tr.chop(wmin, wmax)
                        tr.meta = {'tabu': True, 'annotation': True}
                        ano_traces.append(tr)

                self._disconnect_signals()
                self.add_traces(ano_traces)
                self._connect_signals()

            self.pick_list = output_classify.picks

            markers = []
            for pick in output_classify.picks:

                tpeak = pick.peak_time.timestamp
                if self.scale_factor != 1:
                    tr_starttime = tr_starttimes[pick.trace_id]
                    tpeak = tr_starttime + \
                        (pick.peak_time.timestamp - tr_starttime) \
                        / self.scale_factor

                if wmin <= tpeak < wmax:
                    codes = tuple(pick.trace_id.split('.')) + ('*',)
                    markers.append(PhaseMarker(
                        [codes],
                        tmin=tpeak,
                        tmax=tpeak,
                        kind=0 if pick.phase == 'P' else 1,
                        phasename=pick.phase,
                        incidence_angle=pick.peak_value,
                    ))

            self.add_markers(markers)

    def adjust_thresholds(self) -> None:
        method = self.get_parameter_value('training_model')
        network = self.get_parameter_value('network')
        if method != self.old_method or network != self.old_network:
            if (network == 'LFEDetect') \
                    and (method not in detectionmethods[-8:]):
                logger.info(
                    'The selected model is not compatible with LFEDetect '
                    'please select a model from the last 8 models in the '
                    'list. Default is cascadia.')
                self.set_parameter('training_model', 'cascadia')
            elif (network != 'LFEDetect') \
                    and (method in detectionmethods[-8:]):
                logger.info(
                    'The selected model is not compatible with the selected '
                    'network. Default is original.')
                self.set_parameter('training_model', 'original')
            self.set_default_thresholds()
            self.old_method = method
            self.old_network = network

    def apply_filter(self, tr: Trace, tcut: float) -> Trace:
        viewer = self.get_viewer()
        if viewer.lowpass is not None:
            tr.lowpass(4, viewer.lowpass, nyquist_exception=False)
        if viewer.highpass is not None:
            tr.highpass(4, viewer.highpass, nyquist_exception=False)
        try:
            tr.chop(tr.tmin + tcut, tr.tmax - tcut)
        except NoData:
            pass
        return tr


def __snufflings__():
    return [DeepDetector()]
