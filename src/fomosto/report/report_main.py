# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

import os
import sys
import shutil
import subprocess
import re
import io
import base64
import datetime
from collections import OrderedDict
from tempfile import NamedTemporaryFile, mkdtemp
from string import Template

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm, transforms

from pyrocko import gf, trace, cake, util, plot
from pyrocko.response import DifferentiationResponse
from pyrocko.plot import beachball
from pyrocko.guts import load, Object, String, List, Float, Int, Bool, Dict
from pyrocko.gf import Source, Target
from pyrocko.gf.meta import OutOfBounds

from jinja2 import Environment, PackageLoader

guts_prefix = 'gft'
ex_path = os.path.dirname(os.path.abspath(sys.argv[0]))
ja_latex_env = Environment(block_start_string='\\BLOCK{',
                           block_end_string='}',
                           variable_start_string='\\VAR{',
                           variable_end_string='}',
                           comment_start_string='\\%{',
                           comment_end_string='}',
                           line_statement_prefix='\\-',
                           line_comment_prefix='%%',
                           trim_blocks=True,
                           loader=PackageLoader('pyrocko',
                                                'data/fomosto_report'))


class FomostoReportError(Exception):
    pass


class GreensFunctionError(FomostoReportError):
    pass


class FilterFrequencyError(GreensFunctionError):

    def __init__(self, frequency):
        Exception.__init__(self)
        self.frequency = frequency

    def __str__(self):
        return 'Cannot set {0} frequency to both an absolute and' \
               ' revlative value.'.format(self.frequency)


class SourceError(GreensFunctionError):

    def __init__(self, typ):
        Exception.__init__(self)
        self.type = typ

    def __str__(self):
        return 'Source type not currently supported: {0}'.format(self.type)


class SensorArray(Target):

    distance_min = Float.T()
    distance_max = Float.T()
    strike = Float.T()
    sensor_count = Int.T(default=50)

    __target_allowed = dir(Target)
    __sensorarray_allowed = ['distance_min', 'distance_max', 'strike',
                             'sensor_count', 'change_dists']

    def __init__(self, **kwargs):
        Object.__init__(self, **kwargs)

        self.validate_args(kwargs)
        self.__build(**kwargs)

    @classmethod
    def validate_args(cls, kwargs):
        ks = []
        for k in kwargs:
            if not (k in cls.__target_allowed or
                    k in cls.__sensorarray_allowed):
                ks.append(k)

        for i in ks:
            del kwargs[i]

    @classmethod
    def __validate_args_target(cls, kwargs):
        ks = []
        for k in kwargs:
            if k not in cls.__target_allowed:
                ks.append(k)

        for i in ks:
            del kwargs[i]

    def __build(self, **kwargs):
        if 'quantity' not in kwargs:
            kwargs['quantity'] = 'displacement'
        dists = np.linspace(self.distance_min, self.distance_max,
                            self.sensor_count)
        self.__validate_args_target(kwargs)
        self.sensors = [
            Target(north_shift=float(np.cos(np.radians(self.strike)) * dist),
                   east_shift=float(np.sin(np.radians(self.strike)) * dist),
                   **kwargs)
            for dist in dists]


class GreensFunctionTest(Object):

    __valid_properties = ['store_dir', 'pdf_dir', 'output_format',
                          'lowpass_frequency', 'rel_lowpass_frequency',
                          'highpass_frequency', 'rel_highpass_frequency',
                          'filter_order',
                          'plot_velocity', 'plot_everything']
    __notesize = 7.45
    __scalelist = [1, 5, 9.5, 19, 29]
    __has_phase = True

    store_dir = String.T()
    pdf_dir = String.T(default=os.path.expanduser('~'))
    output_format = String.T(optional=True, default='pdf')
    lowpass_frequency = Float.T(optional=True)
    highpass_frequency = Float.T(optional=True)
    rel_lowpass_frequency = Float.T(optional=True)
    rel_highpass_frequency = Float.T(optional=True)
    filter_order = Int.T(default=4)
    sensor_count = Int.T(default=50)
    plot_velocity = Bool.T(default=False)
    plot_everything = Bool.T(default=True)
    src_ids = List.T(String.T())
    sen_ids = List.T(String.T())
    sources = Dict.T(String.T(), Source.T())
    sensors = Dict.T(String.T(), SensorArray.T())
    trace_configs = List.T(List.T(String.T()), optional=True)

    @classmethod
    def __get_valid_arguments(cls, args):
        tdict = {}
        for k in args:
            if k in cls.__valid_properties:
                tdict[k] = args[k]
        return tdict

    def __init__(self, store_dir, **kwargs):
        Object.__init__(self, store_dir=store_dir, **kwargs)

        if self.lowpass_frequency and self.rel_lowpass_frequency:
            raise FilterFrequencyError('lowpass')
        if self.highpass_frequency and self.rel_highpass_frequency:
            raise FilterFrequencyError('highpass')

        if self.store_dir[-1] != '/':
            self.store_dir += '/'
        self.engine = gf.LocalEngine(store_dirs=[self.store_dir])
        ids = self.engine.get_store_ids()
        self.store_id = ids[0]
        self.store = self.engine.get_store(self.store_id)

        if self.rel_lowpass_frequency is not None:
            self.lowpass_frequency = self.store.config.sample_rate * \
                self.rel_lowpass_frequency

        if self.rel_highpass_frequency is not None:
            self.highpass_frequency = self.store.config.sample_rate * \
                self.rel_highpass_frequency

        self.rel_lowpass_frequency = None
        self.rel_highpass_frequency = None

        self.phases = '|'.join([x.id
                                for x in self.store.config.tabulated_phases])
        if self.phases == '':
            self.phase_ratio_string = r'\color{red}' \
                '{warning: store has no tabulated phases}'
        elif 'begin' in self.phases:
            self.phase_ratio_string = 'begin'
        else:
            self.phase_ratio_string = 'first({0})'.format(self.phases)

        if len(self.src_ids) == 0 and len(self.sources) > 0:
            self.src_ids = sorted(self.sources)
        if len(self.sen_ids) == 0 and len(self.sensors) > 0:
            self.sen_ids = sorted(self.sensors)

        self.traces = OrderedDict()

        if self.pdf_dir is None:
            self.pdf_dir = os.path.expanduser('~')
        if self.pdf_dir[-1] != '/':
            self.pdf_dir += '/'
        self.pdf_name = '{0}_{1}'.format(
            self.store_id, self.getFrequencyString(None))
        util.setup_logging()

        self.temp_dir = mkdtemp(prefix='gft_')
        self.message = None
        self.changed_depth = False
        self.changed_dist_min = False
        self.changed_dist_max = False

    def __addToMessage(self, msg):
        if self.message is None:
            self.message = '\nmessage(s) for store {0}:\n'. \
                format(self.store_id)
        self.message += msg + '\n'

    def __printMessage(self):
        if self.message is not None:
            print(self.message)

    def cleanup(self):
        shutil.rmtree(self.temp_dir)
        for i in ['.aux', '.log', '.out', '.toc']:
            path = '{0}{1}{2}'.format(self.pdf_dir, self.pdf_name, i)
            if os.path.exists(path):
                os.remove(path)

    def getSourceString(self, sid):
        if sid not in self.sources:
            return ''
        src = self.sources[sid]
        if isinstance(src, gf.DCSource):
            typ = 'Double Couple'
            sstr = 'Source Type: {0}, Strike: {1:g}, Dip: {2:g}, Rake: {3:g}' \
                .format(typ, src.strike, src.dip, src.rake)
        elif isinstance(src, gf.RectangularExplosionSource):
            typ = 'Explosion'
            sstr = 'Source Type: {0}, Strike: {1:g}, Dip: {2:g}'.format(
                typ, src.strike, src.dip)
        else:
            typ = '{0}'.format(type(src)).split('.')[-1].split("'")[0]
            sstr = 'Source Type: {0}, see config page'.format(typ)
        return sstr

    def getSensorString(self, sid):
        if sid not in self.sensors:
            return ''
        sen_ar = self.sensors[sid]
        sensors = sen_ar.sensors

        targ = sensors[0]
        senstr = 'Sensor Strike: {0:g}, Azimuth: {1:g}, Dip: {2:g}'. \
            format(sen_ar.strike, targ.azimuth, targ.dip)
        return senstr

    def getFrequencyString(self, tid):
        lpf = self.lowpass_frequency
        hpf = self.highpass_frequency
        if tid is None:
            if lpf and hpf:
                return '{0:.4g}-{1:.4g}Hz'.format(lpf, hpf)
            elif lpf:
                return 'lowpass {0:.4g}Hz'.format(lpf)
            elif hpf:
                return 'highpass {0:.4g}Hz'.format(hpf)
            else:
                return 'unfiltered'

        if tid not in self.traces:
            return ''

        tdict = self.traces[tid]
        lpfa = tdict['lowpass_applied']
        hpfa = tdict['highpass_applied']
        if lpf and hpf and lpfa and hpfa:
            return '{0:.4g}-{1:.4g} [Hz]'.format(lpf, hpf)
        elif lpf and lpfa:
            return 'lowpass frequency: {0:.4g} [Hz]'.format(lpf)
        elif hpf and hpfa:
            return 'highpass frequency: {0:.4g} [Hz]'.format(hpf)
        else:
            return 'unfiltered'

    def getAnnotateString(self, src_id, sen_id, fac=None):
        tstr = 'Green\'s Function: {0}\n{1}\n{2}'.format(
            self.store_id, self.getSourceString(src_id),
            self.getSensorString(sen_id))
        if fac is not None:
            tstr += '\nAmplitude Scale Factor: {0:0.4g} [microns/km]'. \
                format(fac * 1e-9 / 1e3)
        return tstr

    def createSource(self, typ, depth, strike, dip, rake=None, lat=0., lon=0.,
                     north_shift=0., east_shift=0., change_dists=True,
                     **kwargs):
        stCfg = self.store.config
        stMin = stCfg.source_depth_min
        stMax = stCfg.source_depth_max
        if depth is None:
            depth = (stMax + stMin) / 2.
        elif depth < stMin or depth > stMax:
            if not change_dists:
                raise OutOfBounds('Source depth.')
            diff = abs(stMin - depth)
            if diff < stMax - depth:
                depth = stMin
            else:
                depth = stMax
            if not self.changed_depth:
                self.changed_depth = True
                self.__addToMessage(
                    'Source depth out of bounds. Changed to: {0:g}m.'.
                    format(depth))

        if typ.upper() in ('EX' 'EXPLOSION'):
            src = gf.RectangularExplosionSource(
                lat=lat,
                lon=lon,
                north_shift=north_shift,
                east_shift=east_shift,
                depth=depth,
                strike=strike,
                dip=dip)
        elif typ.upper() in ('DC' 'DOUBLE COUPLE'):
            src = gf.DCSource(
                lat=lat,
                lon=lon,
                north_shift=north_shift,
                east_shift=east_shift,
                depth=depth,
                strike=strike,
                dip=dip,
                rake=rake)
        else:
            raise SourceError(typ)

        srcstr = '{0}.{1:.2g}.{2:.2g}.{3:.2g}'.format(
            typ[0:2], depth, strike, dip)
        tstr = srcstr
        tint = 96
        while tstr in self.sources:
            tint += 1
            tstr = srcstr + chr(tint)

        self.sources[tstr] = src
        self.src_ids.append(tstr)
        return tstr

    def createSensors(self, distance_min=None, distance_max=None,
                      change_dists=True, **kwargs):
        stCfg = self.store.config
        stMin = stCfg.distance_min
        stMax = stCfg.distance_max
        if distance_min is None:
            distance_min = stMin
        elif distance_min < stMin or distance_min > stMax:
            if not change_dists:
                raise OutOfBounds('Minimum sensor distance.')
            distance_min = stMin
            if not self.changed_dist_min:
                self.changed_dist_min = True
                self.__addToMessage(
                    'Sensor minimum distance out of bounds. Changed to allowed'
                    ' minimum {0:g}m.'.format(stMin))

        if distance_max is None:
            distance_max = stMax
        elif (distance_max > stMax or distance_max < stMin):
            if not change_dists:
                raise OutOfBounds('Maximum sensor distance')
            distance_max = stMax
            if not self.changed_dist_max:
                self.changed_dist_max = True
                self.__addToMessage(
                    'Sensor maximum distance out of bounds. Changed to allowed'
                    ' maximum {0:g}m.'.format(stMax))

        if change_dists and distance_min == distance_max:
            distance_min = stMin
            distance_max = stMax
            self.__addToMessage(
                'Sensor minimum and maximum distance equal. Changed to'
                ' allowed minimum {0:g}m and maximum {1:g}m.'.format(stMin,
                                                                     stMax))

        sen_ar = SensorArray(distance_min=distance_min,
                             distance_max=distance_max, **kwargs)
        senstr = '{0}.{1:.2g}'.format('.'.join(sen_ar.codes), sen_ar.strike)
        tstr = senstr
        tint = 96
        while tstr in self.sensors:
            tint += 1
            tstr = senstr + chr(tint)

        self.sensors[tstr] = sen_ar
        self.sen_ids.append(tstr)
        return tstr

    def createDisplacementTraces(self, src_id='all', sen_id='all'):
        for sr_id in self.src_ids:
            if sr_id not in self.sources:
                continue
            if not (src_id == 'all' or sr_id == src_id):
                continue
            for sn_id in self.sen_ids:
                if sn_id not in self.sensors:
                    continue
                if not (sen_id == 'all' or sn_id == sen_id):
                    continue
                try:
                    response = self.engine.process(
                        self.sources[sr_id],
                        self.sensors[sn_id].sensors)
                    trcs = response.pyrocko_traces()
                    tstr = '{0}|{1}'.format(sr_id, sn_id)
                    if tstr not in self.traces:
                        self.traces[tstr] = {}
                    tdict = self.traces[tstr]
                    tdict['displacement_traces'] = trcs
                    mina, maxa, minb, maxb, ratio = \
                        self.__tracesMinMax(trcs, sr_id, sn_id)
                    if ratio != 0.:
                        tdict['displacement_spectra'] = [
                            trc.spectrum() for trc in trcs]
                    tdict['lowpass_applied'] = False
                    tdict['highpass_applied'] = False
                    tdict['displacement_ratio'] = ratio
                    tdict['displacement_scale'] = max(abs(mina), abs(maxa))
                except IndexError:
                    self.__addToMessage(
                        'warning: IndexError: no traces created for'
                        ' source-sensor combination: {0} - {1}. Try increasing'
                        ' the sensor minimum distance.'.format(
                            sr_id, sn_id))

    def createVelocityTraces(self, trc_id='all'):
        for tid in self.traces:
            ids = trc_id.split('|')
            if len(ids) == 2:
                src_id, sen_id = ids
            else:
                src_id = sen_id = None
            if not (trc_id == 'all' or
                    ((src_id == 'all' or src_id in self.sources) and
                     (sen_id == 'all' or sen_id in self.sensors))):
                continue
            tdict = self.traces[tid]
            if 'displacement_traces' not in tdict:
                continue
            trcs = [trc.transfer(
                transfer_function=DifferentiationResponse())
                for trc in tdict['displacement_traces']]
            tdict['velocity_traces'] = trcs
            src_id, sen_id = tid.split('|')
            mina, maxa, minb, maxb, ratio = \
                self.__tracesMinMax(trcs, src_id, sen_id)
            if ratio != 0.:
                tdict['velocity_spectra'] = [
                    trc.spectrum() for trc in trcs]
            tdict['velocity_ratio'] = ratio
            tdict['velocity_scale'] = max(abs(mina), abs(maxa))

    def getPhaseArrivals(self, trc_id='all', phase_ids='all'):
        for tid in self.traces:
            if trc_id == 'all' or trc_id == tid:
                self.__setPhaseArrivals(tid, phase_ids)

    def __setPhaseArrivals(self, trc_id, phase_ids='all'):
        st = self.store
        src_id, sen_id = trc_id.split('|')
        src = self.sources[src_id]
        tdict = self.traces[trc_id]
        if 'phase_arrivals' not in tdict:
            tdict['phase_arrivals'] = {}
        phdict = tdict['phase_arrivals']
        for pid in self.phases.split('|'):
            if pid == '':
                continue
            if phase_ids == 'all' or pid == phase_ids:
                dists = []
                times = []
                for sen in self.sensors[sen_id].sensors:
                    time = st.t(pid, src, sen)
                    if time is None:
                        continue
                    dists.append(src.distance_to(sen))
                    times.append(time)

                if len(times) > 0:
                    phdict[pid] = (times, dists)

    def applyFrequencyFilters(self, trc_id='all'):
        dispname = 'displacement_traces'
        velname = 'velocity_traces'
        for tid in self.traces:
            if trc_id == 'all' or trc_id == tid:
                tdict = self.traces[tid]
                if self.lowpass_frequency:
                    if dispname in tdict:
                        for trc in tdict[dispname]:
                            trc.lowpass(self.filter_order,
                                        self.lowpass_frequency, demean=False)
                            tdict['lowpass_applied'] = True

                    if velname in tdict:
                        for trc in tdict[velname]:
                            trc.lowpass(self.filter_order,
                                        self.lowpass_frequency, demean=False)
                        tdict['lowpass_applied'] = True

                if self.highpass_frequency:
                    if dispname in tdict:
                        for trc in tdict[dispname]:
                            trc.highpass(self.filter_order,
                                         self.highpass_frequency, demean=False)
                        tdict['highpass_applied'] = True

                    if velname in tdict:
                        for trc in tdict[velname]:
                            trc.highpass(self.filter_order,
                                         self.highpass_frequency, demean=False)
                        tdict['highpass_applied'] = True

                sr_id, sn_id = tid.split('|')
                if dispname in tdict:
                    trcs = tdict[dispname]
                    mina, maxa, minb, maxb, ratio = \
                        self.__tracesMinMax(trcs, sr_id, sn_id)
                    tdict['displacement_ratio'] = ratio
                    tdict['displacement_scale'] = max(abs(mina), abs(maxa))
                if velname in tdict:
                    trcs = tdict[velname]
                    mina, maxa, minb, maxb, ratio = \
                        self.__tracesMinMax(trcs, sr_id, sn_id)
                    tdict['velocity_ratio'] = ratio
                    tdict['velocity_scale'] = max(abs(mina), abs(maxa))

    def __createOutputDoc(self, artefacts, chapters,
                          gft2=None, together=False):

        str_id = self.store_id
        is_tex = self.output_format == 'pdf'
        if is_tex:
            file_type = 'tex'
            dir = self.temp_dir
            fstr_id = self.__formatLatexString(str_id)
        else:
            file_type = self.output_format
            dir = self.pdf_dir
            fstr_id = str_id

        temp = ja_latex_env.get_template('gfreport.{0}'.format(file_type))
        out = '{0}/{1}.{2}'.format(dir, self.pdf_name, file_type)
        config = self.store.config.dump()

        if together:
            tpath = self.__createVelocityProfile(gft2)
            if is_tex:
                img_ttl = r'{0} \& {1}'.format(
                    fstr_id, gft2.__formatLatexString(gft2.store_id))
            else:
                img_ttl = r'{0} & {1}'.format(str_id, gft2.store_id)
        else:
            tpath = self.__createVelocityProfile()
            img_ttl = fstr_id

        info = [(fstr_id, config, tpath, img_ttl)]

        if gft2 is not None:
            if is_tex:
                fstr_id = self.__formatLatexString(gft2.store_id)
                str_id = r'{0} \& {1}'.format(str_id, gft2.store_id)
            else:
                fstr_id = gft2.store_id
                str_id = r'{0} & {1}'.format(str_id, gft2.store_id)

            config = gft2.store.config.dump()
            if together:
                tpath = ''
            else:
                tpath = gft2.__createVelocityProfile()

            info += [(fstr_id, config, tpath, fstr_id)]

        with open(out, 'w') as f:
            if is_tex:
                f.write(temp.render(
                    rpt_id=self.__formatLatexString(str_id), str_info=info,
                    artefacts=artefacts, chapters=chapters,
                    config=config, headings='headings'))
            elif file_type == 'html':
                f.write(temp.render(
                    rpt_id=str_id, str_info=info, artefacts=artefacts,
                    chapters=chapters, config=config,
                    date=datetime.datetime.now().strftime('%B %d, %Y')))

        if is_tex:
            pro_call = ['pdflatex', '-output-directory', self.pdf_dir, out,
                        '-interaction', 'nonstop']
            try:
                subprocess.call(pro_call)
                subprocess.call(pro_call)
                subprocess.call(pro_call)
            except OSError:
                raise FomostoReportError(
                    'Cannot run "pdflatex" executable. Is it installed?')

        self.cleanup()
        if gft2 is not None:
            gft2.cleanup()

    @staticmethod
    def __formatLatexString(string):
        rep = {'_': r'\_', r'\n': r'\\', '|': r'\textbar '}
        rep = {re.escape(k): v for k, v in rep.items()}
        pat = re.compile('|'.join(rep.keys()))
        return pat.sub(lambda m: rep[re.escape(m.group(0))], string)

    def createOutputDoc(self, *trc_ids):
        trcs = self.traces
        if len(trc_ids) == 0:
            trc_ids = trcs.keys()
        artefacts = self.__getArtefactPageInfo(trc_ids)
        chapters = []
        if self.plot_everything:
            for tid in trc_ids:
                if tid not in trcs:
                    continue
                src_id, sen_id = tid.split('|')
                ttl = '{0}, {1}'.format(
                    self.getSourceString(src_id),
                    self.getSensorString(sen_id))
                tdict = trcs[tid]
                img_data = []
                href = r'\hypertarget{${trc_id}|${str_id}|${type}}{}'
                trcname = 'displacement_traces'
                if trcname in tdict:
                    hstr = Template(href).substitute(
                        trc_id=tid, str_id=self.store_id, type='d')
                    figs = self.__createTraceFigures(tid, trcname)
                    fig = figs[0]
                    img_data.extend([(self.__getFigureTitle(fig), hstr,
                                      self.__saveTempFigure(fig))])
                    img_data.extend([('', '', self.__saveTempFigure(x))
                                     for x in figs[1:]])

                if 'displacement_spectra' in tdict:
                    fig = self.__createMaxAmpFigure(tid, trcname)
                    img_data.append((self.__getFigureTitle(fig), '',
                                     self.__saveTempFigure(fig)))

                    fig = self.__createSpectraFigure(tid,
                                                     'displacement_spectra')
                    img_data.append((self.__getFigureTitle(fig), '',
                                     self.__saveTempFigure(fig)))

                trcname = 'velocity_traces'
                if self.plot_velocity and trcname in tdict:
                    hstr = Template(href).substitute(
                        trc_id=tid, str_id=self.store_id, type='v')
                    figs = self.__createTraceFigures(tid, trcname)
                    fig = figs[0]
                    img_data.extend([(self.__getFigureTitle(fig), hstr,
                                      self.__saveTempFigure(fig))])
                    img_data.extend([('', '', self.__saveTempFigure(x))
                                     for x in figs[1:]])

                    if 'velocity_spectra' in tdict:
                        fig = self.__createMaxAmpFigure(tid, trcname)
                        img_data.append((self.__getFigureTitle(fig), '',
                                         self.__saveTempFigure(fig)))

                        fig = self.__createSpectraFigure(tid,
                                                         'velocity_spectra')
                        img_data.append((self.__getFigureTitle(fig), '',
                                         self.__saveTempFigure(fig)))
                if self.output_format == 'pdf':
                    src_str = self.__formatLatexString(
                        self.sources[src_id].dump())
                    sen_str = self.__formatLatexString(
                        self.sensors[sen_id].dump())
                else:
                    src_str = self.sources[src_id].dump()
                    sen_str = self.sensors[sen_id].dump()
                chapters.append([ttl, src_str, sen_str, img_data])
        if self.output_format == 'pdf':
            self.__createOutputDoc(
                [[self.__formatLatexString(self.store_id),
                  self.__formatLatexString(self.phase_ratio_string),
                  artefacts]], chapters)
        elif self.output_format == 'html':
            self.__createOutputDoc(
                [[self.store_id, self.phase_ratio_string, artefacts]],
                chapters)

    @classmethod
    def createComparisonOutputDoc(cls, gft1, gft2,
                                  *trc_ids, **kwargs):
        # only valid kwargs is 'together'
        if 'together' in kwargs:
            together = kwargs['together']
        else:
            together = True

        if len(trc_ids) == 0:
            trc_ids = gft1.traces.keys()

        tname = '{0}-{1}{2}{3}'.format(
            gft1.store_id, gft2.store_id, '_together_' if together else '_',
            gft1.getFrequencyString(None))
        gft1.pdf_name = tname
        gft2.pdf_name = tname

        trcs1 = gft1.traces
        trcs2 = gft2.traces

        art1 = gft1.__getArtefactPageInfo(trc_ids)
        art2 = gft2.__getArtefactPageInfo(trc_ids, gft1.store_id)
        chapters = []
        if gft1.plot_everything:
            for tid in trc_ids:
                if tid not in trcs1 or tid not in trcs2:
                    continue
                src_id, sen_id = tid.split('|')
                ttl = '{0}, {1}'.format(
                    gft1.getSourceString(src_id), gft1.getSensorString(sen_id))
                tdict1 = trcs1[tid]
                tdict2 = trcs2[tid]
                img_data = []
                href = r'\hypertarget{${trc_id}|${str_id}|${type}}{}'
                trcname = 'displacement_traces'
                tstr = 'displacement_spectra'
                if trcname in tdict1 and trcname in tdict2:
                    hstr = Template(href).substitute(
                        trc_id=tid, str_id=gft1.store_id, type='d')
                    figs = cls.__createComparisonTraceFigures(
                        gft1, gft2, tid, trcname, together)
                    fig = figs[0]
                    img_data.extend([(gft1.__getFigureTitle(fig), hstr,
                                      gft1.__saveTempFigure(fig))])
                    img_data.extend([('', '', gft1.__saveTempFigure(x))
                                     for x in figs[1:]])

                if tstr in tdict1 and tstr in tdict2:
                    fig = gft1.__createComparisonMaxAmpFigure(
                        gft1, gft2, tid, trcname, together)
                    img_data.append((gft1.__getFigureTitle(fig), '',
                                     gft1.__saveTempFigure(fig)))

                trcname = tstr
                if not together:
                    if trcname in tdict1 and trcname in tdict2:
                        fig = cls.__createComparissonSpectraFigure(
                            gft1, gft2, tid, trcname)
                        img_data.append((gft1.__getFigureTitle(fig), '',
                                         gft1.__saveTempFigure(fig)))

                trcname = 'velocity_traces'
                tstr = 'velocity_spectra'
                if gft1.plot_velocity and gft2.plot_velocity and \
                        trcname in tdict1 and trcname in tdict2:

                    hstr = Template(href).substitute(
                        trc_id=tid, str_id=gft1.store_id, type='v')
                    figs = cls.__createComparisonTraceFigures(
                        gft1, gft2, tid, trcname, together)
                    fig = figs[0]
                    img_data.extend([(gft1.__getFigureTitle(fig), hstr,
                                      gft1.__saveTempFigure(fig))])
                    img_data.extend([('', '', gft1.__saveTempFigure(x))
                                     for x in figs[1:]])

                    if tstr in tdict1 and tstr in tdict2:
                        fig = gft1.__createComparisonMaxAmpFigure(
                            gft1, gft2, tid, trcname, together)
                        img_data.append((gft1.__getFigureTitle(fig), '',
                                         gft1.__saveTempFigure(fig)))

                    if not together:
                        if tstr in tdict1 and tstr in tdict2:
                            fig = cls.__createComparissonSpectraFigure(
                                gft1, gft2, tid, tstr)
                            img_data.append((gft1.__getFigureTitle(fig), '',
                                             gft1.__saveTempFigure(fig)))

                if gft1.output_format == 'pdf':
                    src_str = gft1.__formatLatexString(
                        gft1.sources[src_id].dump())
                    sen_str = gft1.__formatLatexString(
                        gft1.sensors[sen_id].dump())
                else:
                    src_str = gft1.sources[src_id].dump()
                    sen_str = gft1.sensors[sen_id].dump()

                chapters.append([ttl, src_str, sen_str, img_data])

        if gft1.output_format == 'pdf':
            gft1.__createOutputDoc(
                [[gft1.__formatLatexString(gft1.store_id),
                  gft1.__formatLatexString(gft1.phase_ratio_string), art1],
                 [gft2.__formatLatexString(gft2.store_id),
                  gft2.__formatLatexString(gft2.phase_ratio_string), art2]],
                chapters, gft2=gft2, together=together)
        elif gft1.output_format == 'html':
            gft1.__createOutputDoc(
                [[gft1.store_id, gft1.phase_ratio_string, art1],
                 [gft2.store_id, gft2.phase_ratio_string, art2]],
                chapters, gft2=gft2, together=together)

    def __getArtefactPageInfo(self, trc_ids, str_id=None):
        is_tex = self.output_format == 'pdf'
        if is_tex:
            sp = r'\ '*6
        else:
            sp = r' '*6

        ratio_tol = 0.4
        artefacts = []
        # list of [<trace string>, <ratio text color>, <ratio string>]
        if str_id is None:
            str_id = self.store_id
        for tid in trc_ids:
            if tid not in self.traces:
                continue
            src_id, sen_id = tid.split('|')
            tdict = self.traces[tid]
            ttl_str = r'%s, %s' % (self.getSourceString(src_id),
                                   self.getSensorString(sen_id))
            if self.output_format == 'pdf':
                ttl_str = r'\textbr{%s}' % ttl_str

            artefacts.append([ttl_str, 'black', ''])

            chCode = self.sensors[sen_id].codes[3]
            ttl_str = r'{0}{1} traces ({2})'.format(
                sp, 'Displacement', chCode)
            ratio = tdict['displacement_ratio']
            color = ('red' if ratio == 0. or ratio > ratio_tol else 'black')
            rat_str = '{0:0.3f}'.format(ratio)
            if is_tex:
                tex = r'\hyperlink{${tid}|${str_id}|${type}}{${title}}'
                ttl_str = Template(tex).substitute(
                    tid=tid, str_id=str_id, type='d', title=ttl_str)

            artefacts.append([ttl_str, color, rat_str])

            if self.plot_velocity and 'velocity_traces' in tdict:
                ttl_str = r'{0}{1} traces ({2})'.format(sp, 'Velocity', chCode)
                ratio = tdict['velocity_ratio']
                color = ('red' if ratio == 0. or ratio > ratio_tol
                         else 'black')
                rat_str = '{0:0.3f}'.format(ratio)
                if is_tex:
                    tex = r'\hyperlink{${tid}|${str_id}|${type}}{${title}}'
                    ttl_str = Template(tex).substitute(
                        tid=tid, str_id=str_id, type='v', title=ttl_str)

                artefacts.append([ttl_str, color, rat_str])
            artefacts.append(['', 'black', ''])
        return artefacts

    def __createTraceFigures(self, trc_id, trc_type):
        figs = []
        for i in self.__scalelist:
            fig = self.__setupFigure(trc_type, 1)
            ax, ax2 = fig.axes
            zerotrc = self.__drawTraceData(trc_id, trc_type, ax, ax2, i,
                                           (0.01, 0.01))
            self.__drawBeachball(trc_id, ax)
            figs.append(fig)
            if zerotrc:
                break
        return figs

    @staticmethod
    def __createComparisonTraceFigures(
            gfTest1, gfTest2, trc_id, trc_type, together):

        tdict = gfTest1.traces[trc_id]
        sc1 = tdict['{0}_scale'.format(trc_type.split('_')[0])]

        tdict = gfTest2.traces[trc_id]
        sc2 = tdict['{0}_scale'.format(trc_type.split('_')[0])]

        absmax = (sc1 + sc2) / 2.
        figs = []
        for i in gfTest1.__scalelist:
            if together:
                fig = gfTest1.__setupFigure(trc_type, 1)
                ax, ax2 = fig.axes
                zerotrc1 = gfTest1.__drawTraceData(
                    trc_id, trc_type, ax, ax2, i, (0.01, 0.01),
                    showphases=False, absmax=absmax)
                zerotrc2 = gfTest2.__drawTraceData(
                    trc_id, trc_type, ax, ax2, i, (0.92, 0.01),
                    color=plot.mpl_color('scarletred2'), hor_ali='right',
                    showphases=False, absmax=absmax)
                gfTest1.__drawBeachball(trc_id, ax)
            else:
                fig = gfTest1.__setupFigure(trc_type, 2)
                ax, ax2, ax3, ax4 = fig.axes
                zerotrc1 = gfTest1.__drawTraceData(
                    trc_id, trc_type, ax, ax2, i, (0.01, 0.01), absmax=absmax)
                gfTest1.__drawBeachball(trc_id, ax)
                zerotrc2 = gfTest2.__drawTraceData(
                    trc_id, trc_type, ax3, ax4, i, (0.98, 0.01),
                    hor_ali='right', absmax=absmax)
                gfTest2.__drawBeachball(trc_id, ax3)

            figs.append(fig)
            if zerotrc1 and zerotrc2:
                break
        return figs

    def __drawTraceData(self, trc_id, trc_type, lfax, rtax, yfac, anno_pt,
                        color='black', hor_ali='left', showphases=True,
                        absmax=None):
        new_axis = lfax.get_ylim() == (0.0, 1.0)
        tdict = self.traces[trc_id]
        trcs = tdict[trc_type]
        phdict = None
        if showphases and 'phase_arrivals' in tdict:
            phdict = tdict['phase_arrivals']
            if len(phdict) == 0:
                phdict = None

        times = trace.minmaxtime(trcs, key=lambda trc: None)[None]
        times = np.linspace(times[0], times[1], 10)
        diff = (times[1] - times[0]) / 10.

        src_id, sen_id = trc_id.split('|')
        chCode = self.sensors[sen_id].codes[3]
        dists = self.__getSensorDistances(src_id, sen_id)

        ysc = dists[1] - dists[0]

        zerotrc = False
        if absmax is None:
            absmax = tdict['{0}_scale'.format(trc_type.split('_')[0])]
        if absmax == 0.:
            absmax = 1.
            zerotrc = True

        ysc2 = ysc / absmax * yfac

        for i in range(len(self.sensors[sen_id].sensors)):
            trc = trcs[i]
            dist = dists[i]
            lfax.plot(trc.get_xdata(),
                      (dist + trc.get_ydata() * ysc2) * cake.m2d, '-',
                      color=color)

        if phdict is not None:
            for i, pid in enumerate(phdict):
                ts, ds = phdict[pid]
                ds = [d * cake.m2d for d in ds]
                lfax.plot(ts, ds,
                          label='{0}-wave'.format(pid), marker='o',
                          markersize=3,
                          color=plot.to01(plot.graph_colors[i % 7]))

            lfax.legend(loc='lower right', shadow=False, prop={'size': 10.})

        xmin = times[0] - diff
        xmax = times[-1] + diff
        dmin = dists[0]
        dmax = dists[-1]
        ymin = (dmin - ysc * 3.)
        ymax = (dmax + ysc * 3.)
        if new_axis:
            lfax.set_xlim(xmin, xmax)
            lfax.set_ylim(ymin * cake.m2d, ymax * cake.m2d)
            rtax.set_ylim(ymin / 1e3, ymax / 1e3)
        else:
            xlims = lfax.get_xlim()
            xlims = (min(xmin, xlims[0]), max(xmax, xlims[1]))
            lfax.set_xlim(xlims)

        lfax.set_title('{0} traces ({1}), {2}'.format(
            trc_type.split('_')[0].title(), chCode,
            self.getFrequencyString(trc_id)), y=1.015)
        lfax.annotate(self.getAnnotateString(src_id, sen_id, ysc2),
                      xy=anno_pt, fontsize=(self.__notesize),
                      xycoords='figure fraction', ha=hor_ali, color=color)
        if zerotrc:
            lfax.annotate("Zero amplitudes!\nSpectra will not\nbe plotted.",
                          xy=(0.001, 0.), fontsize=25, alpha=0.75,
                          rotation=0., xycoords='axes fraction',
                          color=plot.mpl_color('aluminium4'))

        self.__drawGrid(lfax)
        return zerotrc

    def __createSpectraFigure(self, trc_id, spc_id):
        fig = self.__setupFigure('spectra', 1)
        ax = fig.axes[0]
        self.__drawSpectraData(trc_id, ax, (0.01, 0.01), spc_id)
        return fig

    @staticmethod
    def __createComparissonSpectraFigure(gfTest1, gfTest2, trc_id, spc_id):
        fig = gfTest1.__setupFigure('spectra', 2)
        ax, ax2 = fig.axes
        gfTest1.__drawSpectraData(trc_id, ax, (0.01, 0.01), spc_id,
                                  show_cbar=False)
        gfTest2.__drawSpectraData(trc_id, ax2, (0.98, 0.01), spc_id,
                                  hor_ali='right')
        # evenly space the axes
        ax1, ax2, ax3 = fig.axes
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        fac = (pos2.x0 - pos1.x1) / 2.
        mid = (pos2.x1 + pos1.x0) / 2.
        wid = mid - fac - pos1.x0
        ax1.set_position([pos1.x0, pos1.y0, wid, pos1.height])
        ax2.set_position([mid + fac, pos2.y0, wid, pos2.height])
        return fig

    def __drawSpectraData(self, trc_id, ax, anno_pt, spc_id,
                          hor_ali='left', show_cbar=True):
        tdict = self.traces[trc_id]
        spcs = tdict[spc_id]
        clrs = np.linspace(0., 1., len(spcs))
        cmap = cm.jet
        src_id, sen_id = trc_id.split('|')
        chCode = self.sensors[sen_id].codes[3]
        dists = self.__getSensorDistances(src_id, sen_id)
        for idx, data in enumerate(spcs):
            f, v = data
            ax.plot(f, abs(v), color=cmap(clrs[idx]),
                    marker='x')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('Amplitude [{0}]'.format(
            'm' if spc_id.startswith('displacement') else 'm/s'))
        ax.set_xlabel('Frequency [Hz]')
        ax.set_title('{0} spectra for ({1})'.
                     format(spc_id.split('_')[0].title(), chCode), y=1.015)
        ax.annotate(self.getAnnotateString(src_id, sen_id),
                    xy=anno_pt, ha=hor_ali, fontsize=(self.__notesize),
                    xycoords='figure fraction')
        if show_cbar:
            tmin = min(dists) / 1e3
            tmax = max(dists) / 1e3
            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(vmin=tmin, vmax=tmax))
            sm.set_array(np.linspace(tmin, tmax,
                         self.sensors[sen_id].sensor_count))
            cbar = plt.colorbar(sm, shrink=0.95)
            cbar.ax.set_ylabel('Sensor distance [km]')

    def __createMaxAmpFigure(self, trc_id, trc_typ):
        fig = self.__setupFigure('maxamp_{0}'.format(trc_typ), 1)
        ax, ax2 = fig.axes
        self.__drawMaxAmpData(trc_id, ax, ax2, (0.01, 0.01), trc_typ)
        return fig

    @staticmethod
    def __createComparisonMaxAmpFigure(gfTest1, gfTest2, trc_id, trc_typ,
                                       together):
        if together:
            fig = gfTest1.__setupFigure('maxamp_{0}'.format(trc_typ), 1)
            ax, ax2 = fig.axes
            gfTest1.__drawMaxAmpData(trc_id, ax, ax2, (0.01, 0.01), trc_typ)
            gfTest2.__drawMaxAmpData(trc_id, ax, ax2, (0.92, 0.01), trc_typ,
                                     color=plot.mpl_color('scarletred2'),
                                     hor_ali='right')
        else:
            fig = gfTest1.__setupFigure('maxamp_{0}'.format(trc_typ), 2)
            ax, ax2, ax3, ax4 = fig.axes
            gfTest1.__drawMaxAmpData(trc_id, ax, ax2, (0.01, 0.01), trc_typ)
            gfTest2.__drawMaxAmpData(trc_id, ax3, ax4, (0.98, 0.01), trc_typ,
                                     hor_ali='right')
        return fig

    def __drawMaxAmpData(self, trc_id, btmax, topax, anno_pt, trc_typ,
                         color='black', hor_ali='left'):
        new_axis = btmax.get_ylim() == (0.0, 1.0)
        src_id, sen_id = trc_id.split('|')
        chCode = self.sensors[sen_id].codes[3]
        trcs = self.traces[trc_id][trc_typ]
        dists = [x / 1e3 for x in self.__getSensorDistances(src_id, sen_id)]
        before, after = self.__tracesMinMaxs(trcs, src_id, sen_id)
        amps = [max(abs(x[0]), abs(x[1])) for x in after]

        btmax.plot(dists, amps, '-x', color=color)
        xlims = [x * cake.m2d * 1e3 for x in btmax.get_xlim()]
        ylims = (min(amps) * 1e-1, max(amps) * 1e1)
        if new_axis:
            topax.set_xlim(xlims)
            btmax.set_ylim(ylims)
        else:
            tlims = btmax.get_ylim()
            ylims = (min(ylims[0], tlims[0]), max(ylims[1], tlims[1]))
            btmax.set_ylim(ylims)
            topax.set_ylim(ylims)
        btmax.set_title('{0} amplitudes for ({1})'.
                        format(trc_typ.split('_')[0].title(), chCode), y=1.08)
        btmax.annotate(self.getAnnotateString(src_id, sen_id),
                       xy=anno_pt, ha=hor_ali, color=color,
                       fontsize=(self.__notesize), xycoords='figure fraction')
        self.__drawGrid(btmax)

    def __getModelProperty(self, prop):
        mod = self.store.config.earthmodel_1d
        depths = mod.profile('z') / 1e3
        if '/' in prop:
            pros = prop.split('/')
            data = mod.profile(pros[0]) / mod.profile(pros[1])
        else:
            data = mod.profile(prop)

        if prop in ['vp', 'vs', 'rho']:
            data /= 1e3

        return data, depths

    def __createVelocityProfile(self, gft2=None):
        opts = ['vp', 'vs', 'vp/vs', 'rho', 'qp', 'qs', 'qp/qs']
        fig = self.__setupFigure('profile', 4, rows=2)
        axs = fig.axes
        for i, opt in enumerate(opts):
            ax = axs[i]
            if gft2 is None:
                data, depths = self.__getModelProperty(opt)
                ax.plot(data, depths)
            else:
                data, depths = self.__getModelProperty(opt)
                ax.plot(data, depths, linewidth=3,
                        linestyle='--', alpha=0.8, label=self.store_id,
                        color=plot.mpl_color('aluminium4'))
                data, depths = gft2.__getModelProperty(opt)
                ax.plot(data, depths, linewidth=1,
                        color=plot.mpl_color('scarletred2'),
                        label=gft2.store_id)

            if opt in ['vp', 'vs']:
                ex = ' [km/s]'
            elif opt == 'rho':
                ex = ' [kg/m^3]'
            else:
                ex = ''
            ax.set_xlabel(opt + ex)
            ax.xaxis.set_label_coords(0.5, -0.13)
            ax.invert_yaxis()
            pos = ax.get_position()
            if i == 0 or i == 4:
                ax.set_ylabel('Depth [km]')
            if i > 1:
                for j in ax.xaxis.get_ticklabels()[1::2]:
                    j.set_visible(False)
            if (i // 4) == 0:
                y = pos.y0 * 1.025
                ax.set_position([pos.x0, y, pos.width,
                                 pos.height - (y - pos.y0)])
            else:
                y = pos.height * 0.975
                ax.set_position([pos.x0, pos.y0, pos.width, y])

        if gft2 is None:
            ttl = 'Earth model plots: {0}'.format(self.store_id)
        else:
            ttl = 'Earth model plots: {0} & {1}'.format(self.store_id,
                                                        gft2.store_id)
            ax.legend(bbox_to_anchor=(1.25, 1.), loc=2)

        ax.annotate(ttl, xy=(0.5, 0.95), fontsize=(self.__notesize * 2),
                    xycoords='figure fraction', ha='center', va='top')
        return self.__saveTempFigure(fig)

    @staticmethod
    def __setupFigure(fig_type, cols, rows=1):
        fontsize = 10.
        figsize = plot.mpl_papersize('a4', 'landscape')
        plot.mpl_init(fontsize=fontsize)

        fig = plt.figure(figsize=figsize)
        labelpos = plot.mpl_margins(fig, w=7., h=6., units=fontsize,
                                    wspace=7., nw=cols, nh=rows)
        for cnt in range(1, (cols * rows) + 1):
            if fig_type == 'profile' and cnt >= 8:
                continue
            ax = fig.add_subplot(rows, cols, cnt)
            labelpos(ax, 2., 2.25)
            if fig_type.startswith('maxamp'):
                if fig_type.split('_')[1] == 'displacement':
                    ax.set_ylabel('Max. Amplitude [m]')
                else:
                    ax.set_ylabel('Max. Amplitude [m/s]')
                ax.set_yscale('log')
                ax2 = ax.twiny()
                ax.set_xlabel('Distance [km]')
                ax2.set_xlabel('Distance [deg]')
            elif '_traces' in fig_type:
                ax.set_xlabel('Time [s]')
                ax2 = ax.twinx()
                ax2.set_ylabel('Distance [km]')
                ax.set_ylabel('Distance [deg]')
        return fig

    @staticmethod
    def __drawGrid(ax, major=True, minor=True):
        if major:
            ax.grid(
                visible=True,
                which='major',
                c='grey',
                linestyle='-',
                alpha=.45)

        if minor:
            ax.minorticks_on()
            ax.grid(
                visible=True,
                which='minor',
                c='grey',
                linestyle=':',
                alpha=.8)

    @staticmethod
    def __getFigureTitle(fig):
        for ax in fig.axes:
            ttl = ax.get_title()
            if ttl == '':
                continue
            return ttl
        return ''

    def __drawBeachball(self, trc_id, ax):
        src_id, sen_id = trc_id.split('|')
        src = self.sources[src_id]
        mt = src.pyrocko_moment_tensor()

        sz = 20.
        szpt = sz / 72.
        bbx = ax.get_xlim()[0]
        bby = ax.get_ylim()[1]
        move_trans = transforms.ScaledTranslation(
            szpt, -szpt, ax.figure.dpi_scale_trans)
        inv_trans = ax.transData.inverted()
        bbx, bby = inv_trans.transform(move_trans.transform(
            ax.transData.transform((bbx, bby))))
        beachball.plot_beachball_mpl(mt, ax, beachball_type='full',
                                     size=sz, position=(bbx, bby))

    def __saveTempFigure(self, fig):
        if self.output_format == 'pdf':
            fname = NamedTemporaryFile(
                prefix='gft_', suffix='.pdf', dir=self.temp_dir, delete=False)
            fname.close()
            fig.savefig(fname.name, transparent=True)
            out = fname.name
        elif self.output_format == 'html':
            imgdata = io.BytesIO()
            fig.savefig(imgdata, format='png')
            imgdata.seek(0)
            out = base64.b64encode(imgdata.read())
        plt.close(fig)
        return out

    def __tracesMinMaxs(self, trcs, src_id, sen_id):
        # return the min/max amplitudes before and after the arrival of the
        # self.phase_ratio_string phase found in the traces,
        # plus the maximum ratio between the max absolute value
        before = []
        after = []

        tbrk = None
        src = self.sources[src_id]
        sensors = self.sensors[sen_id].sensors
        for i, trc in enumerate(trcs):
            if self.phases == '':
                tbrk = None
            else:
                tbrk = self.store.t(self.phase_ratio_string, src, sensors[i])
            times = trc.get_xdata()
            data = trc.get_ydata()
            tmina = None
            tmaxa = None
            tminb = None
            tmaxb = None
            for idx, time in enumerate(times):
                val = data[idx]
                if time < tbrk or tbrk is None:
                    if tminb is None or tminb > val:
                        tminb = val
                    if tmaxb is None or tmaxb < val:
                        tmaxb = val
                else:
                    if tmina is None or tmina > val:
                        tmina = val
                    if tmaxa is None or tmaxa < val:
                        tmaxa = val
            if tminb is None:
                tminb = 0.
            if tmaxb is None:
                tmaxb = 0.
            before.append((tminb, tmaxb))
            if tbrk is None:
                after.append((tminb, tmaxb))
            else:
                after.append((tmina, tmaxa))

        return before, after

    def __tracesMinMax(self, trcs, src_id, sen_id):
        # return the min/max amplitudes before and after the arrival of the
        # self.phase_ratio_string phase found in the traces,
        # plus the maximum ratio between the max absolute value
        before, after = self.__tracesMinMaxs(trcs, src_id, sen_id)
        mina = min([x[0] for x in after])
        maxa = max([x[1] for x in after])
        minb = min([x[0] for x in before])
        maxb = max([x[1] for x in before])
        ratios = list(map(lambda b, a: 0. if a[0] == a[1] == 0. else
                          max(abs(b[0]), abs(b[1]))/max(abs(a[0]), abs(a[1])),
                          before, after))

        return mina, maxa, minb, maxb, max(ratios)

    def __getSensorDistances(self, src_id, sen_id):
        src = self.sources[src_id]
        return [src.distance_to(sen) for sen in self.sensors[sen_id].sensors]

    @classmethod
    def __createStandardSetups(cls, store_dir, **kwargs):
        tdict = cls.__get_valid_arguments(kwargs)

        gft = cls(store_dir, **tdict)

        if 'source_depth' in kwargs:
            depth = kwargs['source_depth']
        else:
            depth = None
        src1 = gft.createSource('DC', depth, 0., 90., 0., **kwargs)
        # src2 = gft.createSource('DC', depth, -90., 90., -90., **kwargs)
        src3 = gft.createSource('DC', depth, 45., 90., 180., **kwargs)
        # src4 = gft.createSource('Explosion', depth, 0., 90., **kwargs)

        SensorArray.validate_args(kwargs)
        sen1 = gft.createSensors(strike=0., codes=('', 'STA', '', 'R'),
                                 azimuth=0., dip=0., **kwargs)
        sen2 = gft.createSensors(strike=0., codes=('', 'STA', '', 'T'),
                                 azimuth=90., dip=0., **kwargs)
        sen3 = gft.createSensors(strike=0., codes=('', 'STA', '', 'Z'),
                                 azimuth=0., dip=-90., **kwargs)

        gft.trace_configs = [[src3, sen1], [src1, sen2], [src3, sen3]]
        # gft.trace_configs = [[src3, sen1]]
        # gft.trace_configs = [[src3, sen1], [src1, sen2], [src3, sen3],
        #                      [src4, 'all']]
        # gft.trace_configs = [[src4, 'all']]
        # gft.trace_configs = [['all', 'all']]
        gft.__applyTypicalProcedures()
        return gft

    def __applyTypicalProcedures(self):
        if self.trace_configs is None:
            self.createDisplacementTraces()
        else:
            for src, sen in self.trace_configs:
                self.createDisplacementTraces(src, sen)
        if self.plot_velocity:
            if self.trace_configs is None:
                self.createVelocityTraces()
            else:
                for src, sen in self.trace_configs:
                    self.createVelocityTraces('{0}|{1}'.format(src, sen))
        self.applyFrequencyFilters()
        self.getPhaseArrivals()

    def __update(self, **kwargs):
        for k in kwargs:
            temp = kwargs[k]
            if temp is not None:
                setattr(self, k, temp)

    @classmethod
    def runStandardCheck(
            cls, store_dir, source_depth=None,
            lowpass_frequency=None, highpass_frequency=None,
            rel_lowpass_frequency=(1. / 110), rel_highpass_frequency=(1. / 16),
            distance_min=None, distance_max=None, sensor_count=50,
            filter_order=4, pdf_dir=None, plot_velocity=False,
            plot_everything=True, output_format='pdf'):

        args = locals()
        del args['cls']
        gft = cls.__createStandardSetups(**args)
        gft.createOutputDoc()
        gft.__printMessage()
        return gft

    @classmethod
    def runComparissonStandardCheck(
            cls, store_dir1, store_dir2, distance_min, distance_max,
            together=True, source_depth=None, output_format='pdf',
            lowpass_frequency=None, highpass_frequency=None,
            rel_lowpass_frequency=(1. / 110), rel_highpass_frequency=(1. / 16),
            filter_order=4, pdf_dir=None, plot_velocity=False,
            plot_everything=True, sensor_count=50):

        args = locals()
        del args['cls']
        args['change_dists'] = False
        gft1 = cls.__createStandardSetups(store_dir1, **args)
        if 'source_depth' not in args or args['source_depth'] is None:
            args['source_depth'] = gft1.sources[list(gft1.sources.keys())[0]]\
                .depth

        gft2 = cls.__createStandardSetups(store_dir2, **args)

        cls.createComparisonOutputDoc(gft1, gft2, together=together)
        gft1.__printMessage()
        gft2.__printMessage()
        return gft1, gft2

    @staticmethod
    def createDocumentFromFile(filename, allowed=1, **kwargs):
        with open(filename, 'r') as f:
            fstr = f.read()
        gfts = []
        st = None
        end = -1
        tstr = '--- !{0}.GreensFunctionTest'.format(guts_prefix)
        try:
            while True:
                st = fstr.index(tstr, end + 1)
                if st is None:
                    break
                if st > 0 and fstr[st - 1:st] != '\n':
                    end = st
                    st = None
                    continue
                end = fstr.index('\n{0}'.format(tstr), st)
                gfts.append(fstr[st:end])
        except ValueError:
            if st is not None:
                gfts.append(fstr[st:])

        cnt = len(gfts)
        if cnt == 0:
            raise TypeError('error: GreensFunctionTest: non-valid config'
                            ' file passed: {0}'.format(filename))
        elif cnt > allowed:
            raise TypeError('error: GreensFunctionTest: to many configs in'
                            ' file passed: {0}'.format(filename))
        elif cnt < allowed:
            raise TypeError('error: GreensFunctionTest: not enough configs in'
                            ' file passed: {0}'.format(filename))

        if cnt == allowed == 1:
            gft = load(stream=gfts[0])
            gft.__update(**kwargs)
            gft.__applyTypicalProcedures()
            gft.createOutputDoc()
            gft.__printMessage()
            return gft

        if cnt == allowed == 2:
            gft = load(stream=gfts[0])
            gft.__update(**kwargs)
            gft.__applyTypicalProcedures()

            gft2 = load(stream=gfts[1])
            gft2.__update(**kwargs)
            gft2.__applyTypicalProcedures()

            gft.createComparisonOutputDoc(gft, gft2, **kwargs)
            gft.__printMessage()
            gft2.__printMessage()
            return gft, gft2
