
import time
import numpy as num
import pasimple
import logging

from pyrocko import trace, util


logger = logging.getLogger('pyrocko.streaming.pulse')


class PulseInput(object):

    def __init__(self):

        # Audio attributes for the recording
        self.format = pasimple.PA_SAMPLE_S32LE
        self.sample_size = pasimple.format2width(self.format)
        self.nchannels = 1
        self.rate = 41000
        self.deltat = 1.0 / num.float128(self.rate)
        self.time_zero = None
        self.nsamples_read = 0
        self.nsamples_block = trace.nextpow2(41000//20)
        self.pa = None
        self.previous_samples = None
        self.tdone = None

    def acquisition_start(self):
        self.pa = pasimple.PaSimple(
            pasimple.PA_STREAM_RECORD, self.format, self.nchannels, self.rate)

    def acquisition_stop(self):
        self.pa.close()
        self.pa = None

    def process(self):

        if self.pa is None:
            return

        audio_data = self.pa.read(
            self.nchannels * self.sample_size * self.nsamples_block)

        if self.time_zero is None:
            self.time_zero = (
                time.time()
                - self.nsamples_block * self.deltat * self.nchannels)

            self.time_zero = num.round(
                self.time_zero / self.deltat) * self.deltat

        samples = num.frombuffer(audio_data, dtype='<i4')
        samples = samples.reshape(
            (samples.size // self.nchannels, self.nchannels))

        tmin = self.time_zero + self.nsamples_read * self.deltat
        # time_deviation = tmin - (
        #     time.time() - self.nsamples_block * self.deltat * self.nchannels)
        #
        # print(time_deviation)

        if self.previous_samples is not None:
            samples_combi = num.vstack([self.previous_samples, samples])
            toffset = - self.previous_samples.shape[0] * self.deltat
        else:
            samples_combi = samples.copy()
            toffset = 0

        for ichannel in range(self.nchannels):
            tr = trace.Trace(
                '', 'A', '%02i' % ichannel, 'ACU',
                tmin=tmin + toffset,
                deltat=self.deltat,
                ydata=samples_combi[:, ichannel])

            tr_down = tr.copy()

            tr_down.downsample_to(tr.deltat*10, snap=True)

            tr_down = tr_down.chop(
                self.tdone + tr_down.deltat
                if self.tdone is not None
                else tr_down.tmin,
                tr_down.tmax + tr_down.deltat,
                inplace=False)

            if tr_down.data_len() > 0:
                self.got_trace(tr_down)

        self.previous_samples = samples
        self.tdone = tr_down.tmax

        self.nsamples_read += samples.shape[0]

    def got_trace(self, tr):
        logger.info(
            'Got trace from PulseAudio: %s, mean: %g, std: %g' % (
                tr.summary, num.mean(tr.ydata), num.std(tr.ydata)))

        print(tr.tmin / self.deltat, tr.tmax / tr.deltat)


def main():
    import sys
    util.setup_logging('main', 'info')

    if len(sys.argv) != 1:
        sys.exit('usage: python -m pyrocko.streaming.pulse')

    cs = PulseInput()
    try:
        cs.acquisition_start()
    except Exception as e:
        sys.exit(str(e))

    try:
        while True:
            cs.process()

    except KeyboardInterrupt:
        pass

    finally:
        cs.acquisition_stop()


if __name__ == '__main__':
    main()
