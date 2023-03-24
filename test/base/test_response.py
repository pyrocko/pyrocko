
import os
import unittest
import numpy as num
from pyrocko import util, evalresp, pz, response, guts

from .. import common


show_plot = int(os.environ.get('MPL_SHOW', 0))


def plot_tfs(freqs, tfs):
    colors = ['black', 'red', 'blue']
    import pylab as lab
    lab.gcf().add_subplot(2, 1, 1)
    for itf, tf in enumerate(tfs):
        lab.plot(freqs, num.abs(tf), color=colors[itf])

    lab.xscale('log')
    lab.yscale('log')

    lab.gcf().add_subplot(2, 1, 2)
    for itf, tf in enumerate(tfs):
        lab.plot(freqs, num.angle(tf), color=colors[itf])

    lab.xscale('log')
    lab.show()


class ResponseTestCase(unittest.TestCase):

    @unittest.skipUnless(
        evalresp.have_evalresp(), 'evalresp not supported on this platform')
    def test_evalresp(self, plot=show_plot):

        resp_fpath = common.test_data_file('test2.resp')

        freqs = num.logspace(num.log10(0.001), num.log10(10.), num=1000)

        transfer = evalresp.evalresp(
            sta_list='BSEG',
            cha_list='BHZ',
            net_code='GR',
            locid='',
            instant=util.str_to_time('2012-01-01 00:00:00'),
            freqs=freqs,
            units='DIS',
            file=resp_fpath,
            rtype='CS')[0][4]

        pz_fpath = common.test_data_file('test2.sacpz')

        zeros, poles, constant = pz.read_sac_zpk(pz_fpath)

        resp = response.PoleZeroResponse(zeros, poles, constant)

        transfer2 = resp.evaluate(freqs)
        from scipy import signal
        if hasattr(signal, 'freqs_zpk'):
            freqs_zpk = signal.freqs_zpk
            del signal.freqs_zpk
        else:
            freqs_zpk = None

        transfer3 = resp.evaluate(freqs)

        if freqs_zpk is not None:
            signal.freqs_zpk = freqs_zpk

        assert cnumeq(transfer3, transfer2, 1e-6)

        if plot:
            plot_tfs(freqs, [transfer, transfer2, transfer3])

        assert numeq(transfer, transfer2, 1e-4)

    @unittest.skipUnless(
        evalresp.have_evalresp(), 'evalresp not supported on this platform')
    def test_conversions(self):

        from pyrocko import model
        from pyrocko.io import resp, enhanced_sacpz
        from pyrocko.io import stationxml

        t = util.str_to_time('2014-01-01 00:00:00')
        codes = 'GE', 'EIL', '', 'BHZ'

        resp_fpath = common.test_data_file('test1.resp')
        stations = [model.Station(
            *codes[:3],
            lat=29.669901,
            lon=34.951199,
            elevation=210.0,
            depth=0.0)]

        sx_resp = resp.make_stationxml(
            stations, resp.iload_filename(resp_fpath))

        sx_resp.validate()

        assert sx_resp.network_list[0].station_list[0].channel_list[0] \
            .dip is None

        stations[0].set_channels_by_name('BHE', 'BHN', 'BHZ')

        sx_resp2 = resp.make_stationxml(
            stations, resp.iload_filename(resp_fpath))

        sx_resp2.validate()

        assert sx_resp2.network_list[0].station_list[0].channel_list[0] \
            .dip.value == -90.0

        pr_sx_resp = sx_resp.get_pyrocko_response(
            codes, time=t, fake_input_units='M/S')
        pr_evresp = response.Evalresp(
            resp_fpath, nslc_id=codes, target='vel', time=t)

        sacpz_fpath = common.test_data_file('test1.sacpz')
        sx_sacpz = enhanced_sacpz.make_stationxml(
            enhanced_sacpz.iload_filename(sacpz_fpath))
        pr_sx_sacpz = sx_sacpz.get_pyrocko_response(
            codes, time=t, fake_input_units='M/S')
        pr_sacpz = response.PoleZeroResponse(*pz.read_sac_zpk(sacpz_fpath))

        try:
            pr_sacpz.zeros.remove(0.0j)
        except ValueError:
            pr_sacpz.poles.append(0.0j)

        sx_sacpz_resp = \
            sx_sacpz.network_list[0].station_list[0].channel_list[0].response
        sx_sacpz_resp2 = pz.read_to_stationxml_response(
            input_unit=sx_sacpz_resp.instrument_sensitivity.input_units.name,
            output_unit=sx_sacpz_resp.instrument_sensitivity.output_units.name,
            normalization_frequency=10.,
            filename=sacpz_fpath)

        pr_sx_sacpz2 = sx_sacpz_resp2.get_pyrocko_response(
            '.'.join(codes)).expect_one()
        try:
            pr_sx_sacpz2.responses[0].zeros.remove(0.0j)
        except ValueError:
            pr_sx_sacpz2.responses[0].poles.append(0.0j)

        sxml_geofon_fpath = common.test_data_file('test1.stationxml')
        sx_geofon = stationxml.load_xml(filename=sxml_geofon_fpath)
        pr_sx_geofon = sx_geofon.get_pyrocko_response(
            codes, time=t, fake_input_units='M/S')

        sxml_iris_fpath = common.test_data_file('test2.stationxml')
        sx_iris = stationxml.load_xml(filename=sxml_iris_fpath)
        pr_sx_iris = sx_iris.get_pyrocko_response(
            codes, time=t, fake_input_units='M/S')

        freqs = num.logspace(num.log10(0.001), num.log10(1.0), num=1000)
        tf_ref = pr_evresp.evaluate(freqs)
        for pr in [pr_sx_resp, pr_sx_sacpz, pr_sacpz, pr_sx_geofon,
                   pr_sx_iris, pr_sx_sacpz2]:
            tf = pr.evaluate(freqs)
            # plot_tfs(freqs, [tf_ref, tf])
            assert cnumeqrel(tf_ref, tf, 0.01)

    def test_converters_butter(self):
        deltat = 0.01
        butter = response.ButterworthResponse(
            type='high', corner=0.1, order=4.0)
        butter_pz = butter.to_polezero()
        butter_analog = butter.to_analog()
        butter_digital = butter.to_digital(deltat)
        butter_digital_polezero = butter.to_digital_polezero(deltat)

        if show_plot:
            from pyrocko.plot.response import plot
            plot(
                [butter, butter_pz, butter_analog, butter_digital,
                 butter_digital_polezero],
                labels=[
                    'butter', 'butter_pz', 'butter_analog', 'butter_digital',
                    'butter_digital_polezero'],
                fmin=0.001, fmax=0.5/deltat, nf=200)

    def test_converters_pz(self):
        deltat = 0.01
        butter = response.ButterworthResponse(
            type='high', corner=0.1, order=4.0)

        butter_pz = butter.to_polezero()
        butter_pz_analog = butter_pz.to_analog()
        butter_pz_digital_pz = butter_pz.to_digital_polezero(deltat)
        resps = [butter, butter_pz, butter_pz_analog, butter_pz_digital_pz]
        labels = [
            'butter', 'butter_pz', 'butter_pz_analog', 'butter_pz_digital_pz']

        for method in ['zoh', 'foh', 'euler', 'bilinear', 'backward_diff']:
            try:
                resps.append(butter_pz.to_digital(deltat, method=method))
                labels.append('butter_pz_digital_' + method)
            except ValueError:
                pass

        if show_plot:
            from pyrocko.plot.response import plot
            plot(resps, labels=labels, fmin=0.001, fmax=0.5/deltat, nf=200)

    def test_converters_analog(self):
        deltat = 0.01
        butter = response.ButterworthResponse(
            type='high', corner=0.1, order=4.0)
        butter_analog = butter.to_analog()
        resps = [butter, butter_analog]
        labels = ['butter', 'butter_analog']

        for method in ['zoh', 'foh', 'euler', 'bilinear', 'backward_diff']:
            try:
                resps.append(butter_analog.to_digital(deltat, method=method))
                labels.append('butter_analog_digital_' + method)
            except ValueError:
                pass

        if show_plot:
            from pyrocko.plot.response import plot
            plot(resps, labels=labels, fmin=0.001, fmax=0.5/deltat, nf=200)

    def test_dump_load(self):

        r = response.FrequencyResponse()

        r = response.PoleZeroResponse([0j, 0j], [1j, 2j, 1+3j, 1-3j], 1.0)
        r.regularize()
        r2 = guts.load_string(r.dump())
        assert cnumeq(r.poles, r2.poles, 1e-6)
        assert cnumeq(r.zeros, r2.zeros, 1e-6)
        assert numeq(r.constant, r2.constant)

        r = response.SampledResponse(
            [0., 1., 5., 10.],
            [0., 1., 1., 0.])

        r.regularize()
        r2 = guts.load_string(r.dump())
        assert numeq(r.frequencies, r2.frequencies, 1e-6)
        assert cnumeq(r.values, r2.values, 1e-6)

        r = response.IntegrationResponse(2, 5.0)
        r2 = guts.load_string(r.dump())
        assert numeq(r.n, r2.n)
        assert numeq(r.gain, r2.gain, 1e-6)

        r = response.DifferentiationResponse(2, 5.0)
        r2 = guts.load_string(r.dump())
        assert numeq(r.n, r2.n)
        assert numeq(r.gain, r2.gain, 1e-6)

        r = response.AnalogFilterResponse(
            a=[1.0, 2.0, 3.0],
            b=[2.0, 3.0])
        r2 = guts.load_string(r.dump())
        assert numeq(r.a, r2.a, 1e-6)
        assert numeq(r.b, r2.b, 1e-6)


def numeq(a, b, eps=0.0):
    a = num.asarray(a)
    b = num.asarray(b)
    return num.all(num.abs(a - b) <= eps)


def cnumeq(a, b, eps=0.0):
    a = num.asarray(a)
    b = num.asarray(b)
    return num.all(num.abs(a.real - b.real) <= eps) and \
        num.max(num.abs(a.imag - b.imag) <= eps)


def cnumeqrel(a, b, eps=0.0):
    a = num.asarray(a)
    b = num.asarray(b)
    return num.all(num.abs(a - b)/(num.abs(a) + num.abs(b)) <= eps)


if __name__ == '__main__':
    util.setup_logging('test_response', 'warning')
    unittest.main()
