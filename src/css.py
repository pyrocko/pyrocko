from struct import unpack
import os
import numpy as num
import logging

from pyrocko import trace, util


logging.basicConfig(level='INFO')
logger = logging.getLogger('css-convert')

use_template = True

storage_types = {
        's4': ('>%ii'),
        'i4': ('<%ii'),
}


template = [
    ('sta', str, (0, 6), 'station code'),
    ('chan', str, (7, 15), 'channel code'),
    ('time', float, (16, 33), 'epoch time of first sample in file'),
    ('wfid', int, (34, 43), 'waveform identifier'),
    ('chanid', int, (44, 52), 'channel identifier'),
    ('jdate', int, (53, 61), 'julian date'),
    ('endtime', float, (62, 79),  'time +(nsamp -1 )/samles'),
    ('nsamp', int, (80, 88), 'number of samples'),
    ('samprate', float, (89, 100), 'sampling rate in samples/sec'),
    ('calib', float, (101, 117), 'nominal calibration'),
    ('calper', float, (118, 134), 'nominal calibration period'),
    ('instype', str, (135, 141), 'instrument code'),
    ('segtype', str, (142, 143), 'indexing method'),
    ('datatype', str, (144, 146), 'numeric storage'),
    ('clip', str, (147, 148), 'clipped flag'),
    ('dir', str, (149, 213), 'directory'),
    ('dfile', str, (214, 246), 'data file'),
    ('foff', int, (247, 257), 'byte offset of data seg ment within file'),
    ('commid', int, (258, 267), 'comment identifier'),
    ('Iddate', util.stt, (268, 287), 'load date')
]


class CSSWfError(Exception):
    def __init__(self, **kwargs):
        f2str = {
            str: 'string',
            int: 'integer',
            float: 'float',
            util.stt: 'time'
        }
        kwargs['convert'] = f2str[kwargs['convert']]

        error_str = 'Error while parsing "{data}" to {convert} (line {iline}, \
columns {istart} - {istop}, description="{desc}")'.format(**kwargs)
        Exception.__init__(self, error_str)
        self.error_arguments = kwargs


class Wfdisc():
    ''' Wfdisc header file class

    :param fn: filename of wfdisc header file'''
    def __init__(self, fn):

        self.fn = fn
        self.data = []
        self.read()

    def read_wf_file(self, fn, nbytes, dtype):
        ''' Read binary waveform file
        :param fn: filename
        :param nbytes: number of bytes to be read
        :param dtype: datatype string
        '''
        with open(fn, 'rb') as f:
            fmt = dtype % nbytes
            try:
                data = num.array(unpack(fmt, f.read(nbytes * 4 + 1)),
                                 dtype=num.int32)
            except:
                logger.exception('Error while unpacking %s' % fn)
                return
        return data

    def read(self):
        ''' read header file'''
        with open(self.fn, 'r') as f:
            lines = f.readlines()
            for iline, line in enumerate(lines):
                if use_template:
                    d = {}
                    for (ident, convert, (istart, istop), desc) in template:
                        try:
                            d[ident] = convert(line[istart: istop].strip())
                        except:
                            raise CSSWfError(
                                iline=iline+1, data=line[istart:istop],
                                ident=ident, convert=convert,
                                istart=istart+1, istop=istop+1, desc=desc,
                            )
                else:
                    d = {}
                    split = line.split()
                    d['sta'] = template[0][1](split[0])
                    d['chan'] = template[1][1](split[1])
                    d['time'] = template[2][1](split[2])
                    d['nsamp'] = template[7][1](split[7])
                    d['samprate'] = template[8][1](split[8])
                    d['datatype'] = template[-8][1](split[-8])
                    d['dir'] = template[-6][1](split[-6])
                    d['dfile'] = template[-5][1](split[-5])
                self.data.append(d)

    def iter_pyrocko_traces(self, load_data=True):
        for idata, d in enumerate(self.data):
            fn = os.path.join(d['dir'], d['dfile'])
            logger.debug('converting %s', d['dfile'])
            try:
                if load_data:
                    ydata = self.read_wf_file(
                            fn, d['nsamp'],
                            storage_types[d['datatype']])
                else:
                    ydata = None

            except IOError as e:
                if e.errno == 2:
                    logger.debug(e)
                    continue
                else:
                    raise e
            dt = 1./d['samprate']
            yield trace.Trace(station=d['sta'],
                              channel=d['chan'],
                              deltat=dt,
                              tmin=d['time'],
                              tmax=d['time'] + dt*d['nsamp'],
                              ydata=ydata)


def iload(file_name, load_data, **kwargs):
    '''
    :param load_data: wfdisc filename
    '''
    wfdisc = Wfdisc(file_name)
    for pyrocko_trace in wfdisc.iter_pyrocko_traces(load_data=load_data):
        yield pyrocko_trace
