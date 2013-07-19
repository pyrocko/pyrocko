import os, struct, math, shutil, fcntl, copy, logging, re
from collections import Counter

import numpy as num
from scipy import signal

from pyrocko import util
from pyrocko.gf import meta as meta_module

logger = logging.getLogger('pyrocko.gf.store')

gf_dtype = num.dtype(num.float32)
gf_dtype_nbytes_per_sample = 4

gf_store_header_dtype = [
        ('nrecords', '<u8'),
        ('deltat', '<f4'),
    ]

gf_store_header_fmt = '<Qf'
gf_store_header_fmt_size = struct.calcsize(gf_store_header_fmt)

gf_record_dtype = num.dtype([
        ('data_offset', '<u8'),
        ('itmin', '<i4'),
        ('nsamples', '<u4'),
        ('begin_value', '<f4'),
        ('end_value', '<f4'),
    ])

def check_string_id(s):
    if not re.match(meta_module.StringID.pattern, s):
        raise ValueError('invalid name %s' % s)

# - data_offset
#
# Data file offset of first sample in bytes (the seek value).
# Special values:
# 
#  0x00 - missing trace
#  0x01 - zero trace (GF trace is physically zero)
#  0x02 - short trace of 1 or 2 samples (no need for data allocation)
#
# - itmin
#
# Time of first sample of the trace as a multiple of the sampling interval. All
# GF samples must be evaluated at multiples of the sampling interval with
# respect to a global reference time zero.
#
# - nsamples
# 
# Number of samples in the GF trace. 
# 
# - begin_value, end_value
#
# Values of first and last sample. These values are included in data[] 
# redunantly.

class GFTrace:
    def __init__(self, data=None, itmin=0, deltat=1.0, 
            is_zero=False, begin_value=None, end_value=None):

        if data is not None:
            data = num.asarray(data, dtype=gf_dtype)
        else:
            data = num.array([], dtype=gf_dtype)

        if data is None or data.size == 0 or num.all(data) == 0.0:
            self.is_zero = True

        self.data = data
        self.itmin = itmin
        self.deltat = deltat
        self.is_zero = is_zero

        if data is not None and data.size > 0:
            if begin_value is None:
                begin_value = data[0]
            if end_value is None:
                end_value = data[-1]

        self.begin_value = begin_value
        self.end_value = end_value

    @property
    def t(self):
        return num.linspace(self.itmin*self.deltat, 
                (self.itmin+self.data.size-1)*self.deltat, self.data.size)

    def __str__(self, itmin=0):
        
        if self.is_zero:
            return 'ZERO'

        s = []
        for i in range(itmin, self.itmin + self.data.size):
            if i >= self.itmin and i < self.itmin + self.data.size:
                s.append( '%7.4g' % self.data[i-self.itmin] )
            else:
                s.append( ' '*7 )
        
        return '|'.join(s)


Zero = GFTrace(is_zero=True)

class StoreError(Exception):
    pass

class CannotCreate(StoreError):
    pass

class CannotOpen(StoreError):
    pass

class DuplicateInsert(StoreError):
    pass

class NotAllowedToInterpolate(StoreError):
    def __str__(self):
        return 'not allowed to interpolate'

def remove_if_exists(fn, force=False):
    if os.path.exists(fn):
        if force:
            os.remove(fn)
        else:
            raise CannotCreate('file %s already exists' % fn)

class Store_:

    @staticmethod
    def index_fn_(store_dir):
        return os.path.join(store_dir, 'index')

    @staticmethod
    def data_fn_(store_dir):
        return os.path.join(store_dir, 'traces')

    @staticmethod
    def create(store_dir, deltat, nrecords, force=False):


        try:
            util.ensuredir(store_dir)
        except:
            raise CannotCreate('cannot create directory %s' % store_dir)
        
        index_fn = Store_.index_fn_(store_dir)
        data_fn = Store_.data_fn_(store_dir)

        for fn in (index_fn, data_fn):
            remove_if_exists(fn, force)

        with open(index_fn, 'wb') as f:
            f.write(struct.pack(gf_store_header_fmt, nrecords, deltat))
            records = num.zeros(nrecords, dtype=gf_record_dtype)
            records.tofile(f)

        with open(data_fn, 'wb') as f:
            f.write('\0' * 32)

    def __init__(self, store_dir, mode='r', use_memmap=True):
        self.store_dir = store_dir
        self.mode = mode

        index_fn = self.index_fn()
        data_fn = self.data_fn()

        if mode == 'r':
            fmode = 'rb'
        elif mode == 'w':
            fmode = 'r+b'
        else:
            assert False

        try:
            self._f_index = open(index_fn, fmode)
            self._f_data = open(data_fn, fmode)
        except:
            self.mode = ''
            raise CannotOpen('cannot open gf store: %s' % self.store_dir)

        dataheader = self._f_index.read(gf_store_header_fmt_size)
        nrecords, deltat = struct.unpack(gf_store_header_fmt, dataheader)
        self.nrecords = nrecords
        self.deltat = deltat
        self._use_memmap = use_memmap

        self._load_index()
        

    def __del__(self):
        if self.mode != '':
            self.close()

    def lock(self):
        fcntl.lockf(self._f_index, fcntl.LOCK_EX)

    def unlock(self):
        self._f_data.flush()
        fcntl.lockf(self._f_index, fcntl.LOCK_UN)

    def put(self, irecord, trace):
        self._put(irecord, trace)

    def get_record(self, irecord):
        return self._get_record(irecord)

    def get_span(self, irecord, decimate=1):
        return self._get_span(irecord, decimate=decimate)

    def get(self, irecord, itmin=None, nsamples=None, decimate=1):
        return self._get(irecord, itmin=itmin, nsamples=nsamples, decimate=decimate)

    def sum(self, irecords, delays, weights, itmin=None, nsamples=None, decimate=1):
        return self._sum(irecords, delays, weights, itmin, nsamples, 
                    decimate)
    
    def sum_reference(self, irecords, delays, weights, itmin=None, nsamples=None, decimate=1):
        return self._sum_reference(irecords, delays, weights, itmin, nsamples, 
                    decimate)

    def close(self):
        if self.mode == 'w':
            self._save_index()

        self._f_data.close()
        self._f_index.close()
        self.mode = ''

    def _get_record(self, irecord):
        return self._records[irecord]

    def _get(self, irecord, itmin=None, nsamples=None, decimate=1):
        '''Retrieve complete GF trace from storage.'''

        assert self.mode == 'r'
        assert 0 <= irecord < self.nrecords, 'irecord = %i, nrecords = %i' % (irecord, self.nrecords)

        (ipos, itmin_data, nsamples_data, begin_value, end_value) = self._records[irecord]

        if None in (itmin, nsamples): 
            itmin = itmin_data
            itmax = itmin_data + nsamples_data - 1
            nsamples = nsamples_data
        else:
            itmin = itmin * decimate
            nsamples = nsamples * decimate
            itmax = itmin + nsamples - decimate

        if ipos == 0:
            return None

        elif ipos == 1:
            return Zero

        if decimate == 1:
            ilo = max(itmin, itmin_data) - itmin_data
            ihi = min(itmin+nsamples, itmin_data+nsamples_data) - itmin_data
            data = self._get_data(ipos, begin_value, end_value, ilo, ihi)

            return GFTrace(data, itmin_data+ilo, self.deltat,
                begin_value=begin_value, end_value=end_value)

        else:
            itmax_data = itmin_data + nsamples_data - 1

            # put begin and end to multiples of new sampling rate
            itmin_ext = (max(itmin,itmin_data)/decimate) * decimate
            itmax_ext = -((-min(itmax,itmax_data))/decimate) * decimate
            nsamples_ext = itmax_ext - itmin_ext + 1

            # add some padding for the aa filter
            order = 30
            itmin_ext_pad = itmin_ext - order/2
            itmax_ext_pad = itmax_ext + order/2
            nsamples_ext_pad = itmax_ext_pad - itmin_ext_pad + 1
            
            itmin_overlap = max(itmin_data, itmin_ext_pad)
            itmax_overlap = min(itmax_data, itmax_ext_pad)

            ilo = itmin_overlap - itmin_ext_pad
            ihi = max(ilo, itmax_overlap - itmin_ext_pad + 1)
            ilo_data = itmin_overlap - itmin_data
            ihi_data = max( ilo_data, itmax_overlap - itmin_data + 1)

            data_ext_pad = num.empty(nsamples_ext_pad, dtype=gf_dtype)
            data_ext_pad[ilo:ihi] = self._get_data(ipos, begin_value, end_value, 
                    ilo_data, ihi_data)
            data_ext_pad[:ilo] = begin_value
            data_ext_pad[ihi:] = end_value

            b = signal.firwin(order + 1, 1. / decimate, window='hamming')
            a = 1.
            data_filt_pad = signal.lfilter(b,a, data_ext_pad)
            data_deci = data_filt_pad[order:order+nsamples_ext:decimate]
            if data_deci.size >= 1:
                if itmin_ext <= itmin_data:
                    data_deci[0] = begin_value

                if itmax_ext >= itmax_data:
                    data_deci[-1] = end_value

            return GFTrace(data_deci, itmin_ext/decimate, self.deltat*decimate,
                begin_value=begin_value, end_value=end_value)

    def _get_span(self, irecord, decimate=1):
        '''Get temporal extent of GF trace at given index.'''
        assert 0 <= irecord < self.nrecords, 'irecord = %i, nrecords = %i' % (irecord, self.nrecords)
        
        (_, itmin, nsamples, _, _) = self._records[irecord]

        itmax = itmin + nsamples - 1

        if decimate == 1:
            return itmin, itmax
        else:
            return itmin/decimate, -((-itmax)/decimate)

    def _put(self, irecord, trace):
        '''Save GF trace to storage.'''

        assert self.mode == 'w'
        assert abs(trace.deltat - self.deltat) < 1e-7 * self.deltat
        assert 0 <= irecord < self.nrecords, 'irecord = %i, nrecords = %i' % (irecord, self.nrecords)

        if self._records[irecord][0] != 0:
            raise DuplicateInsert('record %i already in store' % irecord)

        if trace.is_zero or num.all(trace.data == 0.0):
            self._records[irecord] = (1,0,0,0.,0.)
            return
        
        ndata = trace.data.size

        if ndata > 2:
            self._f_data.seek(0, 2)
            ipos = self._f_data.tell()
            trace.data.tofile( self._f_data )
        else:
            ipos = 2

        self._records[irecord] = (ipos, trace.itmin, ndata, trace.data[0], trace.data[-1])

        
    def _sum(self, irecords, delays, weights, itmin=None, nsamples=None, decimate=1):
        '''Sum delayed and weighted GF traces.'''

        assert self.mode == 'r'
    
        deltat = self.deltat * decimate

        if len(irecords) == 0:
            return Zero

        assert len(irecords) == len(delays)
        assert len(irecords) == len(weights)

        if None in (itmin, nsamples):
            itmin_delayed, itmax_delayed = [], []
            for irecord, delay in zip(irecords, delays):
                itmin, itmax = self._get_span(irecord, decimate=decimate)
                itmin_delayed.append(itmin + delay/deltat)
                itmax_delayed.append(itmax + delay/deltat)
            
            itmin = int(math.floor(min(itmin_delayed)))
            nsamples = int(math.ceil(max(itmax_delayed))) - itmin + 1


        out = num.zeros(nsamples, dtype=gf_dtype)
        if nsamples == 0:
            return GFTrace(out, itmin, deltat)
        
        for irecord, delay, weight in zip(irecords, delays, weights):

            if weight == 0.0:
                continue

            idelay_floor = int(math.floor(delay/deltat))
            idelay_ceil = int(math.ceil(delay/deltat))

            gf_trace = self._get(irecord, 
                    itmin - idelay_ceil, 
                    nsamples + idelay_ceil - idelay_floor,
                    decimate=decimate)

            assert gf_trace.itmin >= itmin - idelay_ceil
            assert gf_trace.data.size <= nsamples + idelay_ceil - idelay_floor

            if gf_trace.is_zero:
                continue

            ilo = gf_trace.itmin - itmin + idelay_floor
            ihi = ilo + gf_trace.data.size + (idelay_ceil-idelay_floor)

            data = gf_trace.data

            if idelay_floor == idelay_ceil:
                out[ilo:ihi] += data * weight
            else:
                if data.size:
                    k = 1 
                    if ihi <= nsamples:
                        out[ihi-1] += gf_trace.end_value * ((idelay_ceil-delay/deltat) * weight)
                        k = 0

                    out[ilo+1:ihi-k] += data[:data.size-k] * ((delay/deltat-idelay_floor) * weight)
                    k = 1
                    if ilo >= 0:
                        out[ilo] += gf_trace.begin_value * ((delay/deltat-idelay_floor) * weight)
                        k = 0

                    out[ilo+k:ihi-1] += data[k:] * ((idelay_ceil-delay/deltat) * weight)


            if ilo > 0 and gf_trace.begin_value != 0.0:
                out[:ilo] += gf_trace.begin_value * weight

            if ihi < nsamples and gf_trace.end_value != 0.0:
                out[ihi:] += gf_trace.end_value * weight

        
        return GFTrace(out, itmin, deltat)

    def _sum_reference(self, irecords, delays, weights, itmin=None, nsamples=None, decimate=1):


        deltat = self.deltat * decimate
        
        datas = []
        itmins = []
        for i, delay, weight in zip(irecords, delays, weights):
            tr = self._get(i, decimate=decimate)
            if tr.is_zero:
                continue

            idelay_floor = int(math.floor(delay/deltat))
            idelay_ceil = int(math.ceil(delay/deltat))

            if idelay_floor == idelay_ceil:
                itmins.append(tr.itmin + idelay_floor)
                datas.append(tr.data.copy()*weight)
            else:
                itmins.append(tr.itmin + idelay_floor)
                datas.append(tr.data.copy()*weight*(idelay_ceil-delay/deltat))
                itmins.append(tr.itmin + idelay_ceil)
                datas.append(tr.data.copy()*weight*(delay/deltat-idelay_floor))

        if not itmins:
            return Zero
            
        itmin_all = min(itmins)

        itmax_all = max( itmin_ + data.size for (itmin_, data) in zip(itmins, datas) )
        if itmin is not None:
            itmin_all = min(itmin_all, itmin)
        if nsamples is not None:
            itmax_all = max(itmax_all, itmin+nsamples)

        nsamples_all = itmax_all - itmin_all 

        arr = num.zeros((len(datas), nsamples_all), dtype=gf_dtype)
        for i, itmin_, data in zip(num.arange(len(datas)), itmins, datas):
            ilo = itmin_-itmin_all
            ihi = ilo + data.size
            arr[i, :ilo] = data[0]
            arr[i, ilo:ihi] = data
            arr[i, ihi:] = data[-1]

        sum_arr = arr.sum(axis=0)

        if itmin is not None and nsamples is not None:
            ilo = itmin-itmin_all
            ihi = ilo + nsamples
            sum_arr = sum_arr[ilo:ihi]

        else:
            itmin = itmin_all

        return GFTrace(sum_arr, itmin, deltat)

    def _load_index(self):
        if self._use_memmap:
            records = num.memmap(self._f_index, dtype=gf_record_dtype, 
                    offset=gf_store_header_fmt_size,
                    mode= ('r','r+')[self.mode == 'w'])

        else:
            self._f_index.seek(gf_store_header_fmt_size)
            records = num.fromfile(self._f_index, dtype=gf_record_dtype)

        assert len(records) == self.nrecords

        self._records = records

    def _save_index(self):
        self._f_index.seek(0)
        self._f_index.write(struct.pack(gf_store_header_fmt, self.nrecords, 
            self.deltat))

        if self._use_memmap:
            del self._records
        else:
            self._f_index.seek(gf_store_header_fmt_size)
            self._records.tofile(self._f_index)
            self._f_index.flush()

    def _get_data(self, ipos, begin_value, end_value, ilo, ihi): 
        if ihi - ilo > 0:
            if ipos == 2:
                data_orig = num.empty(2, dtype=gf_dtype)
                data_orig[0] = begin_value
                data_orig[1] = end_value
                return data_orig[ilo:ihi]
            else:
                self._f_data.seek(ipos + ilo*gf_dtype_nbytes_per_sample)
                return num.fromfile(self._f_data, gf_dtype, ihi-ilo)
        else:
            x  = num.empty((0,), dtype=gf_dtype)
            return num.empty((0,), dtype=gf_dtype)

    def index_fn(self):
        return Store_.index_fn_(self.store_dir)
    
    def data_fn(self):
        return Store_.data_fn_(self.store_dir)

    def count_special_records(self):
        return num.histogram( self._records['data_offset'], bins=[0,1,2,3, num.uint64(-1) ] )[0]

class Store(Store_):

    '''
    Green's function disk storage and summation machine.
    
    The `Store` can be used to efficiently store, retrieve, and sum Green's
    function traces. A `Store` contains many 1D time traces sampled at even
    multiples of a global sampling rate, where each time trace has an
    individual start and end time.  The traces are treated as having repeating
    end points, so the functions they represent can be non-constant only
    between begin and end time. It provides capabilities to retrieve decimated
    traces and to extract parts of the traces. The main purpose of this class
    is to provide a fast, easy to use, and flexible machanism to compute
    weighted delay-and-sum stacks with many Green's function traces involved.
    
    Indiviual Green's functions are accessed through a single integer index at
    low level.  At higher level, various indexing schemes can be implemented by
    providing a mapping from physical coordinates to the low level index. E.g.
    for a problem with cylindrical symmetry, one might define a mapping from
    (z1, z2, r) -> i. Index translation is done in the
    :py:class:`pyrocko.gf.meta.GFSet` subclass object associated with the Store.
    '''

    @staticmethod
    def create(store_dir, meta, force=False, extra=None):
        '''Create new GF store.
        
        Creates a new GF store at path `store_dir`. The layout of the GF is
        defined with the parameters given in `meta`, which should be an
        object of a subclass of :py:class:`pyrocko.gf.meta.GFSet`. This function will
        refuse to overwrite an existing GF store, unless `force` is set  to
        ``True``. If more information, e.g. parameters used for the modelling
        code, earth models or other, should be saved along with the GF store,
        these may be provided though a dict given to `extra`. The keys of 
        this dict must be names and the values must be *guts* type objects.
        '''

        store = Store_.create(store_dir, meta.deltat, meta.nrecords, 
                force=force)

        meta_fn = os.path.join(store_dir, 'meta')
        remove_if_exists(meta_fn, force)

        meta_module.dump(meta, filename=meta_fn)

        for sub_dir in 'decimated', 'extra':
            dpath = os.path.join(store_dir, sub_dir)
            if os.path.exists(dpath):
                if force:
                    shutil.rmtree(dpath)
                else:
                    raise CannotCreate('directory %s already exists' % dpath)

            os.mkdir(dpath)

        if extra:
            for k,v in extra.iteritems():
                check_string_id(k)
                fn = os.path.join(store_dir, 'extra', k)
                remove_if_exists(fn, force)
                meta_module.dump(v, filename=fn)

    def __init__(self, store_dir, mode='r'):
        Store_.__init__(self, store_dir, mode=mode)
        meta_fn = os.path.join(store_dir, 'meta')
        self.meta = meta_module.load(filename=meta_fn)
        self._decimated = {}
        self._extra = {}
        for decimate in range(2,9):
            if os.path.isdir(self._decimated_store_dir(decimate)):
                self._decimated[decimate] = None

    def get_extra(self, key):
        '''Get extra information stored under given key.'''

        check_string_id(key)
        x = self._extra
        if key not in x:
            fn = os.path.join(self.store_dir, 'extra', key)
            if not os.path.exists(fn):
                raise KeyError(k)

            x[key] = meta_module.load(filename=fn)

        return x[key]

    def put(self, args, trace):
        '''Insert trace into GF store.
        
        Store a single GF trace at (high-level) index `args`.'''

        irecord = self.meta.irecord(*args)
        self._put(irecord, trace)

    def get_record(self, args):
        irecord = self.meta.irecord(*args)
        return self._get_record(irecord)

    def get(self, args, itmin=None, nsamples=None, decimate=1, interpolate='nearest_neighbor'):
        '''Retrieve GF trace from store.

        Retrieve a single GF trace from the store at (high-level) index `args`.
        By default, the full trace is retrieved. Given `itmin` and `nsamples`,
        only the selected portion of the trace is extracted. If `decimate` is
        an integer in the range [2,8], the trace is decimated on the fly or, if
        available, the trace is read from a decimated version of the GF store.
        '''

        store, decimate = self._decimated_store(decimate)
        if interpolate == 'nearest_neighbor':
            irecord = store.meta.irecord(*args)
            return store._get(irecord, itmin=itmin, nsamples=nsamples, decimate=decimate)

        else:
            irecords, weights = zip(*store.meta.vicinity(*args))
            if interpolate == 'off' and len(irecords) != 1:
                raise NotAllowedToInterpolate()

            return store._sum(irecords, num.zeros(len(irecords)), weights, itmin, nsamples, decimate)

    def sum(self, args, delays, weights, itmin=None, nsamples=None, decimate=1):
        '''Sum delayed and weighted GF traces.

        Calculate sum of delayed and weighted GF traces. `args` is a tuple of
        arrays forming the (high-level) indices of the GF traces to be
        selected.  Delays and weights for the summation are given in the arrays
        `delays` and `weights`. If `itmin` and `nsamples` are given,
        computation is restricted to the output time range *(decimated) sampling
        interval x [ itmin, (itmin + nsamples - 1) ]*.  If `decimate` is an
        integer in the range [2,8], decimated traces are used in the summation.
        '''

        store, decimate = self._decimated_store(decimate)
        irecords = store.meta.irecords(*args)
        return store._sum(irecords, delays, weights, itmin, nsamples, decimate)
    
    def sum_reference(self, args, delays, weights, itmin=None, nsamples=None, decimate=1):
        '''Alternative version of :py:meth:`sum`.'''

        store, decimate = self._decimated_store(decimate)
        irecords = store.meta.irecords(*args)
        return store._sum_reference(irecords, delays, weights, itmin, nsamples, decimate)

    def make_decimated(self, decimate, meta=None, force=False):
        '''Create decimated version of GF store.

        Create a downsampled version of the GF store. Downsampling is done for
        the integer factor `decimate` which should be in the range [2,8].  If
        `meta` is ``None``, all traces of the GF store are decimated and held
        available (i.e. the index mapping of the original store is used),
        otherwise, a different spacial stepping can be specified by giving a
        modified GF store configuration in `meta` (see :py:meth:`create`).
        Decimated GF sub-stores are created under the `decimated` subdirectory
        within the GF store directory. Holding available decimated versions of
        the GF store can save computation time, IO bandwidth, or decrease
        memory footprint at the cost of increased disk space usage, when
        computation are done for lower frequency signals.
        '''

        if not (2 <= decimate <= 8):
            raise StoreError('decimate argument must be in the range [2,8]')

        assert self.mode == 'r'

        if meta is None:
            meta = self.meta

        meta = copy.deepcopy(meta)
        meta.sample_rate = self.meta.sample_rate / decimate

        if decimate in self._decimated:
            del self._decimated[decimate]

        store_dir = self._decimated_store_dir(decimate)
        if os.path.exists(store_dir):
            if force:
                shutil.rmtree(store_dir)
            else:
                raise CannotCreate('store already exists at %s' % store_dir)


        store_dir_incomplete = store_dir + '-incomplete'
        Store.create(store_dir_incomplete, meta, force=force)

        decimated = Store(store_dir_incomplete, 'w')
        for args in decimated.meta.iter_nodes():
            tr = self.get(args, decimate=decimate)
            decimated.put(args, tr)

        decimated.close()

        shutil.move(store_dir_incomplete, store_dir)

        self._decimated[decimate] = None

    def stats(self):
        counter = self.count_special_records()

        sdata = os.stat(self.data_fn()).st_size
        sindex = os.stat(self.index_fn()).st_size

        stats = dict(
                total = self.nrecords,
                inserted = (counter[1] + counter[2] + counter[3]), 
                empty = counter[0],
                short = counter[2],
                zero = counter[1],
                size_data = sdata,
                size_index = sindex,
                decimated = sorted(self._decimated.keys())
            )

        return stats

    stats_keys = 'total inserted empty short zero size_data size_index decimated'.split()

    def check(self):
        problems = 0
        i =0
        for args in self.meta.iter_nodes():
            tr = self.get(args)
            if tr and not tr.is_zero:
                if not tr.begin_value == tr.data[0]:
                    logger.warn('wrong begin value for trace at %s (data corruption?)' % str(args))
                    problems += 1
                if not tr.end_value == tr.data[-1]:
                    logger.warn('wrong end value for trace at %s (data corruption?)' % str(args))
                    problems += 1
                if not num.all(num.isfinite(tr.data)):
                    logger.warn('nans or infs in trace at %s' % str(args))
                    problems += 1

        return problems

    def _decimated_store_dir(self, decimate):
        return os.path.join(self.store_dir, 'decimated', str(decimate))

    def _decimated_store(self, decimate):
        if decimate == 1 or decimate not in self._decimated:
            return self, decimate
        else:
            store = self._decimated[decimate]
            if store is None:
                store = Store(self._decimated_store_dir(decimate), 'r')
                self._decimated[decimate] = store

            return store, 1

__all__ = 'Store GFTrace Zero StoreError CannotCreate CannotOpen'.split()

