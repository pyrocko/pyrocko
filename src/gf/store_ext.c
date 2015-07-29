#define NPY_NO_DEPRECATED_API 7

#define GF_STORE_HEADER_SIZE (8+4)

/* security limit for length of traces, shifts and offsets (samples) */
#define SLIMIT 1000000

#define inlimits(i) (-SLIMIT <= (i) && (i) <= SLIMIT)
#define inposlimits(i) (0 <= (i) && (i) <= SLIMIT)

#include "Python.h"
#include "numpy/arrayobject.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__linux__)
  #include <endian.h>
#elif defined (__APPLE__)
  #include <libkern/OSByteOrder.h>
  #define be32toh(x) OSSwapBigToHostInt32(x)
  #define le32toh(x) OSSwapLittleToHostInt32(x)
  #define be64toh(x) OSSwapBigToHostInt64(x)
  #define le64toh(x) OSSwapLittleToHostInt64(x)
#endif 

typedef npy_float32 gf_dtype;
typedef npy_float32 float32_t;

#if (PY_VERSION_HEX >= 0x02070000)
  #define HAVE_CAPSULE
#endif

#define GF_STORE_IS_LITTLE_ENDIAN

#ifdef GF_STORE_IS_LITTLE_ENDIAN
  #define xe64toh le64toh
  #define xe32toh le32toh
#endif

#ifdef GF_STORE_IS_BIG_ENDIAN
  #define xe64toh be64toh
  #define xe32toh be32toh
#endif

#define fe32toh(x) \
   ({ int32_t _i; \
      _i = xe32toh(*((int32_t*)&(x))); \
      *((gf_dtype*)&_i); })

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

static PyObject *StoreExtError;

typedef struct {
    uint64_t data_offset;
    int32_t itmin;
    int32_t nsamples;
    gf_dtype begin_value;
    gf_dtype end_value;
} record_t;

typedef struct {
    int f_index;
    int f_data;
    uint64_t nrecords;
    uint64_t data_size;
    float32_t deltat;
    record_t *records;
    gf_dtype *data;
    gf_dtype **memdata;
} store_t;

typedef struct {
    int is_zero;
    int32_t itmin;
    int32_t nsamples;
    gf_dtype begin_value;
    gf_dtype end_value;
    gf_dtype *data;
} trace_t;

typedef enum {
    SUCCESS = 0,
    INVALID_RECORD,
    EMPTY_RECORD,
    BAD_RECORD,
    ALLOC_FAILED,
    BAD_REQUEST,
    BAD_DATA_OFFSET,
    READ_DATA_FAILED,
    SEEK_INDEX_FAILED,
    READ_INDEX_FAILED,
    FSTAT_TRACES_FAILED,
    BAD_STORE,
    MMAP_INDEX_FAILED,
    MMAP_TRACES_FAILED,
} store_error_t;

const char* store_error_names[] = {
    "SUCCESS",
    "INVALID_RECORD",
    "EMPTY_RECORD",
    "BAD_RECORD",
    "ALLOC_FAILED",
    "BAD_REQUEST",
    "BAD_DATA_OFFSET",
    "READ_DATA_FAILED",
    "SEEK_INDEX_FAILED",
    "READ_INDEX_FAILED",
    "FSTAT_TRACES_FAILED",
    "BAD_STORE",
    "MMAP_INDEX_FAILED",
    "MMAP_TRACES_FAILED",
};

#define REC_EMPTY 0
#define REC_ZERO 1
#define REC_SHORT 2

static trace_t ZERO_TRACE = { 1, 0, 0, 0.0, 0.0, NULL };
static store_t ZERO_STORE = { 0, 0, 0, 0, 0.0, NULL, NULL, NULL };

static store_error_t store_get_span(const store_t *store, uint64_t irecord,
                             int32_t *itmin, int32_t *nsamples, int *is_zero) {
    record_t *record;

    if (irecord >= store->nrecords) {
        return INVALID_RECORD;
    }

    record = &store->records[irecord];
    *itmin = xe32toh(record->itmin);
    *nsamples = xe32toh(record->nsamples);
    *is_zero = REC_ZERO == xe64toh(record->data_offset);

    if (!inlimits(*itmin) || !inposlimits(*nsamples)) {
        return BAD_RECORD;
    }

    return SUCCESS;
}

static store_error_t store_read(
        const store_t *store,
        uint64_t data_offset,
        size_t nbytes,
        void *data) {

    size_t nhave;
    ssize_t nread;

    nhave = 0;
    while (nhave < nbytes) {
        nread = pread(store->f_data, data, nbytes-nhave, data_offset+nhave);
        if (-1 == nread) {
            return READ_DATA_FAILED;
        }
        nhave += nread;
    }

    return SUCCESS;
}

static store_error_t store_get(
        const store_t *store,
        uint64_t irecord,
        trace_t *trace) {

    record_t *record;
    uint64_t data_offset;
    store_error_t err;
    size_t nbytes;

    if (irecord >= store->nrecords) {
        *trace = ZERO_TRACE;
        return INVALID_RECORD;
    }

    record = &store->records[irecord];
    data_offset = xe64toh(record->data_offset);
    trace->itmin = xe32toh(record->itmin);
    trace->nsamples = xe32toh(record->nsamples);
    trace->begin_value = fe32toh(record->begin_value);
    trace->end_value = fe32toh(record->end_value);

    if (!inlimits(trace->itmin) || !inposlimits(trace->nsamples) ||
            data_offset >= UINT64_MAX - SLIMIT * sizeof(gf_dtype)) {
        return BAD_RECORD;
    }

    if (REC_EMPTY == data_offset) {
        *trace = ZERO_TRACE;
        return EMPTY_RECORD;
    }

    if (REC_ZERO == data_offset) {
        *trace = ZERO_TRACE;
        return SUCCESS;
    }

    trace->is_zero = 0;

    if (data_offset + trace->nsamples*sizeof(gf_dtype) > store->data_size) {
        *trace = ZERO_TRACE;
        return BAD_DATA_OFFSET;
    }

    if (REC_SHORT == data_offset) {
        trace->data = &record->begin_value;
    } else {
        if (NULL != store->data) {
            trace->data = &store->data[data_offset/sizeof(gf_dtype)];
        } else {
            if (NULL == store->memdata[irecord]) {
                nbytes = trace->nsamples * sizeof(gf_dtype);
                store->memdata[irecord] = (gf_dtype*)malloc(nbytes);
                if (NULL == store->memdata[irecord]) {
                    *trace = ZERO_TRACE;
                    return ALLOC_FAILED;
                }
                err = store_read(store, data_offset, nbytes, store->memdata[irecord]);
                if (SUCCESS != err) {
                    free(store->memdata[irecord]);
                    store->memdata[irecord] = NULL;
                    *trace = ZERO_TRACE;
                    return err;
                }
            }
            trace->data = store->memdata[irecord];
        }
    }

    return SUCCESS;
}

static int clipint32(int32_t lo, int32_t hi, int32_t n) {
    return n <= lo ? lo : n >= hi ? hi : n;
}

static void trace_trim(trace_t *trace, int32_t itmin, int32_t nsamples) {
    int32_t ilo, ihi;
    ilo = max(itmin, trace->itmin);
    ihi = min(itmin + nsamples, trace->itmin + trace->nsamples);
    trace->data = (trace->nsamples == 0) ? NULL : (trace->data + (ilo - trace->itmin));
    trace->itmin = ilo;
    trace->nsamples = max(0, ihi - ilo);
}

static void trace_trim_sticky(trace_t *trace, int32_t itmin, int32_t nsamples) {
    int32_t ilo, ihi;
    ilo = clipint32(0, trace->nsamples-1, itmin-trace->itmin);
    ihi = clipint32(1, trace->nsamples, itmin+nsamples-trace->itmin);
    trace->itmin = trace->itmin + ilo;
    trace->nsamples = max(0, (ihi - ilo));
    trace->data += ilo;
}

static store_error_t store_sum(
        const store_t *store,
        const uint64_t *irecords,
        const float32_t *delays,
        const float32_t *weights,
        int32_t n,
        int32_t itmin,
        int32_t nsamples,
        trace_t *result) {

    int32_t itmax;
    int is_zero;
    float itmin_d, itmax_d;
    float32_t weight, delay;
    trace_t trace;
    float32_t deltat = store->deltat;
    gf_dtype begin_value, end_value;
    gf_dtype *out;
    int ilo;
    /*int32_t ifloor, iceil;*/
    int i, j;
    int idelay_floor, idelay_ceil;
    int ihave;
    float w1, w2;
    store_error_t err;

    *result = ZERO_TRACE;
    if (0 == n) {
        return SUCCESS;
    }

    if (-1 == nsamples) {
        itmin_d = itmax_d = 0.;
        ihave = 0;
        for (j=0; j<n; j++) {
            err = store_get_span(store, irecords[j], &itmin, &nsamples, &is_zero);
            if (SUCCESS != err) {
                return err;
            }

            if (is_zero || 0.0 == weights[j]) {
                continue;
            }

            itmax = itmin + nsamples - 1;

            if (ihave) {
                itmin_d = min(itmin_d, itmin + delays[j]/deltat);
                itmax_d = max(itmax_d, itmax + delays[j]/deltat);
            }
            else {
                itmin_d = itmin + delays[j]/deltat;
                itmax_d = itmax + delays[j]/deltat;
                ihave = 1;
            }
        }

        if (!ihave) {
            return SUCCESS; /* result is zero */
        }

        itmin = floor(itmin_d);
        nsamples = ceil(itmax_d) - itmin + 1;
    }

    if (!inlimits(itmin) || !inposlimits(nsamples)) {
        return BAD_REQUEST;
    }

    out = NULL;
    if (0 != nsamples) {
        out = (gf_dtype*)calloc(nsamples, sizeof(gf_dtype));
        if (out == NULL) {
            return ALLOC_FAILED;
        }
    }

    begin_value = 0.0;
    end_value = 0.0;

    for (j=0; j<n; j++) {

        delay = delays[j];
        weight = weights[j];
        idelay_floor = (int)floor(delay/deltat);
        idelay_ceil = (int)ceil(delay/deltat);
        if (!inlimits(idelay_floor) || !inlimits(idelay_ceil)) {
            free(out);
            return BAD_REQUEST;
        }

        if (0.0 == weight) {
            continue;
        }

        err = store_get(store, irecords[j], &trace);
        if (SUCCESS != err) {
            free(out);
            return err;
        }
        if (trace.is_zero) {
            continue;
        }

        trace_trim_sticky(&trace, itmin - idelay_ceil, nsamples + idelay_ceil - idelay_floor);

        if (idelay_floor == idelay_ceil) {
            ilo = itmin - idelay_floor - trace.itmin;
            /*for (i=0; i<nsamples; i++) {
                ifloor = i + ilo;
                ifloor = max(0, min(ifloor, trace.nsamples-1));
                out[i] += fe32toh(trace.data[ifloor]) * weight;
            } // version below is a bit faster */
            for (i=0; i<min(-ilo,nsamples); i++) {
                out[i] += fe32toh(trace.data[0]) * weight;
            }
            for (i=max(0,-ilo); i<min(nsamples, trace.nsamples-ilo); i++) {
                out[i] += fe32toh(trace.data[i+ilo]) * weight;
            }
            for (i=max(0,trace.nsamples-ilo); i<nsamples; i++) {
                out[i] += fe32toh(trace.data[trace.nsamples-1]) * weight;
            }
        } else {
            ilo = itmin - idelay_floor - trace.itmin;
            /*ihi = ilo - 1;*/
            w1 = (idelay_ceil - delay/deltat) * weight;
            w2 = (delay/deltat - idelay_floor) * weight;
            /*for (i=0; i<nsamples; i++) {
                ifloor = i + ilo;
                iceil = i + ihi;
                ifloor = max(0, min(ifloor, trace.nsamples-1));
                iceil = max(0, min(iceil, trace.nsamples-1));
                out[i] += fe32toh(trace.data[ifloor]) * w1;
                out[i] += fe32toh(trace.data[iceil]) * w2;
            } // version below is a bit faster */
            for (i=0; i<min(-ilo,nsamples); i++) {
                out[i] += fe32toh(trace.data[0]) * weight;
            }
            for (i=max(0,-ilo); i<min(nsamples, -ilo+1); i++) {
                out[i] += fe32toh(trace.data[i+ilo])*w1
                          + fe32toh(trace.data[0])*w2;
            }
            for (i=max(0,-ilo+1); i<min(nsamples, trace.nsamples-ilo); i++) {
                out[i] += fe32toh(trace.data[i+ilo])*w1
                          + fe32toh(trace.data[i+ilo-1])*w2;
            }
            for (i=max(0,trace.nsamples-ilo);
                 i<min(nsamples, trace.nsamples-ilo+1); i++) {
                out[i] += fe32toh(trace.data[trace.nsamples-1]) * w1
                          + fe32toh(trace.data[i+ilo-1])*w2;
            }
            for (i=max(0,trace.nsamples-ilo+1); i<nsamples; i++) {
                out[i] += fe32toh(trace.data[trace.nsamples-1]) * weight;
            }
        }

        begin_value += trace.begin_value * weight;
        end_value += trace.end_value * weight;
    }

    result->is_zero = 0;
    result->itmin = itmin;
    result->nsamples = nsamples;
    result->begin_value = begin_value;
    result->end_value = end_value;
    result->data = out;

    return SUCCESS;
}

static store_error_t store_init(int f_index, int f_data, store_t *store) {
    void *p;
    struct stat st;
    size_t mmap_index_size;
    int use_mmap;

    use_mmap = 0;

    *store = ZERO_STORE;

    store->f_index = f_index;
    store->f_data = f_data;
    if (8 != pread(store->f_index, &store->nrecords, 8, 0)) {
        return READ_INDEX_FAILED;
    }
    if (4 != pread(store->f_index, &store->deltat, 4, 8)) {
        return READ_INDEX_FAILED;
    }

    store->nrecords = xe64toh(store->nrecords);
    store->deltat = fe32toh(store->deltat);

    if (-1 == fstat(store->f_data, &st)) {
        return FSTAT_TRACES_FAILED;
    }

    if (st.st_size < 0) {
        return FSTAT_TRACES_FAILED;
    }

    store->data_size = (uint64_t)st.st_size;
    if (store->nrecords >= (UINT64_MAX - GF_STORE_HEADER_SIZE) / sizeof(record_t)) {
        return BAD_STORE;
    }

    mmap_index_size = sizeof(record_t) * store->nrecords + GF_STORE_HEADER_SIZE;
    if (mmap_index_size >= SIZE_MAX) {
        return MMAP_INDEX_FAILED;
    }

    p = mmap(NULL, mmap_index_size, PROT_READ, MAP_SHARED, store->f_index, 0);
    if (MAP_FAILED == p) {
        return MMAP_INDEX_FAILED;
    }

    store->records = (record_t*)((char*)p+GF_STORE_HEADER_SIZE);

    /* on 32-bit systems, use mmap only if traces file is considerably smaller
     * than address space */

    use_mmap = store->data_size < SIZE_MAX / 8; 

    if (use_mmap) {
        if (store->data_size >= SIZE_MAX) {
            return MMAP_TRACES_FAILED;
        }
        p = mmap(NULL, store->data_size, PROT_READ, MAP_SHARED, store->f_data, 0);
        if (MAP_FAILED == p) {
            return MMAP_TRACES_FAILED;
        }

        store->data = (gf_dtype*)p;
    } else {
        if (store->nrecords > SIZE_MAX) {
            return ALLOC_FAILED;
        }
        store->memdata = (gf_dtype**)calloc(store->nrecords, sizeof(gf_dtype*));
        if (NULL == store->memdata) {
            return ALLOC_FAILED;
        }
    }
    return SUCCESS;
}

void store_deinit(store_t *store) {
    size_t mmap_index_size;
    uint64_t irecord;

    mmap_index_size = sizeof(record_t) * store->nrecords + GF_STORE_HEADER_SIZE;
    if (store->records != NULL) {
        munmap(((char*)store->records)-GF_STORE_HEADER_SIZE, mmap_index_size);
    }

    if (store->data != NULL) {
        munmap(store->data, store->data_size);
    }

    if (store->memdata != NULL) {
        for (irecord=0; irecord<store->nrecords; irecord++) {
            if (NULL != store->memdata[irecord]) {
                free(store->memdata[irecord]);
                store->memdata[irecord] = NULL;
            }
        }
        free(store->memdata);
    }

    *store = ZERO_STORE;
}

#ifdef HAVE_CAPSULE
static void w_store_delete(PyObject *capsule) {
    store_t *store;
    store = (store_t*)PyCapsule_GetPointer(capsule, NULL);
    store_deinit(store);
    free(store);
}
#else
static void w_store_delete(void *store) {
    store_deinit((store_t*)store);
    free(store);
}
#endif

static PyObject* w_store_init(PyObject *dummy, PyObject *args) {
    int f_index, f_data;
    store_t *store;
    store_error_t err;

    (void)dummy; /* silence warning */

    if (!PyArg_ParseTuple(args, "ii", &f_index, &f_data)) {
        PyErr_SetString(StoreExtError, "usage store_init(f_index, f_data)" );
        return NULL;
    }

    store = (store_t*) calloc(1, sizeof(store_t));
    if (store == NULL) {
        PyErr_SetString(StoreExtError, "memory allocation failed.");
        return NULL;
    }

    err = store_init(f_index, f_data, store);
    if (SUCCESS != err) {
        PyErr_SetString(StoreExtError, store_error_names[err]);
        store_deinit(store);
        free(store);
        return NULL;
    }

#ifdef HAVE_CAPSULE
    return Py_BuildValue("N",
        PyCapsule_New((void*)store, NULL, w_store_delete));
#else
    return Py_BuildValue("N",
        PyCObject_FromVoidPtr((void*)store, w_store_delete));
#endif
}

static PyObject* w_store_get(PyObject *dummy, PyObject *args) {
    PyObject *capsule;
    uint64_t irecord;
    store_t *store;
    gf_dtype *adata;
    trace_t trace;
    PyArrayObject *array = NULL;
    npy_intp array_dims[1] = {0};
    int32_t itmin;
    int32_t nsamples;
    int i;
    store_error_t err;

    (void)dummy; /* silence warning */

    if (!PyArg_ParseTuple(args, "OKii", &capsule, &irecord, &itmin, &nsamples)) {
        PyErr_SetString(StoreExtError, "usage store_get(cstore, irecord, itmin, nsamples)");
        return NULL;
    }
#ifdef HAVE_CAPSULE
    if (!PyCapsule_IsValid(capsule, NULL)) {
#else
    if (!PyCObject_Check(capsule)) {
#endif
        PyErr_SetString(StoreExtError, "invalid cstore argument");
        return NULL;
    }
    if (!inlimits(itmin)) {
        PyErr_SetString(StoreExtError, "invalid itmin argument");
        return NULL;
    }
    if (!(inposlimits(nsamples) || -1 == nsamples)) {
        PyErr_SetString(StoreExtError, "invalid nsamples argument");
        return NULL;
    }

#ifdef HAVE_CAPSULE
    store = (store_t*)PyCapsule_GetPointer(capsule, NULL);
#else
    store = (store_t*)PyCObject_AsVoidPtr(capsule);
#endif

    err = store_get(store, irecord, &trace);
    if (SUCCESS != err) {
        PyErr_SetString(StoreExtError, store_error_names[err]);
        return NULL;
    }

    if (-1 != nsamples) {
        trace_trim(&trace, itmin, nsamples);
    }

    array_dims[0] = trace.nsamples;
    array = (PyArrayObject*)PyArray_EMPTY(1, array_dims, NPY_FLOAT32, 0);
    adata = (gf_dtype*)PyArray_DATA(array);
    for (i=0; i<trace.nsamples; i++) {
        adata[i] = fe32toh(trace.data[i]);
    }

    return Py_BuildValue("Nififf", array, trace.itmin, store->deltat,
                         trace.is_zero, trace.begin_value, trace.end_value);
}

static PyObject* w_store_sum(PyObject *dummy, PyObject *args) {
    PyObject *capsule, *irecords_arr, *delays_arr, *weights_arr;
    store_t *store;
    gf_dtype *adata;
    trace_t result;
    PyArrayObject *array = NULL;
    npy_intp array_dims[1] = {0};
    PyArrayObject *c_irecords_arr, *c_delays_arr, *c_weights_arr;
    uint64_t *irecords;
    float32_t *delays, *weights;
    npy_intp n, n1, n2;
    int32_t itmin;
    int32_t nsamples;
    store_error_t err;

    (void)dummy; /* silence warning */

    if (!PyArg_ParseTuple(args, "OOOOii", &capsule, &irecords_arr, &delays_arr,
                                     &weights_arr, &itmin, &nsamples)) {
        PyErr_SetString(StoreExtError,
            "usage: store_sum(cstore, irecords, delays, weights, itmin, nsamples)");

        return NULL;
    }

#ifdef HAVE_CAPSULE
    if (!PyCapsule_IsValid(capsule, NULL)) {
#else
    if (!PyCObject_Check(capsule)) {
#endif
        PyErr_SetString(StoreExtError, "invalid cstore argument");
        return NULL;
    }
    if (!PyArray_Check(irecords_arr) ||
            NPY_UINT64 != PyArray_TYPE((PyArrayObject*)irecords_arr)) {
        PyErr_SetString(StoreExtError,
            "store_sum: 'irecords' must be a NumPy array of type uint64");
        return NULL;
    }
    if (!PyArray_Check(delays_arr) ||
            NPY_FLOAT32 != PyArray_TYPE((PyArrayObject*)delays_arr)) {
        PyErr_SetString(StoreExtError,
            "store_sum: 'delays' must be a NumPy array of type float32");
        return NULL;
    }
    if (!PyArray_Check(weights_arr) ||
            NPY_FLOAT32 != PyArray_TYPE((PyArrayObject*)weights_arr)) {
        PyErr_SetString(StoreExtError,
            "store_sum: 'weights' must be a NumPy array of type float32");
        return NULL;
    }
    if (!inlimits(itmin)) {
        PyErr_SetString(StoreExtError, "invalid itmin argument");
        return NULL;
    }
    if (!(inposlimits(nsamples) || -1 == nsamples)) {
        PyErr_SetString(StoreExtError, "invalid nsamples argument");
        return NULL;
    }
#ifdef HAVE_CAPSULE
    store = (store_t*)PyCapsule_GetPointer(capsule, NULL);
#else
    store = (store_t*)PyCObject_AsVoidPtr(capsule);
#endif

    c_irecords_arr = PyArray_GETCONTIGUOUS((PyArrayObject*)irecords_arr);
    c_delays_arr = PyArray_GETCONTIGUOUS((PyArrayObject*)delays_arr);
    c_weights_arr = PyArray_GETCONTIGUOUS((PyArrayObject*)weights_arr);

    n = PyArray_SIZE(c_irecords_arr);
    n1 = PyArray_SIZE(c_delays_arr);
    n2 = PyArray_SIZE(c_weights_arr);

    if (n != n1 || n != n2) {
        PyErr_SetString(StoreExtError,
            "store_sum: 'irecords', 'delays', and 'weights' must have same length");
        return NULL;
    }

    irecords = PyArray_DATA(c_irecords_arr);
    delays = PyArray_DATA(c_delays_arr);
    weights = PyArray_DATA(c_weights_arr);

    err = store_sum(store, irecords, delays, weights, n, itmin, nsamples, &result);
    if (SUCCESS != err) {
        PyErr_SetString(StoreExtError, store_error_names[err]);
        return NULL;
    }

    Py_DECREF(c_irecords_arr);
    Py_DECREF(c_delays_arr);
    Py_DECREF(c_weights_arr);

    array_dims[0] = result.nsamples;
    array = (PyArrayObject*)PyArray_EMPTY(1, array_dims, NPY_FLOAT32, 0);
    adata = (gf_dtype*)PyArray_DATA(array);
    memcpy(adata, result.data, result.nsamples*sizeof(gf_dtype));
    free(result.data);

    return Py_BuildValue("Nififf", array, result.itmin, store->deltat,
                         result.is_zero, result.begin_value, result.end_value);
}


static PyMethodDef StoreExtMethods[] = {
    {"store_init",  w_store_init, METH_VARARGS,
        "Initialize store struct." },

    {"store_get", w_store_get, METH_VARARGS,
        "Get a GF trace." },

    {"store_sum", w_store_sum, METH_VARARGS,
        "Get weight-and-delay-sum of GF traces." },

    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initstore_ext(void)
{
    PyObject *m;

    m = Py_InitModule("store_ext", StoreExtMethods);
    if (m == NULL) return;
    import_array();

    StoreExtError = PyErr_NewException("store_ext.error", NULL, NULL);
    Py_INCREF(StoreExtError);  /* required, because other code could remove `error`
                               from the module, what would create a dangling
                               pointer. */
    PyModule_AddObject(m, "StoreExtError", StoreExtError);
}

