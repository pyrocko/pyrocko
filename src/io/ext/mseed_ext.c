
/* Copyright (c) 2009, Sebastian Heimann <sebastian.heimann@zmaw.de>

  This file is part of pyrocko. For licensing information please see the file
  COPYING which is included with pyrocko. */

#define NPY_NO_DEPRECATED_API 7

#include "Python.h"
#include "numpy/arrayobject.h"

#include <libmseed.h>
#include <assert.h>

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state); (void) m;
static struct module_state _state;
#endif


/*********************************************************************
 * ms_readtraces:
 *
 * This is a simple wrapper for ms_readtraces_selection() that uses no
 * selections.
 *
 * See the comments with ms_readtraces_selection() for return values
 * and further description of arguments.
 *********************************************************************/
static int
pyrocko_ms_readtraces (MSTraceGroup **ppmstg, const char *msfile, flag dataflag, int segment, int segment_nrecords)
{
  MSRecord *msr     = NULL;
  MSFileParam *msfp = NULL;
  int retcode;
  int reclen_detected = 0;
  off_t fpos = 0;

  if (!ppmstg)
    return MS_GENERROR;

  /* Initialize MSTraceGroup if needed */
  if (!*ppmstg)
  {
    *ppmstg = mst_initgroup (*ppmstg);

    if (!*ppmstg) {
      return MS_GENERROR;
    }
  }

  /* Loop over the input file */
  while ((retcode = ms_readmsr_main (&msfp, &msr, msfile, 0, fpos ? &fpos : NULL, NULL,
                                     1, dataflag, NULL, 0)) == MS_NOERROR)
  {
    /* Add to trace group */

    if ((segment >= 0) && (reclen_detected == 0)) {
        reclen_detected = msr->reclen;
        fpos = -reclen_detected * segment_nrecords * segment;

        if ((-1 * fpos) >= msfp->filesize) {
            retcode = MS_ENDOFFILE;
            break;
        }

        ms_readmsr_main (&msfp, &msr, NULL, 0, NULL, NULL, 0, 0, NULL, 0);
        continue;
    }
    mst_addmsrtogroup (*ppmstg, msr, 0, -1., -1.);

    if ((segment >= 0) && (msfp->recordcount >= segment_nrecords))
        break;

  }

  ms_readmsr_main (&msfp, &msr, NULL, 0, NULL, NULL, 0, 0, NULL, 0);
  return retcode;
}


static PyObject*
mseed_get_traces(PyObject *m, PyObject *args, PyObject *kwds) {
    char          *filename;
    MSTraceGroup  *mstg = NULL;
    MSTrace       *mst = NULL;
    int           retcode;
    npy_intp      array_dims[1] = {0};
    PyObject      *array = NULL;
    PyObject      *out_traces = NULL;
    PyObject      *out_trace = NULL;
    int           numpytype;
    PyObject      *unpackdata = NULL;

    int           segment = -1;
    int           segmented_trs = 0;
    int           segment_nrecords = 512;

    struct module_state *st = GETSTATE(m);
    (void) m;

    static char *kwlist[] = {"filename", "dataflag", "segment", "segment_nrecords", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|Oii", kwlist, &filename, &unpackdata, &segment, &segment_nrecords))
        return NULL;

    if (!PyBool_Check(unpackdata)) {
        PyErr_SetString(st->error, "dataflag argument must be a boolean");
        return NULL;
    }
    if (segment_nrecords <= 0) {
        PyErr_SetString(st->error, "segment_nrecords must be positive");
        return NULL;
    }

    out_traces = Py_BuildValue("[]");
    /* get data from mseed file */
    retcode = pyrocko_ms_readtraces(&mstg, filename, (unpackdata == Py_True), segment, segment_nrecords);
    if (retcode < 0) {
        PyErr_Format(st->error, "Cannot read file '%s': %s", filename, ms_errorstr(retcode));
        Py_XDECREF(out_traces);
        return NULL;
    }

    if (! mstg) {
        PyErr_SetString(st->error, "Error reading file");
        Py_XDECREF(out_traces);
        return NULL;
    }

    /* check that there is data in the traces */
    if (unpackdata == Py_True) {
        mst = mstg->traces;
        while (mst) {
            if (mst->datasamples == NULL) {
                PyErr_SetString(st->error, "Error reading file - datasamples is NULL");
                Py_XDECREF(out_traces);
                return NULL;
            }
            mst = mst->next;
        }
    }

    mst = mstg->traces;
    while (mst) {
        
        if (unpackdata == Py_True) {
            array_dims[0] = mst->numsamples;
            switch (mst->sampletype) {
                case 'i':
                    assert(ms_samplesize('i') == 4);
                    numpytype = NPY_INT32;
                    break;
                case 'a':
                    assert(ms_samplesize('a') == 1);
                    numpytype = NPY_INT8;
                    break;
                case 'f':
                    assert(ms_samplesize('f') == 4);
                    numpytype = NPY_FLOAT32;
                    break;
                case 'd':
                    assert(ms_samplesize('d') == 8);
                    numpytype = NPY_FLOAT64;
                    break;
                default:
                    PyErr_Format(st->error, "Unknown sampletype %c\n", mst->sampletype);
                    Py_XDECREF(out_traces);
                    mst_freegroup (&mstg);
                    return NULL;
            }
            array = PyArray_SimpleNew(1, array_dims, numpytype);
            memcpy(PyArray_DATA((PyArrayObject*)array), mst->datasamples, mst->numsamples*ms_samplesize(mst->sampletype));
        } else {
            Py_INCREF(Py_None);
            array = Py_None;
        }

        /* convert data to python tuple */
        out_trace = Py_BuildValue("(c,s,s,s,s,L,L,d,N,d)",
                                  mst->dataquality, mst->network, mst->station, mst->location, mst->channel,
                                  mst->starttime, mst->endtime, mst->samprate, array, segment);
        
        PyList_Append(out_traces, out_trace);
        Py_DECREF(out_trace);
        mst = mst->next;
    }

    mst_freegroup (&mstg);
    return out_traces;
}

static int tuple2mst(PyObject* in_trace, MSTrace* mst, int* msdetype) {
    int           numpytype;
    int           length;
    char          *network, *station, *location, *channel, *dataquality;
    PyObject      *array = NULL;
    PyArrayObject *contiguous_array = NULL;

    if (!PyTuple_Check(in_trace)) {
        PyErr_SetString(PyExc_ValueError, "Trace record must be a tuple of (network, station, location, channel, starttime, endtime, samprate, dataquality, data).");
        return EXIT_FAILURE;
    }
    
    if (!PyArg_ParseTuple(in_trace, "ssssLLdsO",
                          &network, &station, &location, &channel,
                          &mst->starttime, &mst->endtime, &mst->samprate, &dataquality, &array)) {
        PyErr_SetString(PyExc_ValueError, "Trace record must be a tuple of (network, station, location, channel, starttime, endtime, samprate, dataquality, data).");
        return EXIT_FAILURE;
    }

    strncpy(mst->network, network, 10);
    strncpy(mst->station, station, 10);
    strncpy(mst->location, location, 10);
    strncpy(mst->channel, channel, 10);
    mst->network[10] = '\0';
    mst->station[10] = '\0';
    mst->location[10] ='\0';
    mst->channel[10] = '\0';
    mst->dataquality = dataquality[0];

    if (!PyArray_Check((PyArrayObject*) array)) {
        PyErr_SetString(PyExc_ValueError, "Data must be given as NumPy array.");
        return EXIT_FAILURE;
    }

    if (PyArray_ISBYTESWAPPED((PyArrayObject*) array)) {
        PyErr_SetString(PyExc_ValueError, "Data must be given in machine byte-order.");
        return EXIT_FAILURE;
    }

    numpytype = PyArray_TYPE((PyArrayObject*) array);
    switch (numpytype) {
        case NPY_INT16:
            assert(ms_samplesize('i') == 4);
            mst->sampletype = 'i';
            *msdetype = DE_INT16;
        case NPY_INT32:
            assert(ms_samplesize('i') == 4);
            mst->sampletype = 'i';
            *msdetype = DE_STEIM1;
            break;
        case NPY_BYTE:
            assert(ms_samplesize('a') == 1);
            mst->sampletype = 'a';
            *msdetype = DE_ASCII;
            break;
        case NPY_FLOAT32:
            assert(ms_samplesize('f') == 4);
            mst->sampletype = 'f';
            *msdetype = DE_FLOAT32;
            break;
        case NPY_FLOAT64:
            assert(ms_samplesize('d') == 8);
            mst->sampletype = 'd';
            *msdetype = DE_FLOAT64;
            break;
        default:
            PyErr_SetString(PyExc_ValueError, "Data must be of type float64, float32, int32 or int8.");
            return EXIT_FAILURE;
        }
    contiguous_array = PyArray_GETCONTIGUOUS((PyArrayObject*) array);

    length = PyArray_SIZE(contiguous_array);
    mst->numsamples = length;
    mst->samplecnt = length;

    mst->datasamples = calloc(length, ms_samplesize(mst->sampletype));
    if (memcpy(mst->datasamples, PyArray_DATA(contiguous_array), length*ms_samplesize(mst->sampletype)) == NULL) {
        Py_DECREF(contiguous_array);
        PyErr_SetString(PyExc_MemoryError, "Could not copy memory.");
        return EXIT_FAILURE;
    }

    Py_DECREF(contiguous_array);
    return EXIT_SUCCESS;
}


static void write_mseed_file(char *record, int reclen, void *outfile) {
    if (fwrite(record, reclen, 1, outfile) != 1 )
        fprintf(stderr, "Error writing mseed record to output file\n");
}


static PyObject*
mseed_store_traces (PyObject *m, PyObject *args, PyObject *kwds) {
    char          *filename;
    MSTrace       *mst = NULL;
    PyObject      *in_traces = NULL;
    PyObject      *in_trace = NULL;
    PyObject      *append = NULL;
    int           itr;
    int           msdetype = DE_FLOAT64;
    int64_t       psamples;
    size_t        record_length = 4096;
    
    FILE          *outfile;


    (void) m;

    static char *kwlist[] = {"traces", "filename", "record_length", "append", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Os|nO", kwlist, &in_traces, &filename, &record_length, &append))
        return NULL;

    if (!PySequence_Check(in_traces)) {
        PyErr_SetString(PyExc_TypeError, "Traces is not of sequence type.");
        return NULL;
    }

    if (!PyBool_Check(append)) {
        PyErr_SetString(PyExc_TypeError, "append must be a boolean");
        return NULL;
    }

    outfile = fopen(filename, append == Py_True ? "a" : "w");
    if (outfile == NULL) {
        PyErr_SetString(PyExc_OSError, "Error opening file.");
        return NULL;
    }

    for (itr=0; itr < PySequence_Length(in_traces); itr++) {

        in_trace = PySequence_GetItem(in_traces, itr);
        mst = mst_init(NULL);

        if (tuple2mst(in_trace, mst, &msdetype) != EXIT_SUCCESS) {
            mst_free(&mst);
            fclose(outfile);

            Py_DECREF(in_trace);
            return NULL;
        }

        Py_BEGIN_ALLOW_THREADS
        mst_pack(mst, &write_mseed_file, outfile, record_length, msdetype,
                 1, &psamples, 1, 0, NULL);
        mst_free(&mst);
        Py_END_ALLOW_THREADS

        Py_DECREF(in_trace);
    }
    fclose(outfile);
    Py_RETURN_NONE;
}

typedef struct MemoryInfo_t {
    void *head;
    size_t capacity;
    size_t nbytes_written;
} MemoryInfo;


static void copy_memory(char *record, int reclen, void *mem) {
    MemoryInfo *info = (MemoryInfo*) mem;
    if (memcpy(info->head, record, reclen) == NULL)
        fprintf(stderr, "Could not write to memory\n");
    info->head = (void *) (char *) info->head + reclen;
    info->nbytes_written += (size_t) reclen;
}

static PyObject*
mseed_bytes (PyObject *m, PyObject *args, PyObject *kwds) {
    MSTrace       *mst = NULL;
    MSRecord      *msr = NULL;
    PyObject      *in_traces = NULL;
    PyObject      *in_trace = NULL;
    PyObject      *mseed_data;
    Py_buffer     buffer;
    int           itr;
    int           msdetype = DE_FLOAT64;
    int64_t       psamples;
    size_t        nbytes;
    size_t        record_length = 4096;

    MemoryInfo   mem_info;

    static char *kwlist[] = {"traces", "nbytes", "record_length", NULL};

    (void) m;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "On|n", kwlist, &in_traces, &nbytes, &record_length))
        return NULL;

    if (!PySequence_Check(in_traces)) {
        PyErr_SetString(PyExc_TypeError, "traces is not of sequence type");
        return NULL;
    }

    mseed_data = PyBytes_FromStringAndSize(NULL, (Py_ssize_t) nbytes);
    if (mseed_data == NULL) {
        PyErr_SetString(PyExc_BufferError, "could not create bytes object");
        return NULL;
    }

    if (PyObject_GetBuffer(mseed_data, &buffer, PyBUF_SIMPLE) == -1) {
        PyErr_SetString(PyExc_BufferError, "could not get buffer");
        return NULL;
    }

    mem_info.head = buffer.buf;
    mem_info.capacity = nbytes;
    mem_info.nbytes_written = 0;

    msr = msr_init(NULL);
    msr->sequence_number = 0;

    for (itr=0; itr < PySequence_Length(in_traces); itr++) {
        in_trace = PySequence_GetItem(in_traces, itr);
        mst = mst_init(NULL);

        if (tuple2mst(in_trace, mst, &msdetype) != EXIT_SUCCESS) {
            mst_free(&mst);
            msr_free(&msr);
            Py_DECREF(in_trace);
            return NULL;
        }

        Py_BEGIN_ALLOW_THREADS
        mst_pack(mst, &copy_memory, (void *) &mem_info, record_length, msdetype,
                 1, &psamples, 1, 0, NULL);
        mst_free(&mst);
        Py_END_ALLOW_THREADS

        Py_DECREF(in_trace);
    }
    PyBuffer_Release(&buffer);
    msr_free(&msr);

    if (_PyBytes_Resize(&mseed_data, (Py_ssize_t) mem_info.nbytes_written) == -1) {
        PyErr_SetString(PyExc_BufferError, "could not resize bytes object");
        return NULL;
    }
    return mseed_data;
}

static PyMethodDef mseed_ext_methods[] = {
    {"get_traces", (PyCFunction) mseed_get_traces, METH_VARARGS | METH_KEYWORDS, 
     PyDoc_STR("get_traces(filename, dataflag)\n"
               "Get all traces stored in an mseed file.\n\n"
               "Returns a list of tuples, one tuple for each trace in the file. Each tuple\n"
               "has 9 elements:\n\n"
               "  (dataquality, network, station, location, channel,\n"
               "   startime, endtime, samprate, data)\n\n"
               "These come straight from the MSTrace data structure, defined and described\n"
               "in libmseed. If dataflag is True, `data` is a numpy array containing the\n"
               "data. If dataflag is False, the data is not unpacked and `data` is None.\n") },

    {"store_traces", (PyCFunction) mseed_store_traces, METH_VARARGS | METH_KEYWORDS, 
     PyDoc_STR("store_traces(traces, filename, record_length=4096)\n") },

    {"mseed_bytes", (PyCFunction) mseed_bytes, METH_VARARGS | METH_KEYWORDS, 
     PyDoc_STR("mseed_bytes(traces, nbytes, record_length=4096)\n") },

    {NULL, NULL, 0, NULL}        /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3

static int mseed_ext_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int mseed_ext_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "mseed_ext",
        NULL,
        sizeof(struct module_state),
        mseed_ext_methods,
        NULL,
        mseed_ext_traverse,
        mseed_ext_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_mseed_ext(void)

#else
#define INITERROR return

void
initmseed_ext(void)
#endif

{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("mseed_ext", mseed_ext_methods);
#endif
    import_array();

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("pyrocko.mseed_ext.MSeedError", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

    Py_INCREF(st->error);
    PyModule_AddObject(module, "MSeedError", st->error);
    PyModule_AddObject(module, "HPTMODULUS", PyLong_FromLong(HPTMODULUS));

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
