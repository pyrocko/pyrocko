
/* Copyright (c) 2009, Sebastian Heimann <sebastian.heimann@zmaw.de>

  This file is part of pymseed. For licensing information please see the file
  COPYING which is included with pyevalresp. */

#define NPY_NO_DEPRECATED_API 7

#include "Python.h"
#include "numpy/arrayobject.h"

#include <evresp.h>
#include <assert.h>

static PyObject *EvalrespError;

#define BUFSIZE 1024


static PyObject*
evresp_wrapper (PyObject *dummy, PyObject *args)
{
    char *sta_list, *cha_list, *units, *file, *verbose;
    char *net_code, *locid, *rtype;
    int nfreqs;
    int start_stage = -1, stop_stage = 0, stdio_flag = 0;
    int listinterp_out_flag = 0, listinterp_in_flag = 0;
    double listinterp_tension = 1000.0;
    char *datime;
    struct response *first;

    PyArrayObject *freqs_array = NULL, *freqs_array_cont = NULL;
    PyObject      *rvec_array = NULL;
    PyObject      *elem, *out_list;
    npy_intp      array_dims[1] = {0};

    if (!PyArg_ParseTuple(args, "sssssssOssiiiiid",
                            &sta_list,
                            &cha_list,
                            &net_code,
                            &locid,
                            &datime,
                            &units,
                            &file,
                            &freqs_array,
                            &rtype,
                            &verbose,
                            &start_stage,
                            &stop_stage,
                            &stdio_flag,
                            &listinterp_out_flag,
                            &listinterp_in_flag,
                            &listinterp_tension)) {
        PyErr_SetString(EvalrespError, "usage: evalresp(sta_list, cha_list, net_code, locid, datime, units, file, freqs_array, "
                                       "rtype, verbose, start_stage, stop_stage, stdio_flag, "
                                       "listinterp_out_flag, listinterp_in_flag, listinterp_tension)" );
        return NULL;
    }

    if (!PyArray_Check(freqs_array)) {
        PyErr_SetString(EvalrespError, "Frequencies must be given as NumPy array." );
        return NULL;
    }

    assert( sizeof(double) == 8 );
    if (!PyArray_TYPE(freqs_array) == NPY_FLOAT64) {
        PyErr_SetString(EvalrespError, "Frequencies must be of type double.");
        return NULL;
    }

    freqs_array_cont = PyArray_GETCONTIGUOUS((PyArrayObject*)freqs_array);
    nfreqs = PyArray_SIZE(freqs_array_cont);

    first = evresp_itp(sta_list, cha_list, net_code, locid, datime, units, file,
                      PyArray_DATA(freqs_array_cont), nfreqs, rtype, verbose, start_stage, stop_stage,
                      stdio_flag, listinterp_out_flag, listinterp_in_flag,
                      listinterp_tension);

    Py_DECREF(freqs_array_cont);

    if (!first) {
        PyErr_SetString(EvalrespError, "Function evresp() failed" );
        return NULL;
    }

    out_list = Py_BuildValue("[]");



    while (first) {

        array_dims[0] = nfreqs;
        rvec_array = PyArray_SimpleNew(1, array_dims, NPY_COMPLEX128);
        memcpy( PyArray_DATA((PyArrayObject*)rvec_array), first->rvec, nfreqs*16 );

        elem = Py_BuildValue("(s,s,s,s,N)",
            first->station, first->network, first->locid, first->channel, rvec_array);

        PyList_Append(out_list, elem);
        Py_DECREF(elem);

        first = first->next;
    }
    free_response(first);

    return out_list;
}


static PyMethodDef EVALRESPMethods[] = {
    {"evalresp",  evresp_wrapper, METH_VARARGS,
    "" },
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC
initevalresp_ext(void)
{
    PyObject *m;

    m = Py_InitModule("evalresp_ext", EVALRESPMethods);
    if (m == NULL) return;
    import_array();

    EvalrespError = PyErr_NewException("evalresp_ext.error", NULL, NULL);
    Py_INCREF(EvalrespError);  /* required, because other code could remove `error`
                               from the module, what would create a dangling
                               pointer. */
    PyModule_AddObject(m, "EvalrespError", EvalrespError);
}
