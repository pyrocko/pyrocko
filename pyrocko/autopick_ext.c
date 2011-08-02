#define NPY_NO_DEPRECATED_API

#include "Python.h"
#include "numpy/arrayobject.h"

static PyObject *AutoPickError;

int autopick_recursive_stalta( int ns, int nl, double ks, double kl, int initialize, int nsamples, 
                                    float *in_data, float *out_data) {
    int i;

    for (i=0; i<nsamples; i++) {
        out_data[i] = in_data[i] + 1.0;
    }
    return 0;
}


static PyObject* autopick_recursive_stalta_wrapper(PyObject *dummy, PyObject *args) {
    PyObject *in_array_obj;
    PyArrayObject *in_array = NULL;
    PyObject *out_array_obj = NULL;
    int ns, nl, initialize, nsamples; 
    double ks, kl; 
    float *out_data = NULL;

    if (!PyArg_ParseTuple(args, "iiddiO", &ns, &nl, &ks, &kl, &initialize, &in_array_obj)) {
        PyErr_SetString(AutoPickError, "invalid arguments in recursive_stalta(ns, nl, ks, kl, initialize, in_data)" );
        return NULL;
    }
    in_array = (PyArrayObject*)PyArray_ContiguousFromAny(in_array_obj, NPY_FLOAT32, 1, 1);
    if (in_array == NULL) {
        PyErr_SetString(AutoPickError, "cannot create a contiguous float array from in_data." );
        return NULL;
    }

    nsamples = PyArray_SIZE(in_array);

    out_data = (float*)malloc(nsamples*sizeof(float));
    if (out_data == NULL) {
        PyErr_SetString(AutoPickError, "cannot allocate memory" );
        return NULL;
    }

    if (0 != autopick_recursive_stalta(ns, nl, ks, kl, initialize, nsamples, (float*)PyArray_DATA(in_array), out_data)) {
        free(out_data);
        PyErr_SetString(AutoPickError, "running STA/LTA failed.");
        return NULL;
    }

    out_array_obj = PyArray_SimpleNewFromData(1, PyArray_DIMS(in_array), PyArray_TYPE(in_array), out_data);
    if (out_array_obj == NULL) {
        free(out_data);
        PyErr_SetString(AutoPickError, "failed to create output array object.");
        return NULL;
    }
    return Py_BuildValue("N", out_array_obj);
}

static PyMethodDef AutoPickMethods[] = {
    {"recursive_stalta",  autopick_recursive_stalta_wrapper, METH_VARARGS, 
        "Recursive STA/LTA picker." },
        
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initautopick_ext(void)
{
    PyObject *m;

    m = Py_InitModule("autopick_ext", AutoPickMethods);
    if (m == NULL) return;
    import_array();

    AutoPickError = PyErr_NewException("autopick_ext.error", NULL, NULL);
    Py_INCREF(AutoPickError);  /* required, because other code could remove `error` 
                               from the module, what would create a dangling
                               pointer. */
    PyModule_AddObject(m, "AutoPickError", AutoPickError);
}

