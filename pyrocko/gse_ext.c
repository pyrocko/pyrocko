#include "Python.h"
#include "numpy/arrayobject.h"

static PyObject *GSEError;

static PyObject* decode_m6(PyObject *dummy, PyObject *args) {
    char *in_data;
    char *out_data;
    
    if (!PyArg_ParseTuple(args, "s", &in_data)) {
        PyErr_SetString(GSEError, "invalid arguments in decode_m6(data)" );
        return NULL;
    }
    out_data = in_data;
    
    return Py_BuildValue("s", out_data);
}

static PyMethodDef GSEMethods[] = {
    {"decode_m6",  decode_m6, METH_VARARGS, 
    "Decode m6 encoded GSE data." },
    
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initgse_ext(void)
{
    PyObject *m;
    PyObject *hptmodulus;

    m = Py_InitModule("gse_ext", GSEMethods);
    if (m == NULL) return;
    import_array();

    GSEError = PyErr_NewException("gse_ext.error", NULL, NULL);
    Py_INCREF(GSEError);  /* required, because other code could remove `error` 
                               from the module, what would create a dangling
                               pointer. */
    PyModule_AddObject(m, "GSEError", GSEError);
}
