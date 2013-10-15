#define NPY_NO_DEPRECATED_API 7
    
#include "Python.h"
#include "numpy/arrayobject.h"

static PyObject *GSEError;

static char translate[128] = 
	{
		-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
		-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
		-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0,-1, 1,-1,-1, 2,
		 3, 4, 5, 6, 7, 8, 9,10,11,-1,-1,-1,-1,-1,-1,-1,
		12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,
		28,29,30,31,32,33,34,35,36,37,-1,-1,-1,-1,-1,-1,
		38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
		54,55,56,57,58,59,60,61,62,63,-1,-1,-1,-1,-1
	};

static int MODULUS = 100000000;


static PyObject* gse_checksum(PyObject *dummy, PyObject *args) {
        
    int checksum, length, i;
    PyObject *array = NULL;
    PyArrayObject *carray = NULL;
    int *data;

    if (!PyArg_ParseTuple(args, "O", &array )) {
        PyErr_SetString(GSEError, "usage checksum(array)" );
        return NULL;
    }
    if (!PyArray_Check(array)) {
        PyErr_SetString(GSEError, "Data must be given as NumPy array." );
        return NULL;
    }
    if (PyArray_TYPE(array) != NPY_INT32) {
        PyErr_SetString(GSEError, "Data must be 32-bit integers.");
        return NULL;
    }
    
    carray = PyArray_GETCONTIGUOUS((PyArrayObject*)array);
    length = PyArray_SIZE(carray);
    data = (int*)PyArray_DATA(carray);
    
    checksum = 0;
    for (i=0; i<length; i++) {
        checksum += data[i] % MODULUS;
        checksum %= MODULUS;
    }
    
    return Py_BuildValue("i", abs(checksum));
}

static PyObject* gse_decode_m6(PyObject *dummy, PyObject *args) {
    char *in_data;
    int *out_data = NULL;
    char *pos;
    char v;
    int sample, isample, ibyte, sign;
    int bufsize, previous1, previous2;
    char imore = 32, isign = 16;
    PyObject      *array = NULL;
    npy_intp      array_dims[1] = {0};
    
    
    if (!PyArg_ParseTuple(args, "si", &in_data, &bufsize)) {
        PyErr_SetString(GSEError, "invalid arguments in decode_m6(data, sizehint)" );
        return NULL;
    }

    if (bufsize <= 1) bufsize = 64;

    out_data = (int*)malloc(bufsize*sizeof(int));
    if (out_data == NULL) {
        PyErr_SetString(GSEError, "cannot allocate memory" );
        return NULL;
    }

    pos = in_data;
    sample = 0;
    isample = 0;
    ibyte = 0;
    sign = 1;
    previous1 = 0;
    previous2 = 0;
    while (*pos != '\0') {
        v = translate[*pos & 0x7F];
        if (v != -1) {
            if (ibyte == 0) sign = (v & isign) ? -1 : 1;
            
            sample += v & ((ibyte == 0) ? 0xf : 0x1f);
            if ( (v & imore) == 0) {
                if (isample >= bufsize) {
                    bufsize = isample*2;
                    out_data = (int*)realloc(out_data, sizeof(int) * bufsize);
                    if (out_data == NULL) {
                        free(out_data);
                        PyErr_SetString(GSEError, "cannot allocate memory" );
                        return NULL;
                    }
                }
                previous1 = previous1 + sign * sample;
                out_data[isample] = previous2 = previous2 + previous1;
                isample++;
                sample = 0;
                ibyte = 0;
            } else {
                sample *= 32;
                ibyte++;
            }
        }
        pos++;
    }
    array_dims[0] = isample;
    array = PyArray_SimpleNewFromData(1, array_dims, NPY_INT32, out_data);
    return Py_BuildValue("N", array);
}

static PyMethodDef GSEMethods[] = {
    {"decode_m6",  gse_decode_m6, METH_VARARGS, 
        "Decode m6 encoded GSE data." },
        
    {"checksum", gse_checksum, METH_VARARGS,
        "Calculate GSE checksum."},
    
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initgse_ext(void)
{
    PyObject *m;

    m = Py_InitModule("gse_ext", GSEMethods);
    if (m == NULL) return;
    import_array();

    GSEError = PyErr_NewException("gse_ext.error", NULL, NULL);
    Py_INCREF(GSEError);  /* required, because other code could remove `error` 
                               from the module, what would create a dangling
                               pointer. */
    PyModule_AddObject(m, "GSEError", GSEError);
}
