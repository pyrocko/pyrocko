#define NPY_NO_DEPRECATED_API 7

#include "Python.h"
#include "numpy/arrayobject.h"

static PyObject *IMSError;

static char translate[128] = {
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

PyArrayObject *get_good_array(PyObject *array) {

    if (!PyArray_Check(array)) {
        PyErr_SetString(IMSError, "Data must be given as NumPy array." );
        return NULL;
    }
    if (PyArray_ISBYTESWAPPED((PyArrayObject*)array)) {
        PyErr_SetString(IMSError, "Data must be given in machine byte-order.");
        return NULL;
    }
    if (PyArray_TYPE((PyArrayObject*)array) != NPY_INT32) {
        PyErr_SetString(IMSError, "Data must be 32-bit integers.");
        return NULL;
    }
    return PyArray_GETCONTIGUOUS((PyArrayObject*)array);
}

static PyObject* ims_checksum(PyObject *dummy, PyObject *args) {

    int checksum, length, i;
    PyObject *array = NULL;
    PyArrayObject *carray = NULL;
    int *data;

    (void)dummy; /* silence warning */

    if (!PyArg_ParseTuple(args, "O", &array )) {
        PyErr_SetString(IMSError, "usage checksum(array)" );
        return NULL;
    }

    carray = get_good_array(array);
    if (carray == NULL) {
        return NULL;
    }

    length = PyArray_SIZE(carray);
    data = (int*)PyArray_DATA(carray);

    checksum = 0;
    for (i=0; i<length; i++) {
        checksum += data[i] % MODULUS;
        checksum %= MODULUS;
    }
    Py_DECREF(carray);

    return Py_BuildValue("i", abs(checksum));
}

static PyObject* ims_decode_cm6(PyObject *dummy, PyObject *args) {
    char *in_data;
    int *out_data = NULL;
    int *out_data_new = NULL;
    char *pos;
    char v;
    int sample, isample, ibyte, sign;
    int bufsize, previous1, previous2;
    char imore = 32, isign = 16;
    PyObject      *array = NULL;
    npy_intp      array_dims[1] = {0};

    (void)dummy; /* silence warning */

    if (!PyArg_ParseTuple(args, "si", &in_data, &bufsize)) {
        PyErr_SetString(IMSError, "invalid arguments in decode_cm6(data, sizehint)" );
        return NULL;
    }

    if (bufsize <= 1) bufsize = 64;

    out_data = (int*)malloc(bufsize*sizeof(int));
    if (out_data == NULL) {
        PyErr_SetString(IMSError, "cannot allocate memory" );
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
            if ((v & imore) == 0) {
                if (isample >= bufsize) {
                    bufsize = isample*2;
                    out_data_new = (int*)realloc(out_data, sizeof(int) * bufsize);
                    if (out_data_new == NULL) {
                        free(out_data);
                        PyErr_SetString(IMSError, "cannot allocate memory" );
                        return NULL;
                    }
                    out_data = out_data_new;
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
    array = PyArray_SimpleNew(1, array_dims, NPY_INT32);
    memcpy(PyArray_DATA((PyArrayObject*)array), out_data, isample*sizeof(int32_t));
    free(out_data);
    return Py_BuildValue("N", array);
}

static PyObject* ims_encode_cm6(PyObject *dummy, PyObject *args) {
    PyObject *array = NULL;
    PyObject *string = NULL;
    PyArrayObject *contiguous_array = NULL;
    int *in_data;
    char *out_data = NULL;
    int sample;
    int v;
    int isign, imore;
    size_t nsamples, bufsize, isample, ipos, iout, i;
    char temp;
    char rtranslate[64];

    (void)dummy; /* silence warning */

    for (i=0; i<128; i++) {
        if (translate[i] != -1) {
            rtranslate[(int)translate[i]] = i;
        }
    }

    if (!PyArg_ParseTuple(args, "O", &array)) {
        PyErr_SetString(IMSError, "invalid arguments in encode_cm6(data)");
        return NULL;
    }

    contiguous_array = get_good_array(array);
    if (contiguous_array == NULL) {
        return NULL;
    }

    nsamples  = PyArray_SIZE(contiguous_array);
    in_data = PyArray_DATA(contiguous_array);

    if (nsamples >= SIZE_MAX / 7) {
        PyErr_SetString(IMSError, "too many samples.");
        Py_DECREF(contiguous_array);
        return NULL;
    }
    bufsize = nsamples * 7;
    out_data = (char*)malloc(bufsize);
    if (out_data == NULL) {
        PyErr_SetString(IMSError, "cannot allocate memory");
        Py_DECREF(contiguous_array);
        return NULL;
    }

    iout = 0;
    for (isample=0; isample<nsamples; isample++) {
        sample = in_data[isample];
        if (isample >= 1) sample -= 2 * in_data[isample-1];
        if (isample >= 2) sample += in_data[isample-2];
        isign = sample < 0 ? 16 : 32;
        sample = abs(sample);
        imore = 0;
        ipos = iout;
        while (isign != 0) {
            v = sample & 31;
            sample /= 32;
            if (sample == 0 && v < 16) {
                v += isign & 16;
                isign = 0;
            }
            v += imore;
            imore = 32;

            if (iout >= bufsize) {
                free(out_data);
                PyErr_SetString(IMSError,
                    "some assumption of the programmer was wrong...");
                Py_DECREF(contiguous_array);
                return NULL;
            }

            out_data[iout] = rtranslate[v];
            iout++;
        }
        for (i=0; i<(iout-ipos)/2; i++) {
            temp = out_data[ipos+i];
            out_data[ipos+i] = out_data[iout-1-i];
            out_data[iout-1-i] = temp;
        }
    }

    string = PyString_FromStringAndSize(out_data, iout);
    free(out_data);

    if (string == NULL) {
        PyErr_SetString(IMSError, "cannot create output string");
        Py_DECREF(contiguous_array);
        return NULL;
    }

    Py_DECREF(contiguous_array);
    return Py_BuildValue("N", string);
}

static PyMethodDef IMSMethods[] = {
    {"decode_cm6",  ims_decode_cm6, METH_VARARGS,
        "Decode CM6 encoded IMS/GSE data." },

    {"encode_cm6",  ims_encode_cm6, METH_VARARGS,
        "Encode integers to CM6." },

    {"checksum", ims_checksum, METH_VARARGS,
        "Calculate IMS/GSE checksum."},

    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initims_ext(void)
{
    PyObject *m;

    m = Py_InitModule("ims_ext", IMSMethods);
    if (m == NULL) return;
    import_array();

    IMSError = PyErr_NewException("ims_ext.error", NULL, NULL);
    Py_INCREF(IMSError);  /* required, because other code could remove `error`
                               from the module, what would create a dangling
                               pointer. */
    PyModule_AddObject(m, "IMSError", IMSError);
}
