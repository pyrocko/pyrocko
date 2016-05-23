#define NPY_NO_DEPRECATED_API 7

#include "Python.h"
#include "numpy/arrayobject.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef npy_float64 float64_t;

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

typedef enum {
    SUCCESS = 0,
    SINGULARITY,
    BAD_ARRAY,
} ahfullgreen_error_t;

const char* ahfullgreen_error_names[] = {
    "SUCCESS",
    "SINGULARITY",
    "BAD_ARRAY",
};

static PyObject *Error;


int good_array(PyObject* o, int typenum, ssize_t size_want) {
    if (!PyArray_Check(o)) {
        PyErr_SetString(Error, "not a NumPy array" );
        return 0;
    }

    if (PyArray_TYPE((PyArrayObject*)o) != typenum) {
        PyErr_SetString(Error, "array of unexpected type");
        return 0;
    }

    if (!PyArray_ISCARRAY((PyArrayObject*)o)) {
        PyErr_SetString(Error, "array is not contiguous or not well behaved");
        return 0;
    }

    if (size_want != -1 && size_want != PyArray_SIZE((PyArrayObject*)o)) {
        PyErr_SetString(Error, "array is of wrong size");
        return 0;
    }

    return 1;
}

static ahfullgreen_error_t numpy_or_none_to_c(
        PyObject* o, ssize_t size_want, double **arr, size_t *size) {

    if (o == Py_None) {
        if (size_want > 0) {
            PyErr_SetString(Error, "array is of wrong size");
            return BAD_ARRAY;
        }
        *arr = NULL;
        *size = 0;
    } else {
        if (!good_array(o, NPY_DOUBLE, -1)) return BAD_ARRAY;
        *arr = PyArray_DATA((PyArrayObject*)o);
        *size = PyArray_SIZE((PyArrayObject*)o);
    }

    return SUCCESS;
}

static ahfullgreen_error_t add_seismogram(
        double vp,
        double vs,
        double density,
        double qp,
        double qs,
        double *x,
        double *f,
        double *m6,
        int out_type,  // 0: time trace, 1: spectra
        int out_quantity,  // 0: displacement, 1: velocity, 2: acceleration
        double out_delta,
        double out_offset,
        size_t out_size,
        double *out_x,
        double *out_y,
        double *out_z
        ) {

    double r, r2, r4;
    double gamma[3];
    double *out[3];

    int p_map[6] = {0, 1, 2, 0, 0, 1};
    int q_map[6] = {0, 1, 2, 1, 2, 2};
    int pq_factor[6] = {1, 1, 1, 2, 2, 2};

    int n, p, q;
    int m;

    double a1, a2, a3, a4, a5, a6, a7, a8;

    r = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
    if (r == 0.0) {
        return SINGULARITY;
    }

    for (n=0; n<3; n++) gamma[n] = x[n]/r;

    out[0] = out_x;
    out[1] = out_y;
    out[2] = out_z;

    for (n=0; n<3; n++) {
        if (out[n] == NULL) continue;

        for (m=0; m<6; m++) {
            p = p_map[m];
            q = q_map[m];
            r2 = r*r
            r4 = r2*r2

            a1 = pq_factor[m] * (
                15. * gamma[n] * gamma[p] * gamma[q] -
                3. * gamma[n] * (p==q) -
                3. * gamma[p] * (n==q) -
                3. * gamma[q] * (n==p)) /
                (4. * M_PI * density * r4);

            a2 = pq_factor[m] * (
                6. * gamma[n] * gamma[p] * gamma[q] -
                gamma[n] * (p==q) -
                gamma[p] * (n==q) -
                gamma[q] * (n==p)) /
                (4. * M_PI * density * vp*vp * r2);

            a3 = - pq_factor[m] * (
                6. * gamma[n] * gamma[p] * gamma[q] -
                gamma[n] * (p==q) -
                gamma[p] * (n==q) -
                2. * gamma[q] * (n==p)) /
                (4. * M_PI * density * vs*vs * r2);

            a4 = pq_factor[m] * (gamma[n] * gamma[p] * gamma[q]) /
                (4. * M_PI * density * vp*vp*vp * r);

            a5 = - pq_factor[m] * (gamma[q] * (gamma[n] * gamma[p] - (n==p))) /
                (4. * M_PI * density * vs*vs*vs * r);

        }

        for (p=0; p<3; p++) {

            a6 = (3. * gamma[n] * gamma[p] - (n==p)) /
                (4. * M_PI * density * r2 * r);

            a7 = (gamma[n] * gamma[p]) /
                (4. * M_PI * density * vp * vp * r);

            a8 = - (gamma[n] * gamma[p] - (n==p)) /
                (4. * M_PI * density * vs * vs * r);

        }
    }

    return SUCCESS;
}

static PyObject* w_add_seismogram(PyObject *dummy, PyObject *args) {
    double vp, vs, density, qp, qs;
    double *x;
    double *f;
    double *m6;
    int out_type;  // 0: time trace, 1: spectra
    int out_quantity;  // 0: displacement, 1: velocity, 2: acceleration
    double out_delta;
    double out_offset;

    size_t out_size, out_x_size, out_y_size, out_z_size;

    PyObject *x_arr;
    PyObject *f_arr;
    PyObject *m6_arr;

    PyObject *out_x_arr;
    PyObject *out_y_arr;
    PyObject *out_z_arr;

    double *out_x;
    double *out_y;
    double *out_z;
    ahfullgreen_error_t err;
    size_t dummy_size;

    (void)dummy; /* silence warning */

    if (!PyArg_ParseTuple(args, "dddddOOOiiddOOO",
            &vp, &vs, &density, &qp, &qs, &x_arr, &f_arr, &m6_arr,
            &out_type, &out_quantity, &out_delta, &out_offset,
            &out_x_arr, &out_y_arr, &out_z_arr)) {

        PyErr_SetString(Error,
            "usage: add_seismogram(vp, vs, density, qp, qs, x, f, m6, "
            "out_type, out_quantity, out_delta, out_offset, "
            "out_x, out_y, out_z)");

        return NULL;
    }

    if (SUCCESS != numpy_or_none_to_c(out_x_arr, -1, &out_x, &out_x_size)) return NULL;
    if (SUCCESS != numpy_or_none_to_c(out_y_arr, -1, &out_y, &out_y_size)) return NULL;
    if (SUCCESS != numpy_or_none_to_c(out_z_arr, -1, &out_z, &out_z_size)) return NULL;

    out_size = max(max(out_x_size, out_y_size), out_z_size);

    if ((!(out_x_size == 0 || out_x_size == out_size)) ||
        (!(out_y_size == 0 || out_y_size == out_size)) ||
        (!(out_z_size == 0 || out_z_size == out_size))) {

        PyErr_SetString(Error, "differing output array sizes");
        return NULL;
    }

    if (SUCCESS != numpy_or_none_to_c(x_arr, 3, &x, &dummy_size)) return NULL;
    if (SUCCESS != numpy_or_none_to_c(f_arr, 3, &f, &dummy_size)) return NULL;
    if (SUCCESS != numpy_or_none_to_c(m6_arr, 6, &m6, &dummy_size)) return NULL;

    err = add_seismogram(
        vp, vs, density, qp, qs, x, f, m6,
        out_type, out_quantity, out_delta, out_offset, out_size,
        out_x, out_y, out_z);

    if (err != SUCCESS) {
        PyErr_SetString(Error, ahfullgreen_error_names[err]);
        return NULL;
    }

    return Py_BuildValue("");
}

static PyMethodDef StoreExtMethods[] = {
    {"add_seismogram", w_add_seismogram, METH_VARARGS,
        "Add seismogram to array." },

    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initahfullgreen_ext(void)
{
    PyObject *m;

    m = Py_InitModule("ahfullgreen_ext", StoreExtMethods);
    if (m == NULL) return;
    import_array();

    Error = PyErr_NewException("pyrocko.ahfullgreen_ext.Error", NULL, NULL);
    Py_INCREF(Error);  /* required, because other code could remove `error`
                               from the module, what would create a dangling
                               pointer. */
    PyModule_AddObject(m, "Error", Error);
}

