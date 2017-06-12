
#define NPY_NO_DEPRECATED_API 7


#include "Python.h"
#include "numpy/arrayobject.h"

#include <stdlib.h>
#if defined(_OPENMP)
    # include <omp.h>
#endif
#include <stdio.h>
#include <float.h>

#define CHUNKSIZE 10

static PyObject *ParstackError;


int parstack_config(
        size_t narrays,
        int *offsets,
        size_t *lengths,
        size_t nshifts,
        int *shifts,
        double *weights,
        int method,
        size_t *lengthout,
        int *offsetout);

int parstack(
        size_t narrays,
        double **arrays,
        int *offsets,
        size_t *lengths,
        size_t nshifts,
        int *shifts,
        double *weights,
        int method,
        size_t lengthout,
        int offsetout,
        double *result,
        int nparallel);


int min(int a, int b) {
    return (a < b) ? a : b;
}

int max(int a, int b) {
    return (a > b) ? a : b;
}

double dmax(double a, double b) {
    return (a > b) ? a : b;
}

#define SUCCESS 0
#define NODATA 1
#define INVALID 2

int parstack_config(
        size_t narrays,
        int *offsets,
        size_t *lengths,
        size_t nshifts,
        int *shifts,
        double *weights,
        int method,
        size_t *lengthout,
        int *offsetout) {

    (void)weights;  /* silence warnings */
    (void)method;

    if (narrays < 1) {
        return NODATA;
    }

    int imin, imax, istart, iend;
    size_t iarray, ishift;

    imin = offsets[0] + shifts[0];
    imax = imin + lengths[0];
    for (iarray=0; iarray<narrays; iarray++) {
        for (ishift=0; ishift<nshifts; ishift++) {
            istart = offsets[iarray] + shifts[ishift*narrays + iarray];
            iend = istart + lengths[iarray];
            imin = min(imin, istart);
            imax = max(imax, iend);
        }
    }

    *lengthout = imax - imin;
    *offsetout = imin;

    return SUCCESS;
}

int parstack(
        size_t narrays,
        double **arrays,
        int *offsets,
        size_t *lengths,
        size_t nshifts,
        int *shifts,
        double *weights,
        int method,
        size_t lengthout,
        int offsetout,
        double *result,
        int nparallel) {

	(void) nparallel;
    int imin, istart, ishift;
    size_t iarray, nsamp, i;
    double weight;
    int chunk;
    double *temp;
    double m;

    if (narrays < 1) {
        return NODATA;
    }

    if (nshifts > INT_MAX) {
        return INVALID;
    }

    imin = offsetout;
    nsamp = lengthout;

    chunk = CHUNKSIZE;

    Py_BEGIN_ALLOW_THREADS
    if (method == 0) {
	#if defined(_OPENMP)
        #pragma omp parallel private(ishift, iarray, i, istart, weight) num_threads(nparallel)
	#endif
        {

	#if defined(_OPENMP)
        #pragma omp for schedule(dynamic, chunk) nowait
	#endif
        for (ishift=0; ishift<(int)nshifts; ishift++) {
            for (iarray=0; iarray<narrays; iarray++) {
                istart = offsets[iarray] + shifts[ishift*narrays + iarray];
                weight = weights[ishift*narrays + iarray];
                for (i=(size_t)max(0, imin - istart); i<(size_t)max(0, min(nsamp - istart + imin, lengths[iarray])); i++) {
                    result[ishift*nsamp + istart-imin+i] += arrays[iarray][i] * weight;
                }
            }
        }
        }

    } else if (method == 1) {

	#if defined(_OPENMP)
        #pragma omp parallel private(ishift, iarray, i, istart, weight, temp, m)
	#endif
        {
        temp = (double*)calloc(nsamp, sizeof(double));
	#if defined(_OPENMP)
        #pragma omp for schedule(dynamic,chunk) nowait
	#endif
        for (ishift=0; ishift<(int)nshifts; ishift++) {
            for (i=0; i<nsamp; i++) {
                temp[i] = 0.0;
            }
            for (iarray=0; iarray<narrays; iarray++) {
                istart = offsets[iarray] + shifts[ishift*narrays + iarray];
                weight = weights[ishift*narrays + iarray];
                for (i=(size_t)max(0, imin - istart); i<(size_t)max(0, min(nsamp - istart + imin, lengths[iarray])); i++) {
                    temp[istart-imin+i] += arrays[iarray][i] * weight;
                }
            }
            m = 0.;
            for (i=0; i<nsamp; i++) {
                //m += temp[i]*temp[i];
                m = dmax(m, temp[i]);
            }
            result[ishift] = m;
        }
        free(temp);
        }
    }
    Py_END_ALLOW_THREADS
    return SUCCESS;
}

int argmax(double *arrayin, int *arrayout, npy_intp *shape){
    int m, n, im, in, imax, imm;
    double vmax;
    n = shape[1];
    m = shape[0];

    for (in=0; in<n; in++){
        imax = 0;
        vmax = DBL_MIN;
        for (im=0; im<m; im++){
            if (arrayin[im*n + in] > vmax){
                vmax = arrayin[im*n + in];
                imax = im;
            }
        }
        arrayout[in] = imax;
    }
    return SUCCESS;
}


int good_array(PyObject* o, int typenum) {
    if (!PyArray_Check(o)) {
        PyErr_SetString(ParstackError, "not a NumPy array" );
        return 0;
    }

    if (PyArray_TYPE((PyArrayObject*)o) != typenum) {
        PyErr_SetString(ParstackError, "array of unexpected type");
        return 0;
    }

    if (!PyArray_ISCARRAY((PyArrayObject*)o)) {
        PyErr_SetString(ParstackError, "array is not contiguous or not well behaved");
        return 0;
    }

    return 1;
}

static PyObject* w_parstack(PyObject *dummy, PyObject *args) {

    PyObject *arrays, *offsets, *shifts, *weights, *arr;
    PyObject *result;
    int method, nparallel;
    size_t narrays, nshifts, nweights;
    size_t *clengths;
    size_t lengthout;
    int offsetout;
    int lengthout_arg;
    int *coffsets, *cshifts;
    double *cweights, *cresult;
    double **carrays;
    npy_intp array_dims[1];
    size_t i;
    int err;

    (void)dummy; /* silence warning */

    carrays = NULL;
    clengths = NULL;

    if (!PyArg_ParseTuple(args, "OOOOiiiOi", &arrays, &offsets, &shifts,
                          &weights, &method, &lengthout_arg, &offsetout, &result, &nparallel)) {

        PyErr_SetString(
            ParstackError,
            "usage parstack(arrays, offsets, shifts, weights, method, lengthout, offsetout, result, nparallel)" );

        return NULL;
    }
    if (!good_array(offsets, NPY_INT)) return NULL;
    if (!good_array(shifts, NPY_INT)) return NULL;
    if (!good_array(weights, NPY_DOUBLE)) return NULL;
    if (result != Py_None && !good_array(result, NPY_DOUBLE)) return NULL;

    coffsets = PyArray_DATA((PyArrayObject*)offsets);
    narrays = PyArray_SIZE((PyArrayObject*)offsets);

    cshifts = PyArray_DATA((PyArrayObject*)shifts);
    nshifts = PyArray_SIZE((PyArrayObject*)shifts);

    cweights = PyArray_DATA((PyArrayObject*)weights);
    nweights = PyArray_SIZE((PyArrayObject*)weights);

    nshifts /= narrays;
    nweights /= narrays;

    if (nshifts != nweights) {
        PyErr_SetString(ParstackError, "weights.size != shifts.size" );
        return NULL;
    }

    if (!PyList_Check(arrays)) {
        PyErr_SetString(ParstackError, "arg #1 must be a list of NumPy arrays.");
        return NULL;
    }

    if ((size_t)PyList_Size(arrays) != narrays) {
        PyErr_SetString(ParstackError, "len(offsets) != len(arrays)");
        return NULL;
    }

    carrays = (double**)calloc(narrays, sizeof(double*));
    if (carrays == NULL) {
        PyErr_SetString(ParstackError, "alloc failed");
        return NULL;
    }

    clengths = (size_t*)calloc(narrays, sizeof(size_t));
    if (clengths == NULL) {
        PyErr_SetString(ParstackError, "alloc failed");
        free(carrays);
        return NULL;
    }

    for (i=0; i<narrays; i++) {
        arr = PyList_GetItem(arrays, i);
        if (!good_array(arr, NPY_DOUBLE)) {
            free(carrays);
            free(clengths);
            return NULL;
        }
        carrays[i] = PyArray_DATA((PyArrayObject*)arr);
        clengths[i] = PyArray_SIZE((PyArrayObject*)arr);
    }
    if (lengthout_arg < 0) {
        err = parstack_config(narrays, coffsets, clengths, nshifts, cshifts,
                              cweights, method, &lengthout, &offsetout);

        if (err != 0) {
            PyErr_SetString(ParstackError, "parstack_config() failed");
            free(carrays);
            free(clengths);
            return NULL;
        }
    } else {
        lengthout = (size_t)lengthout_arg;
    }

    if (method == 0) {
        array_dims[0] = nshifts * lengthout;
    } else {
        array_dims[0] = nshifts;
    }

    if (result != Py_None) {
        if (PyArray_SIZE((PyArrayObject*)result) != array_dims[0]) {
            free(carrays);
            free(clengths);
            return NULL;
        }
        Py_INCREF(result); 
    } else {
        result = PyArray_SimpleNew(1, array_dims, NPY_FLOAT64);
        cresult = PyArray_DATA((PyArrayObject*)result);

        for (i=0; i<(size_t)array_dims[0]; i++) {
            cresult[i] = 0.0;
        }

        if (result == NULL) {
            free(carrays);
            free(clengths);
            return NULL;
        }
    }
    cresult = PyArray_DATA((PyArrayObject*)result);

    err = parstack(narrays, carrays, coffsets, clengths, nshifts, cshifts,
                   cweights, method, lengthout, offsetout, cresult, nparallel);

    if (err != 0) {
        PyErr_SetString(ParstackError, "parstack() failed");
        free(carrays);
        free(clengths);
        Py_DECREF(result);
        return NULL;
    }

    free(carrays);
    free(clengths);
    return Py_BuildValue("Ni", result, offsetout);
}

static PyObject* w_argmax(PyObject *dummy, PyObject *args) {
    PyObject *arrayin;
    PyObject *result;
    double *carrayin;
    int *cresult;
    npy_intp *shape, shapeout[1];
    size_t i;
    int err;

    (void)dummy; /* silence warning */

    if (!PyArg_ParseTuple(args, "O", &arrayin)) {
        PyErr_SetString(ParstackError, "usage argmax(array)");
    }

    if (!good_array(arrayin, NPY_DOUBLE)) return NULL;

    shape = PyArray_SHAPE((PyArrayObject*)arrayin);

    carrayin = PyArray_DATA((PyArrayObject*)arrayin);

    shapeout[0] = shape[1];

    result = PyArray_SimpleNew(1, shapeout, NPY_UINT32);
    cresult = PyArray_DATA((PyArrayObject*)result);

    for (i=0; i<(size_t)shapeout[0]; i++){
        cresult[i] = 0;
    }

    err = argmax(carrayin, cresult, shape);

    if(err != 0){
        Py_DECREF(cresult);
        return NULL;
    }

    return Py_BuildValue("N", (PyArrayObject*) result);
}


static PyMethodDef ParstackMethods[] = {
    {"parstack",  w_parstack, METH_VARARGS,
        "Parallel weight-and-delay stacking" },

    {"argmax", w_argmax, METH_VARARGS,
        "argmax of 2D numpy array along axis=0" },

    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC initparstack_ext(void) {
    PyObject *m;

    m = Py_InitModule("parstack_ext", ParstackMethods);
    if (m == NULL) return;
    import_array();

    ParstackError = PyErr_NewException("parstack_ext.error", NULL, NULL);
    Py_INCREF(ParstackError);  /* required, because other code could remove `error` 
                               from the module, what would create a dangling
                               pointer. */
    PyModule_AddObject(m, "ParstackError", ParstackError);
}

