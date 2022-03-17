#define NPY_NO_DEPRECATED_API 7

#define SQR(a)  ( (a) * (a) )

#include "Python.h"
#include "numpy/arrayobject.h"

#if defined(_OPENMP)
    #include <omp.h>
#endif

#include <limits.h>
#include <float.h>

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state); (void) m;
static struct module_state _state;
#endif


int good_array(PyObject* o, int typenum) {
    if (!PyArray_Check(o)) {
        PyErr_SetString(PyExc_ValueError, "not a NumPy array" );
        return 0;
    }

    if (PyArray_TYPE((PyArrayObject*)o) != typenum) {
        PyErr_SetString(PyExc_ValueError, "array of unexpected type");
        return 0;
    }

    if (!PyArray_ISCARRAY((PyArrayObject*)o)) {
        PyErr_SetString(PyExc_ValueError, "array is not contiguous or not well behaved");
        return 0;
    }

    return 1;
}


static PyObject* w_spit_lookup(
        PyObject *m,
        PyObject *args,
        PyObject *kwds) {

    PyObject *coords;
    PyObject *req_coords;
    (void) m;

    int nthreads = 4;

    static char *kwlist[] = {"coords", "req_coords", "threads", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|i", kwlist, &coords, &req_coords, &nthreads)) {
        return NULL;
    }

    if (!good_array(coords, NPY_FLOAT32) || !good_array(req_coords, NPY_FLOAT32)) {
        return NULL;
    }

    PyArrayObject* arr_coords = (PyArrayObject*) coords;
    PyArrayObject* arr_req_coords = (PyArrayObject*) req_coords;

    npy_float32 *data_coords = PyArray_DATA(arr_coords);
    npy_float32 *data_req_coords = PyArray_DATA(arr_req_coords);

    npy_intp ncoords = PyArray_SIZE(arr_coords);
    npy_intp nreq = PyArray_SIZE(arr_req_coords);

    PyArrayObject* arr_result_idx = PyArray_ZEROS(1, &nreq, NPY_INTP, 0);

    npy_intp *idx_close = PyArray_DATA(arr_result_idx);

    float dist, dist_prop;
    npy_intp icoord, ireq;

    Py_BEGIN_ALLOW_THREADS

    // npy_intp idx[nreq];
    // float min_val;
    // float temp[ncoords];
    // #pragma omp parallel for schedule(static) nowait private(min_val, temp) shared(idx_close) num_threads(nthreads)
    // for(ireq = 0; ireq < nreq; ireq++) {

    //     // Interchange loops, bigger loop must be inside for better pipelining.
    //     // But care if idx and temp are fitting into chache if not change loops.
    //     #pragma omp simd
    //     for (icoord = 0; icoord < ncoords; icoord++) {
    //         temp[icoord] = fabs(data_coords[icoord] - data_req_coords[ireq]);
    //     }
    //     min_val = temp[0];

    //     // Avoid branching insinde simd loops. Think about ways to get rid of it.
    //     // If we are lucky temp_idx is stored fix in a multipurpose register and no load-store has to be done inside if clause.
    //     for (npy_intp icoord = 1; icoord < ncoords; icoord++) {
    //         if (min_val < temp[icoord]){
    //             min_val = temp[icoord];
    //             idx_close[ireq] = icoord;
    //         }
    //     }
    // }

    #if defined(_OPENMP)
        #pragma omp parallel private(dist, dist_prop, icoord) num_threads(nthreads)
    #endif
    {
    #if defined(_OPENMP)
        #pragma omp for schedule(static) nowait
    #endif
    for (ireq = 0; ireq < nreq; ireq++) {
        dist = fabs(data_coords[0] - data_req_coords[ireq]);

        #pragma omp simd
        for (icoord = 1; icoord < ncoords; icoord++) {
            dist_prop = fabs(data_coords[icoord] - data_req_coords[ireq]);
            if (dist_prop < dist) {
                dist = dist_prop;
                idx_close[ireq] = icoord;
            } else {
                // The coord vector is monotonic.
                // we can break the loop once the distance starts to increase again.
                icoord = ncoords;
            }
        }
    }
    }

    Py_END_ALLOW_THREADS

    return (PyObject*) arr_result_idx;
}


static PyMethodDef spit_ext_methods[] = {
    {
        "spit_lookup",
        (PyCFunction) w_spit_lookup,
        METH_VARARGS | METH_KEYWORDS,
        "Retrieve lookup indices."
    },
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3

static int spit_ext_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int spit_ext_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "spit_ext",
        NULL,
        sizeof(struct module_state),
        spit_ext_methods,
        NULL,
        spit_ext_traverse,
        spit_ext_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_spit_ext(void)

#else
#define INITERROR return

void
initspit_ext(void)
#endif

{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("spit_ext", spit_ext_methods);
#endif
    import_array();

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException(
        "pyrocko.spit_ext.SpitExtError", NULL, NULL);

    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

    Py_INCREF(st->error);
    PyModule_AddObject(module, "SpitExtError", st->error);

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
