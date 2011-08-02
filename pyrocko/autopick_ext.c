#define NPY_NO_DEPRECATED_API

#include "Python.h"
#include "numpy/arrayobject.h"

static PyObject *AutoPickError;
#include<math.h>

#ifndef max
	#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif


int autopick_recursive_stalta( int ns, int nl, float ks, float kl, int nsamples, float *inout, float *intermediates, int init)
{
    int i, istart;
    float eps = 1.0e-7;
    float k, scf0, lcf0, sta0, lta0, nshort, nlong, maxlta;
    float cf[nsamples], sta[nsamples], lta[nsamples]; /* da sistemare */
   
    cf[0] = inout[0];
    k = 3.;
    if (init == 0) {
        cf[0] = inout[0] + abs(k*(inout[0]-intermediates[ns-1]));
    }
    for (i=1;i<nsamples;i++)
    {
        cf[i]=inout[i]+abs(k*(inout[i]-inout[i-1]));
    }
    
    maxlta = 0.;
    if (init == 1)
    {
        if (nsamples <= ns + nl) {
            return 1;
        }
        
        sta0 = 0.;
        scf0 = 0.;

        for (i=nl; i<nl+ns; i++)
        {
             sta0=cf[i]+scf0;
             scf0=sta0;    
        }
        
        lta0 = 0.;
        lcf0 = 0.;
        
        for (i=0; i<nl; i++)
        {
             lta0 = cf[i]+lcf0;
             lcf0 = lta0;    
        }
     
        nshort = (float)ns;
        nlong =(float)nl;
        sta[nl+ns] = (sta0/nshort);
        lta[nl+ns] = (lta0/nlong);
        for (i=0; i<nl+ns; i++) {
            sta[i] = lta[i] = 0.0;
        }
        istart = nl+ns;

    } else {
        
        if (nsamples <= ns) {
            return 1;
        }

        sta[0] = intermediates[ns];
        lta[0] = intermediates[ns+1];
        istart = ns;
        
        for(i=1;i<ns;i++)
        {
            sta[i] = (ks*cf[i]+(1.-ks)*sta[i-1]);
            lta[i] = (kl*(intermediates[i-1]) + (1.-kl)*lta[i-1]);
            maxlta = max(fabs(lta[i]), maxlta);
        }
    }
    
    for(i=istart;i<nsamples;i++)
    {
        sta[i]=(ks*cf[i]+(1.-ks)*sta[i-1]);
        lta[i]=(kl*cf[i-ns]+(1.-kl)*lta[i-1]);
        maxlta = max(fabs(lta[i]), maxlta);
    }
    
    if (maxlta == 0.0) {
        maxlta = eps*eps;
    }

    for(i=0;i<nsamples;i++)
    {
        inout[i] = sta[i]/(lta[i]+eps*maxlta);
    }
            
    for (i=0;i<ns;i++)
    {
        intermediates[i] = cf[nsamples-ns+i];
    }
    
    
    intermediates[ns] = sta[nsamples-1];
    intermediates[ns+1] = lta[nsamples-1];
    
    return 0;
}

static PyObject* autopick_recursive_stalta_wrapper(PyObject *dummy, PyObject *args) {
    PyObject *inout_array_obj, *temp_array_obj;
    PyArrayObject *inout_array = NULL;
    PyArrayObject *temp_array = NULL;
    int ns, nl, initialize, nsamples, ntemp; 
    double ks, kl; 

    if (!PyArg_ParseTuple(args, "iiddOOi", &ns, &nl, &ks, &kl, &inout_array_obj, &temp_array_obj, &initialize)) {
        PyErr_SetString(AutoPickError, "invalid arguments in recursive_stalta(ns, nl, ks, kl, inout_data, temp_data, initialize)" );
        return NULL;
    }
    inout_array = (PyArrayObject*)PyArray_ContiguousFromAny(inout_array_obj, NPY_FLOAT32, 1, 1);
    if (inout_array == NULL) {
        PyErr_SetString(AutoPickError, "cannot create a contiguous float array from inout_data." );
        return NULL;
    }

    temp_array = (PyArrayObject*)PyArray_ContiguousFromAny(temp_array_obj, NPY_FLOAT32, 1, 1);
    if (temp_array == NULL) {
        PyErr_SetString(AutoPickError, "cannot create a contiguous float array from temp_data." );
        return NULL;
    }
    nsamples = PyArray_SIZE(inout_array);
    ntemp = PyArray_SIZE(temp_array);

    if (ntemp != ns+2) {
        PyErr_SetString(AutoPickError, "temp_data must have length of ns+2.");
        return NULL;
    }

    if (0 != autopick_recursive_stalta(ns, nl, ks, kl, nsamples, (float*)PyArray_DATA(inout_array), (float*)PyArray_DATA(temp_array), initialize)) {
        PyErr_SetString(AutoPickError, "running STA/LTA failed.");
        return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
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

