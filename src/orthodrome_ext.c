#define NPY_NO_DEPRECATED_API 7
#define D2R (M_PI / 180.)
#define R2D (1.0 / D2R)

#define EARTH_OBLATENESS 1./298.257223563
#define EARTHRADIUS_EQ 6378140.0
#define EARTHRADIUS 6371000.0

#include "Python.h"
#include "numpy/arrayobject.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#define max(a, b) \
   ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a, b) \
   ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })  

static PyObject *OrthodromeExtError;

typedef npy_float32 float32_t;
typedef npy_float64 float64_t;

int good_array(PyObject* o, int typenum, ssize_t size_want, int ndim_want, npy_intp* shape_want) {
    int i;

    if (!PyArray_Check(o)) {
        PyErr_SetString(OrthodromeExtError, "not a NumPy array" );
        return 0;
    }

    if (PyArray_TYPE((PyArrayObject*)o) != typenum) {
        PyErr_SetString(OrthodromeExtError, "array of unexpected type");
        return 0;
    }

    if (size_want != -1 && size_want != PyArray_SIZE((PyArrayObject*)o)) {
        PyErr_SetString(OrthodromeExtError, "array is of unexpected size");
        return 0;
    }
    if (ndim_want != -1 && ndim_want != PyArray_NDIM((PyArrayObject*)o)) {
        PyErr_SetString(OrthodromeExtError, "array is of unexpected ndim");
        return 0;
    }

    if (ndim_want != -1) {
        for (i=0; i<ndim_want; i++) {
            if (shape_want[i] != -1 && shape_want[i] != PyArray_DIMS((PyArrayObject*)o)[i]) {
                PyErr_SetString(OrthodromeExtError, "array is of unexpected shape");
                return 0;
            }
        }
    }
    return 1;
}

static float64_t sqr(float64_t x) {
    return x*x;
}

static float64_t clip(float64_t x, float64_t mi, float64_t ma) {
    return x < mi ? mi : (x > ma ? ma : x);
}

static float64_t wrap(float64_t x, float64_t mi, float64_t ma) {
    return x - floor((x-mi)/(ma-mi)) * (ma-mi);
}


static float64_t cosdelta(float64_t alat, float64_t alon, float64_t blat, float64_t blon) {
    return min(1., sin(alat*D2R) * sin(blat*D2R) + 
            cos(alat*D2R) * cos(blat*D2R) * cos(D2R* (blon - alon)));
}

static void azibazi(float64_t alat, float64_t alon, float64_t blat, float64_t blon, float64_t *azi, float64_t *bazi) {
    float64_t cd;
    if (alat == blat && alon == blon){
        *azi = 0.;
        *bazi = 180.;
    } else {

        cd = cosdelta(alat, alon, blat, blon);
        *azi = R2D * atan2(
            cos(D2R * alat) * cos(D2R * blat) * sin(D2R * (blon-alon)),
            sin(D2R * blat) - sin(D2R * alat) * cd);
        *bazi = R2D * atan2(
            cos(D2R * blat) * cos(D2R * alat) * sin(D2R * (alon-blon)),
            sin(D2R * alat) - sin(D2R * blat) * cd);
    }
}

static void azibazi_array(float64_t *alats, float64_t *alons, float64_t *blats, float64_t *blons, npy_intp size, float64_t *azis, float64_t *bazis) {
    npy_intp i;
    for (i = 0; i < size; i++) {
        azibazi(alats[i], alons[i], blats[i], blons[i], &azis[i], &bazis[i]);
    }
}

static void ne_to_latlon(float64_t lat, float64_t lon, float64_t north, float64_t east, float64_t *lat_new, float64_t *lon_new) {
    float64_t a, b, c, gamma, alphasign, alpha;

    a = sqrt(sqr(north) + sqr(east)) / EARTHRADIUS;
    gamma = atan2(east, north);

    b = 0.5*M_PI - lat*D2R;

    alphasign = gamma < 0.0 ? -1.0 : 1.0;
    gamma = fabs(gamma);

    c = acos(clip(cos(a)*cos(b)+sin(a)*sin(b)*cos(gamma), -1., 1.));

    alpha = asin(clip(sin(a)*sin(gamma)/sin(c), -1., 1.));
    alpha = cos(a) - cos(b)*cos(c) < 0.0 ? (alpha > 0.0 ? M_PI-alpha : -M_PI-alpha) : alpha;


    *lat_new = R2D * (0.5*M_PI - c);
    *lon_new = wrap(lon + R2D*alpha*alphasign, -180., 180.);
}

static void azibazi4(float64_t *a, float64_t *b, float64_t *azi, float64_t *bazi) {
    /* azimuth and backazimuth for (lat,lon,north,east) coordinates */

    float64_t alat, alon, anorth, aeast;
    float64_t blat, blon, bnorth, beast;
    float64_t alat_eff, alon_eff;
    float64_t blat_eff, blon_eff;

    alat = a[0];
    alon = a[1];
    anorth = a[2];
    aeast = a[3];
    blat = b[0];
    blon = b[1];
    bnorth = b[2];
    beast = b[3];

    if (alat == blat && alon == blon) { /* carthesian */
        *azi = atan2(beast - aeast, bnorth - anorth) * R2D;
        *bazi = *azi + 180.0;
    } else { /* spherical */
        ne_to_latlon(alat, alon, anorth, aeast, &alat_eff, &alon_eff);
        ne_to_latlon(blat, blon, bnorth, beast, &blat_eff, &blon_eff);
        azibazi(alat_eff, alon_eff, blat_eff, blon_eff, azi, bazi);
    }
}

int check_latlon_ranges(float64_t alat, float64_t alon, float64_t blat, float64_t blon){
    if (alat > 90. || alat < -90. || blat > 90. || blat < -90.) {
        PyErr_SetString(PyExc_ValueError, "distance_accurate50m: Latitude must be between -90 and 90 degree.");
        return 0;
    }
    if (alon > 180. || alon < -180. || blon > 180. || blon < -180.) {
        PyErr_SetString(PyExc_ValueError, "distance_accurate50m: Longitude must be between -180 and 180 degree.");
        return 0;
    }

    return 1;
}

static void distance_accurate50m(float64_t alat, float64_t alon, float64_t blat, float64_t blon, float64_t *dist) {
    /* more accurate distance calculation based on a spheroid of rotation

    returns distance in [m] between points a and b
    coordinates must be given in degrees

    should be accurate to 50 m using WGS84

    from wikipedia :  http://de.wikipedia.org/wiki/Orthodrome
    based on: Meeus, J.: Astronomical Algorithms, S 85, Willmann-Bell,
              Richmond 2000 (2nd ed., 2nd printing), ISBN 0-943396-61-1 */
    float64_t f, g, l, s, c, w, r;
    // float64_t d, h1, h2;
    // check_latlon_ranges(alat, alon, blat, blon);
    f = (alat + blat) * D2R / 2.;
    g = (alat - blat) * D2R / 2.;
    l = (alon - blon) * D2R / 2.;

    s = sqr(sin(g)) * sqr(cos(l)) + sqr(cos(f)) * sqr(sin(l));
    c = sqr(cos(g)) * sqr(cos(l)) + sqr(sin(f)) * sqr(sin(l));

    w = atan(sqrt(s/c));

    if (w == 0.) {
        *dist = 0.;
        return;
    }

    r = sqrt(s*c)/w;
    // d = 2. * w * EARTHRADIUS_EQ;
    // h1 = (3.*r-1.) / (2.*c);
    // h2 = (3.*r-1.) / (2.*s);

    *dist = 2. * w * EARTHRADIUS_EQ *
        (1. +
         EARTH_OBLATENESS * ((3.*r-1.) / (2.*c)) * sqr(sin(f)) * sqr(cos(g)) -
         EARTH_OBLATENESS * ((3.*r+1.) / (2.*s)) * sqr(cos(f)) * sqr(sin(g)));
}

static void distance_accurate50m_array(float64_t *alats, float64_t *alons, float64_t *blats, float64_t *blons, npy_intp npairs, float64_t *dists) {
    npy_intp i;
    for (i = 0; i < npairs; i++) {
        distance_accurate50m(alats[i], alons[i], blats[i], blons[i], &dists[i]);
    }
}

/*
Wrapper shizzle
*/

static PyObject* w_azibazi(PyObject *dummy, PyObject *args){
    float64_t alat, alon, blat, blon;
    float64_t azi, bazi;
    
    (void) dummy;

    if (! PyArg_ParseTuple(args, "dddd", &alat, &alon, &blat, &blon)) {
        PyErr_SetString(OrthodromeExtError, "azibazi: invalid call!");
        return NULL;
    }

    if (! check_latlon_ranges(alat, alon, blat, blon)) {
        return NULL;
    };

    azibazi(alat, alon, blat, blon, &azi, &bazi);
    return Py_BuildValue("dd", azi, bazi);
}

static PyObject* w_distance_accurate50m(PyObject *dummy, PyObject *args) {
    float64_t alat, alon, blat, blon, dist;

    (void)dummy;
    if (! PyArg_ParseTuple(args, "dddd", &alat, &alon, &blat, &blon)) {
        PyErr_SetString(OrthodromeExtError, "distance_accurate50m: invalid call!");
        return NULL;
    }
    if (! check_latlon_ranges(alat, alon, blat, blon)) {
        return NULL;
    };

    distance_accurate50m(alat, alon, blat, blon, &dist);
    return Py_BuildValue("d", dist);
}

static PyObject * w_distance_accurate50m_numpy(PyObject *dummy, PyObject *args) {
    PyObject *alats_arr, *alons_arr, *blats_arr, *blons_arr;
    PyArrayObject *c_alats_arr, *c_alons_arr, *c_blats_arr, *c_blons_arr, *dists_arr;
    float64_t *alats, *alons, *blats, *blons;
    npy_intp size[1];

    (void) dummy;
    if (! PyArg_ParseTuple(args, "OOOO", &alats_arr, &alons_arr, &blats_arr, &blons_arr)) {
        PyErr_SetString(OrthodromeExtError, "distance_accurate50m_numpy: invalid call!");
        return NULL;
    }

    if (! (good_array(alats_arr, NPY_FLOAT64, -1, -1, (npy_intp *) -1) || good_array(alons_arr, NPY_FLOAT64, -1, -1, (npy_intp *) -1) ||
           good_array(blats_arr, NPY_FLOAT64, -1, -1, (npy_intp *) -1) || good_array(alons_arr, NPY_FLOAT64, -1, -1, (npy_intp *) -1))) {
        return NULL;
    }

    c_alats_arr = PyArray_GETCONTIGUOUS((PyArrayObject*) alats_arr);
    c_alons_arr = PyArray_GETCONTIGUOUS((PyArrayObject*) alons_arr);
    c_blats_arr = PyArray_GETCONTIGUOUS((PyArrayObject*) blats_arr);
    c_blons_arr = PyArray_GETCONTIGUOUS((PyArrayObject*) blons_arr);
    
    if (PyArray_SIZE(c_alats_arr) + PyArray_SIZE(c_alons_arr) - PyArray_SIZE(c_blats_arr) - PyArray_SIZE(c_blons_arr)) { 
        PyErr_SetString(PyExc_ValueError, "distance_accurate50m_numpy: dimension mismatch!");
        return NULL;   
    }

    alats = PyArray_DATA(c_alats_arr);
    alons = PyArray_DATA(c_alons_arr);
    blats = PyArray_DATA(c_blats_arr);
    blons = PyArray_DATA(c_blons_arr);

    size[0] = PyArray_SIZE((PyArrayObject*) alats_arr);
    dists_arr = (PyArrayObject*) PyArray_EMPTY(1, size, NPY_FLOAT64, 0);

    distance_accurate50m_array(alats, alons, blats, blons, size[0], PyArray_DATA(dists_arr));

    return (PyObject *) dists_arr;
}

static PyObject * w_azibazi_numpy(PyObject *dummy, PyObject *args) {
    PyObject *alats_arr, *alons_arr, *blats_arr, *blons_arr;
    PyArrayObject *c_alats_arr, *c_alons_arr, *c_blats_arr, *c_blons_arr, *azis_arr, *bazis_arr;
    float64_t *alats, *alons, *blats, *blons;
    npy_intp size[1];

    (void) dummy;
    if (! PyArg_ParseTuple(args, "OOOO", &alats_arr, &alons_arr, &blats_arr, &blons_arr)) {
        PyErr_SetString(OrthodromeExtError, "distance_accurate50m_numpy: invalid call!");
        return NULL;
    }

    if (! (good_array(alats_arr, NPY_FLOAT64, -1, -1, (npy_intp *) -1) || good_array(alons_arr, NPY_FLOAT64, -1, -1, (npy_intp *) -1) ||
           good_array(blats_arr, NPY_FLOAT64, -1, -1, (npy_intp *) -1) || good_array(alons_arr, NPY_FLOAT64, -1, -1, (npy_intp *) -1))) {
        return NULL;
    }

    c_alats_arr = PyArray_GETCONTIGUOUS((PyArrayObject*) alats_arr);
    c_alons_arr = PyArray_GETCONTIGUOUS((PyArrayObject*) alons_arr);
    c_blats_arr = PyArray_GETCONTIGUOUS((PyArrayObject*) blats_arr);
    c_blons_arr = PyArray_GETCONTIGUOUS((PyArrayObject*) blons_arr);
    
    if (PyArray_SIZE(c_alats_arr) + PyArray_SIZE(c_alons_arr) - PyArray_SIZE(c_blats_arr) - PyArray_SIZE(c_blons_arr)) { 
        PyErr_SetString(PyExc_ValueError, "distance_accurate50m_numpy: dimension mismatch!");
        return NULL;   
    }

    alats = PyArray_DATA(c_alats_arr);
    alons = PyArray_DATA(c_alons_arr);
    blats = PyArray_DATA(c_blats_arr);
    blons = PyArray_DATA(c_blons_arr);

    size[0] = PyArray_SIZE((PyArrayObject*) alats_arr);
    azis_arr = (PyArrayObject*) PyArray_EMPTY(1, size, NPY_FLOAT64, 0);
    bazis_arr = (PyArrayObject*) PyArray_EMPTY(1, size, NPY_FLOAT64, 0);

    azibazi_array(alats, alons, blats, blons, size[0], PyArray_DATA(azis_arr), PyArray_DATA(azis_arr));

    return Py_BuildValue("NN", (PyObject *) azis_arr, (PyObject *) bazis_arr);
}

static PyMethodDef OrthodromeExtMethods[] = {
    {"distance_accurate50m",  w_distance_accurate50m, METH_VARARGS,
"Calculate distance between a pair of (alat, alon, blat, blon) points on elliptic earth.\n\n\
:param alat: Latitude of point 1.\n\
:type alat: float\n\
:param alon: Longitude of point 1.\n\
:type alon: float\n\
:param alat: Latitude of point 2.\n\
:type alat: float\n\
:param alon: Longitude of point 2.\n\
:type alon: float"
},

    {"distance_accurate50m_numpy",  w_distance_accurate50m_numpy, METH_VARARGS,
"Calculate distance between a arrays of (alat, alon, blat, blon) numpy arrays on elliptic earth.\n\n\
:param alat: Latitudes\n\
:type alat: :py:class:numpy.array\n\
:param alon: Longitudes\n\
:type alon: :py:class:numpy.array\n\
:param alat: Latitudes\n\
:type alat: :py:class:numpy.array\n\
:param alon: Longitudes\n\
:type alon: :py:class:numpy.array" },

    {"azibazi",  w_azibazi, METH_VARARGS,
"Calculate azimuth and backazimuth for tuple (alat, alon, blat, blon).\n\n\
:param alat: Latitude of point 1.\n\
:type alat: float\n\
:param alon: Longitude of point 1.\n\
:type alon: float\n\
:param alat: Latitude of point 2.\n\
:type alat: float\n\
:param alon: Longitude of point 2.\n\
:type alon: float" 
},

    {"azibazi_numpy",  w_azibazi_numpy, METH_VARARGS,
"Calculate azimuth and backazimuth for tuple (alats, alons, blats, blons) of numpy arrays.\n\n\
:param alat: Latitudes\n\
:type alat: :py:class:numpy.array\n\
:param alon: Longitudes\n\
:type alon: :py:class:numpy.array\n\
:param alat: Latitudes\n\
:type alat: :py:class:numpy.array\n\
:param alon: Longitudes\n\
:type alon: :py:class:numpy.array" },

    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initorthodrome_ext(void)
{
    PyObject *m;

    m = Py_InitModule("orthodrome_ext", OrthodromeExtMethods);
    if (m == NULL) return;
    import_array();

    OrthodromeExtError = PyErr_NewException("orthodrome_ext.error", NULL, NULL);
    Py_INCREF(OrthodromeExtError);  /* required, because other code could remove `error`
                               from the module, what would create a dangling
                               pointer. */
    PyModule_AddObject(m, "OrthodromeExtError", OrthodromeExtError);
}
