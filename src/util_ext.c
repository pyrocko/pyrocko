#define _XOPEN_SOURCE 600
#define _BSD_SOURCE
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <errno.h>
#include <assert.h>
#include <math.h>

#include "Python.h"

static PyObject *UtilExtError;

time_t my_timegm(struct tm *tm) {
    time_t ret;
    char *tz;

    tz = getenv("TZ");
    setenv("TZ", "", 1);
    tzset();
    ret = mktime(tm);
    if (tz) {
        setenv("TZ", tz, 1);
    } else {
        unsetenv("TZ");
    }
    tzset();
    return ret;
}

int endswith(const char *s, const char *suffix) {

    size_t n1 = strlen(s);
    size_t n2 = strlen(suffix);

    if (n2 > n1) {
        return 0;
    }
    if (0 == strcmp(s+(n1-n2), suffix)) {
        return 1;
    }
    return 0;
}

int stt(const char *s, const char *format, time_t *t, double *tfrac) {


    struct tm tm;
    size_t nf = strlen(format);
    size_t ns = strlen(s);
    char format2[nf+1];
    char s2[ns+1];
    char *sfrac, *end;
    int nexpect;

    *t = 0;
    *tfrac = 0.0;

    strcpy(format2, format);
    strcpy(s2, s);

    nexpect = -2;
    if (endswith(format2, ".FRAC")) {
        nexpect = -1;
    } else if (endswith(format2, ".1FRAC")) {
        nexpect = 1;
    } else if (endswith(format2, ".2FRAC")) {
        nexpect = 2;
    } else if (endswith(format2, ".3FRAC")) {
        nexpect = 3;
    } else if (endswith(format2, ".OPTFRAC")) {
        nexpect = 0;
    }

    if (nexpect != -2) {
        *rindex(format2, '.') = '\0'; /* cannot fail here */
        sfrac = rindex(s2, '.');
        if (nexpect != 0 && sfrac == NULL) {
            return 1;  /* fractional seconds expected but not found */
        }
        if (sfrac != NULL) {
            if (nexpect > 0 && strlen(sfrac) != (size_t)(nexpect + 1)) {
                return 2;  /* incorrect number of digits in fractional seconds part */
            }
            errno = 0;
            *tfrac = strtod(sfrac, NULL);
            if (errno != 0) {
                return 4;  /* could not convert fractional part to a number */
            }
            *sfrac = '\0';
        }
    }

    end = strptime(s2, format2, &tm);

    if (end == NULL || *end != '\0') {
        return 5;  /* could not parse date/time */
    }

    *t = timegm(&tm);
    if (*t == -1) {
        return 2;  /* mktime failed */
    }

    return 0;
}

int tts(time_t t, double tfrac, const char *format, char **sout) {
    size_t nf = strlen(format);
    char format2[nf+1];
    char sfrac[20] = "";
    char ffmt[5];
    char buf[200];
    struct tm tm;
    size_t n;

    strcpy(ffmt, "%.0f");
    strcpy(format2, format);
    if (nf >= 6 && (format2[nf-6] == '.') && endswith(format2, "FRAC") &&
        '1' <= format2[nf-5] && format2[nf-5] <= '9') {
        ffmt[2] = format2[nf-5];
        format2[nf-6] = '\0';

        snprintf(sfrac, 20, ffmt, tfrac);
        if (sfrac[0] == '1') {
            t += 1;
        }
    }

    if (NULL == gmtime_r(&t, &tm)) {
        return 1;
    }

    n = strftime(buf, 200, format2, &tm);
    if (n == 0) {
        return 2;
    }

    *sout = (char*)malloc(n + strlen(sfrac) - 1 + 1);
    if (*sout == NULL) {
        return 3;
    }
    **sout = '\0';
    strncat(*sout, buf, n);
    strcat(*sout, sfrac+1);
    return 0;
}

static PyObject* w_stt(PyObject *dummy, PyObject *args) {

    char *s;
    char *format;
    time_t t;
    double tfrac;

    (void)dummy; /* silence warning */

    if (!PyArg_ParseTuple(args, "ss", &s, &format)) {
        PyErr_SetString(UtilExtError, "usage stt(s, format)" );
        return NULL;
    }
    if (0 != stt(s, format, &t, &tfrac)) {
        return NULL;
    }
    return Py_BuildValue("Ld", (long long int)t, tfrac);
}

static PyObject* w_tts(PyObject *dummy, PyObject *args) {

    char *s;
    char *format;
    time_t t;
    double tfrac;
    PyObject *val;

    (void)dummy; /* silence warning */

    if (!PyArg_ParseTuple(args, "Lds", &t, &tfrac, &format)) {
        PyErr_SetString(UtilExtError, "usage tts(t, tfrac, format)" );
        return NULL;
    }

    if (0 != tts(t, tfrac, format, &s)) {
        return NULL;
    }

    val = Py_BuildValue("s", s);
    free(s);
    return val;
}


static PyMethodDef UtilExtMethods[] = {
    {"tts",  w_tts, METH_VARARGS,
        "time to string" },

    {"stt", w_stt, METH_VARARGS,
        "string to time" },

    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initutil_ext(void)
{
    PyObject *m;

    m = Py_InitModule("util_ext", UtilExtMethods);
    if (m == NULL) return;

    UtilExtError = PyErr_NewException("util_ext.error", NULL, NULL);
    Py_INCREF(UtilExtError);  /* required, because other code could remove `error`
                               from the module, what would create a dangling
                               pointer. */
    PyModule_AddObject(m, "UtilExtError", UtilExtError);
}

