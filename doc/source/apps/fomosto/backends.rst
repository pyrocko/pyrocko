Fomosto backends
================

The Fomosto tool relies on external programs to do the actual numerical work of
Green's function computation. We refer to these programs (together with the
code to communicate between Fomosto and the external programs) as *backends*.
Several such backends exist and this document should give some
hints on which one to use under what circumstances.

.. highlight:: console

The ``ahfullgreen`` backend
---------------------------

This backend can be used to evaluate the elastodynamic response of a
homogeneous fullspace. Analytical solutions, given e.g. in [AkiRichards2002]_,
are evaluated in the frequency domain and transformed into time domain via FFT.
It is possible to use single force and moment tensor excitations. The solutions
include near, intermediate, and far field contributions, including also the
static end value in the near field.

With its limitation to a homogeneous full space this backend is probably only
useful for some special case applications, e.g. in mining seismology, to gain
theoretical understanding, and for testing purposes.

This backend is included in Pyrocko and no external programs are needed.

To initialize a Green's function store for ``ahfullgreen``, run::

    $ fomosto init ahfullgreen my_ahfullgreen_gfs

The ``qseis`` backend
---------------------

QSEIS is a code to calculate synthetic seismograms based on a layered
viscoelastic half-space model. It has been written by Rongjiang Wang
[Wang1999]_. It uses the orthonormal propagator algorithm, a numerically more
stable alternative to the reflectivity method. QSEIS uses many state-of-the-art
techniques to suppress time-domain aliasing problems and numerical phases.
Synthetic seismograms at up to teleseismic distances can be computed by using
the earth flattening transformation (with some restrictions). Different shallow
structures can be defined for source and receiver site (body-wave phases only).

Use this backend for local and regional setups, or when different source and
receiver structures should be considered. It can also be used to compute single
force excitation Green's functions.

The current version of QSEIS is ``2006a`` (at the time of writing, 2017-01-25),
and can be downloaded from https://git.pyrocko.org/pyrocko/fomosto-qseis .

After downloading and installing, to initialize a Green's function store
to be built with QSEIS, run::

    $ fomosto init qseis.2006a my_qseis_gfs

The ``qssp`` backend
--------------------

QSSP is a code to calculate complete synthetic seismograms of a layered,
self-gravitating spherical Earth using the normal mode theory. It has been
written by Rongjiang Wang [Wang2017]_. It uses a hybrid algorithm with
numerical integration at low frequencies and orthonormal propagator algorithm
at high frequencies and uses state-of-the-art techniques to supress time- and
space-domain aliasing problems and numerical phases.

This backend is the choice when global seismograms at very low (first
eigenmodes) or very high frequencies (4 Hz has been tested) are desired, when
core phases are involved or when the coupling of earth and atmosphere is of
interest.

The current version of QSSP is ``2017`` (at the time of writing, 2020-09-04),
and can be downloaded from https://git.pyrocko.org/pyrocko/fomosto-qssp2017 .

After downloading and installing, to initialize a Green's function store
to be built with QSSP, run::

    $ fomosto init qssp.2017 my_qssp_gfs

QSSP can also be used to calculate rotational seismograms.  To create a Pyrocko
GF store with rotational Green's functions, just set ``stored_quantity:
'rotation'`` in the store's ``config``.

The ``psgrn_pscmp`` backend
---------------------------

Code to calculate synthetic stress/strain/tilt/gravitational fields on a
layered viscoelastic halfspace. It has been written by Rongjiang Wang
[Wang2003]_, [Wang2005]_, [Wang2006]_. It uses the orthonormal propagator
method for numerical stability.

Use this code for calculating synthetic static and/or viscoelastic displacement
fields of tectonic events, magmatic intrusions or fluid migrations. Especially
useful, if synthetics have to be calculated for many points like for InSAR or
GPS.

The stresses, tilts and geoidal changes can still be computed but are (not yet)
supported by a Fomosto Green's function store and the respective stacking
functions.

The current version of PSGRN/PSCMP is ``2008a`` (at the time of writing,
2017-02-14), and can be downloaded from
https://git.pyrocko.org/pyrocko/fomosto-psgrn-pscmp .

After downloading and installing, to initialize a Green's function store to be
built with PSGRN/PSCMP, run::

    $ fomosto init psgrn_pscmp.2008a my_psgrn_pscmp_gfs

References
----------

.. [AkiRichards2002] Aki, Keiiti, and Paul G. Richards. Quantitative
    seismology. Vol. 1. 2002.

.. [Wang1999] Wang, R. (1999): A simple orthonormalization method for the stable and efficient computation of Green's functions. - Bulletin of the Seismological Society of America, 89, 733-741.

.. [Wang2003] Wang, R., Lorenzo Martín, F., Roth, F. (2003): Computation of deformation induced by earthquakes in a multi-layered elastic crust; FORTRAN programs EDGRN/ EDCMP. - Computers and Geosciences, 29, 2, 195-207. https://doi.org/10.1016/S0098-3004(02)00111-5

.. [Wang2005] Wang, R. (2005): The dislocation theory: a consistent way for including the gravity effect in (visco)elastic plane-earth models. - Geophysical Journal International, 161, 1, 191-196. https://doi.org/10.1111/j.1365-246X.2005.02614.x

.. [Wang2006] Wang, R., Lorenzo Martín, F., Roth, F. (2006): PSGRN/PSCMP - a new code for calculating co- and post-seismic deformation, geoid and gravity changes based on the viscoelastic-gravitational dislocation theory. - Computers and Geosciences, 32, 4, 527-541. https://doi.org/10.1016/j.cageo.2005.08.006

.. [Wang2017] Wang, R., Heimann, S., Zhang, Y., Wang, H., Dahm, T. (2017): Complete synthetic seismograms based on a spherical self-gravitating Earth model with an atmosphere–ocean–mantle–core structure. - Geophysical Journal International, 210, 3, 1739-1764. https://doi.org/10.1093/gji/ggx259
