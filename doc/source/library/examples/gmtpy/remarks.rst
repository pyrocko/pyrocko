Random remarks
==============

* Some GMT tools like e.g. :program:`img2grd` cannot handle the ``+gmtdefaults`` option which is by default always set when running GMT programs from GmtPy.
  Use ``suppress_defaults=True`` in these cases.

* To convert PDF to raster images, :program:`libpoppler` based utilities, such as :program:`pdftoppm` currently do a better job than those based on :program:`gs`.

