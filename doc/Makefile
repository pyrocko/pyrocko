# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python3 -m sphinx
SPHINXPROJ    = Pyrocko
SOURCEDIR     = source
BUILDDIR      = build

# Generated sources: command references and autosummary stubs.
GENERATED     = $(SOURCEDIR)/apps/squirrel/reference \
                $(SOURCEDIR)/apps/pyrocko/reference \
                $(SOURCEDIR)/apps/fomosto/reference \
                $(SOURCEDIR)/apps/gato/reference \
                $(SOURCEDIR)/library/reference

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html: Makefile squirrel_cli pyrocko_cli fomosto_cli gato_cli
	@$(SPHINXBUILD) -n "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

squirrel_cli:
	python3 ../maintenance/argparse_help_to_rst.py \
		pyrocko.squirrel.main \
		source/apps/squirrel/reference/

pyrocko_cli:
	python3 ../maintenance/argparse_help_to_rst.py \
		pyrocko.apps.pyrocko.main \
		source/apps/pyrocko/reference/

fomosto_cli:
	python3 ../maintenance/argparse_help_to_rst.py \
		pyrocko.apps.fomosto.main \
		source/apps/fomosto/reference/

gato_cli:
	python3 ../maintenance/argparse_help_to_rst.py \
		pyrocko.gato.main \
		source/apps/gato/reference/

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	rm -rf $(GENERATED)

.PHONY: help Makefile squirrel_cli pyrocko_cli fomosto_cli gato_cli clean

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile squirrel_cli pyrocko_cli fomosto_cli
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
