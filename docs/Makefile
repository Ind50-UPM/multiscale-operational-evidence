# Minimal Sphinx makefile with GitHub Pages publication helpers.

SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
HTMLDIR       = $(BUILDDIR)/html

.PHONY: help html clean github-pages

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)"

clean:
	rm -rf "$(BUILDDIR)"
	find . -maxdepth 1 \( -name '*.html' -o -name '.buildinfo' -o -name 'genindex*' -o -name 'search*' -o -name 'objects.inv' \) -delete
	rm -rf _static _sources _modules

# Build the Sphinx site and copy the generated HTML artefacts into docs/
# itself so the repository can be served directly from the GitHub Pages
# /docs folder configuration.
github-pages: html
	cp -R "$(HTMLDIR)/." .
	touch .nojekyll
