# Main documentation target
all-doc: publish
.PHONY: all-doc

# Set heading text
building = documentation

# Additional variables
SPHINXOPTS  =
SPHINXBUILD = python -m sphinx
SOURCEDIR   = .
BUILDDIR    = build
PUBLISHDIR  = publish

# The content of README.md placed together with the published documentation
define README_body
PROSPECT documentation
=======================
<img align="right" height="250" src="/_static/render2D.png"/>

This is the git branch for the published PROSPECT documentation,
which is most likely not what you are looking for.

Useful links:

* [Hosted documentation][documentation_hosted]
* [Source code for documentation][documentation_source]
* [Code repository][code_repo]

[documentation_hosted]: https://AarhusCosmology.github.io/prospect/
[documentation_source]: https://github.com/AarhusCosmology/prospect_public/tree/main/doc/
[code_repo]: https://github.com/AarhusCosmology/prospect_public/

endef
export README_body

# HTML documentation using Sphinx
sphinx: $(MAKEFILE_LIST)
	@$(call heading)
	@# Forces Sphinx to remake HTML
	@touch index.rst
	@# Build HTML with Sphinx
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
.PHONY: sphinx

# HTML ready for publication
publish: sphinx clean-sphinx-publish
	@$(call heading)
	@mkdir -p $(PUBLISHDIR)
	@cp -r $(BUILDDIR)/html/* $(PUBLISHDIR)/
	@touch $(PUBLISHDIR)/.nojekyll
	@echo "$$README_body" > $(PUBLISHDIR)/README.md
	@printf "\nThe html documentation has been collected in \"$(PUBLISHDIR)\"\n"
.PHONY: publish

# Cleanup targets
.PHONY:                  \
    clean-sphinx-build   \
    clean-sphinx-publish \
    clean-sphinx         \

clean-sphinx-build:
	$(RM) -r "$(BUILDDIR)"
clean-sphinx-publish:
	$(RM) -r "$(PUBLISHDIR)"/{.[!.git],}*
clean-sphinx: clean-sphinx-build clean-sphinx-publish
