# Documentation sources

Sphinx sources for the **Multi-Scale Operational Evidence** documentation.

```bash
pip install -r docs/requirements.txt        # Graphviz must also be installed
sphinx-build -b html docs docs/_build/html  # open docs/_build/html/index.html
```

* `conf.py` – configuration (autodoc + napoleon for Google docstrings,
  recursive autosummary, Graphviz inheritance diagrams, Mermaid).
* `index.rst` – landing page and navigation.
* `getting_started.rst`, `overview.rst`, `architecture.rst` – introduction.
* `packages/` – one guide per package (Model A–D).
* `model_d_*.rst`, `longitudinal_validation.rst` – experiment protocols and results.
* `reproducibility.rst`, `paper_map.rst`, `uml.rst`, `development.rst` – maintenance.
* `api/index.rst` – entry point of the generated API reference
  (`api/generated/` is created at build time and not versioned).
* `_static/` – CSS and the result tables embedded in the pages.
* `_templates/autosummary/` – module/class templates that add per-module
  class diagrams.
