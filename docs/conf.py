"""Sphinx configuration for the Multi-Scale Operational Evidence documentation.

The site is generated from three sources:

* hand-written narrative pages (``docs/*.rst``) describing concepts, the
  evidence model, experiments and reproducibility;
* a systematic API reference produced by ``sphinx.ext.autosummary`` with
  ``:recursive:`` stubs, rendered from Google-style docstrings through
  ``sphinx.ext.napoleon``;
* UML material: class-inheritance diagrams generated from the code by
  ``sphinx.ext.inheritance_diagram`` (Graphviz) and hand-maintained Mermaid
  component, class and sequence diagrams.

Maintenance notes:
    * Every importable package lives in its own ``src/Model_*`` folder; the
      folders are prepended to ``sys.path`` below so autodoc can import them
      without installing the packages.
    * Optional heavy dependencies are mocked (``autodoc_mock_imports``) so the
      documentation builds in a light environment (see ``docs/requirements.txt``).
    * Generated API stubs are written to ``docs/api/generated`` and are not
      version controlled; delete that folder when modules are renamed.
"""

from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PATHS = [
    REPO_ROOT / "src",
    REPO_ROOT / "src" / "Model_A",
    REPO_ROOT / "src" / "Model_B",
    REPO_ROOT / "src" / "Model_C",
    REPO_ROOT / "src" / "Model_D",
]

for package_path in PACKAGE_PATHS:
    sys.path.insert(0, str(package_path))


class MermaidDirective(Directive):
    """Fallback Mermaid directive used when ``sphinxcontrib-mermaid`` is absent.

    The directive emits a raw ``<pre class="mermaid">`` block that the Mermaid
    JavaScript runtime (added in :func:`setup`) renders client side, so the
    diagrams embedded in docstrings stay visible in any HTML build.
    """

    has_content = True
    optional_arguments = 1
    option_spec = {"caption": str, "alt": str, "align": str, "zoom": bool}

    def run(self):
        """Return the raw HTML node holding the Mermaid source.

        Returns:
            list[docutils.nodes.raw]: A single raw HTML node.
        """
        code = "\n".join(self.content)
        return [nodes.raw("", f'<pre class="mermaid">{code}</pre>', format="html")]


project = "Multi-Scale Operational Evidence"
author = "Antonio Bello-García, Javier Villalba-Díez, Ana González-Marcos, Joaquín Ordieres-Meré"
copyright = "2026, the Multi-Scale Operational Evidence authors"
release = "1.0.1"
version = "1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
]
try:
    _mermaid_spec = find_spec("sphinxcontrib.mermaid")
except ModuleNotFoundError:
    _mermaid_spec = None
if _mermaid_spec is not None:
    extensions.append("sphinxcontrib.mermaid")

# API reference generation.
autosummary_generate = True
autosummary_imported_members = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_inherit_docstrings = False
autodoc_preserve_defaults = True
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_mock_imports = ["xgboost", "matplotlib", "typer", "rdflib"]

# Google-style docstrings are the project convention.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = False
napoleon_use_rtype = False
napoleon_attr_annotations = True

# UML diagrams generated from the code.
graphviz_output_format = "svg"
inheritance_graph_attrs = dict(rankdir="LR", size='"9.0, 12.0"', fontsize=11)
inheritance_node_attrs = dict(shape="box", fontsize=11, style='"rounded"', color='"#2A78D6"')

mermaid_version = "11.4.1"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md", "user_validation_checklist.md"]
suppress_warnings = ["ref.ref", "autosummary.import_cycle"]

html_theme = "furo" if find_spec("furo") is not None else "alabaster"
html_title = "Multi-Scale Operational Evidence"
html_static_path = ["_static"]


def setup(app):
    """Register static assets and the Mermaid fallback.

    Args:
        app (sphinx.application.Sphinx): Active Sphinx application.

    Notes:
        When ``sphinxcontrib-mermaid`` is not installed the CDN runtime and the
        fallback directive above are registered instead, so docstring diagrams
        never break the build.
    """
    app.add_css_file("custom.css")
    if _mermaid_spec is None:
        app.add_js_file("https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js")
        app.add_js_file(None, body="document.addEventListener('DOMContentLoaded', function () { if (window.mermaid) { mermaid.initialize({ startOnLoad: true }); } });")
        app.add_directive("mermaid", MermaidDirective)
