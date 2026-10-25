Development guide
=================

This page collects the conventions a maintainer needs to extend the code
without breaking the reproducibility guarantees of the manuscript.

Repository layout
-----------------

.. code-block:: text

   multiscale-operational-evidence/
   ├── data/            raw telemetry (Git LFS; see data/README.md)
   ├── docs/            Sphinx sources (this site)
   ├── ontology/        OWL/Turtle schema, examples and SPARQL queries
   ├── reports/         versioned experiment outputs (Git LFS)
   └── src/
       ├── Model_A/     iabm            – state inference
       ├── Model_B/     iabm_behavior   – sequences
       ├── Model_C/     iabm_semantics  – semantic modes
       ├── Model_D/     iabm_incidents  – evidence, episodes, experiments
       ├── predictions/ derived artifacts (small ones versioned with LFS)
       └── *.py         batch and experiment drivers

Docstrings
----------

* Google style, parsed by ``sphinx.ext.napoleon``; every module, class,
  function and method (including private helpers) is documented.
* Module docstrings state the role of the module in the pipeline, its data
  contract (DataFrame columns, dtypes, half-open time windows, phase labels)
  and *Maintenance notes*.
* Mermaid diagrams may be embedded in docstrings with ``.. mermaid::``.
* Inside function bodies, use ``# Note:``, ``# Invariant:`` and ``# Why:``
  comments only for non-obvious logic (vectorised tricks, leakage guards,
  numerical safeguards).

Invariants that must not be broken
----------------------------------

* **Chronological partitions.** Observers, whitening and :math:`F_0` are fitted
  on the observer/reference partitions only; thresholds and conformal
  calibration use the calibration partition only; nothing is tuned on the
  held-out test partition.
* **No zero-imputation of evidence.** Unsupported dimensions are ``NaN`` and
  excluded from common support or from the Cauchy combination.
* **Variational bound.** ``transient_concentration_vqe`` must never exceed the
  exact score (checked in :class:`iabm_incidents.ablation.FourLevelAblationRunner`).
* **Read-only primary artifacts.** The multiscale and extension runners hash
  the primary study before and after execution.
* **Certification must be provable.** Only bounds with a proved gap may be
  used to skip the exact solver (see :doc:`model_d_extensions`).

Testing
-------

.. code-block:: bash

   for pkg in Model_A Model_B Model_C Model_D; do (cd src/$pkg && python -m pytest -q tests); done

``src/Model_D/tests/test_extensions.py`` covers the quantum, conformal and
injection modules, including the Gaussian FOBI null, QJSD bounds, Pauli
reconstruction, unbiased mitigation, rigorous certificate coverage, conformal
validity and the injection algebra.

Large files
-----------

``*.csv``, ``*.parquet`` and ``*.npz`` are tracked with Git LFS (see
``.gitattributes``), including the raw telemetry under ``data/``. Only the
bulky monthly Model B/C predictions are ignored; regenerate them with the
batch scripts.

Continuous integration
----------------------

``.github/workflows/ci.yml`` installs the four packages, runs their test
suites and builds this documentation; on ``main`` it publishes the HTML to
GitHub Pages.
