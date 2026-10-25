Reproducibility
===============

This page lists, in execution order, the commands that regenerate every
artifact used by the manuscript, with their inputs and outputs. All commands
run from the repository root. Package-level details are in
:doc:`packages/index`; the manuscript-to-code mapping is in :doc:`paper_map`.

.. mermaid::

   flowchart TD
     RAW[data/*.parquet] --> BC[generate_model_bc_batch.py]
     RAW --> REGB[generate_candidate_incident_registry_batch.py]
     REGB --> REG[(candidate_incident_registry_refined.csv)]
     RAW --> ABL[run_model_d_ablation.py]
     REG --> ABL
     ABL --> PRIM[(reports/four_level_ablation/2022_focused)]
     PRIM --> MS[run_model_d_multiscale.py]
     RAW --> MS
     MS --> MSO[(reports/four_level_ablation/multiscale_sensitivity)]
     PRIM --> EXT[run_model_d_extensions.py]
     MSO --> EXT
     RAW -. telemetry stage .-> EXT
     EXT --> EXTO[(reports/extensions)]
     EXTO --> FIG[export_extension_figure_data.py]
     FIG --> FIGO[(reports/extensions/figure_data)]

Prerequisites
-------------

* Raw telemetry in ``data/`` (see ``data/README.md``).
* Packages installed (see :doc:`getting_started`) and
  ``export PYTHONPATH=src:src/Model_D``.

Step 1 – longitudinal behaviour (optional, long running)
--------------------------------------------------------

``python src/generate_model_bc_batch.py`` runs Model B and Model C month by
month and writes ``src/predictions/Model_B`` and ``src/predictions/Model_C``
(not versioned). ``python src/generate_candidate_incident_registry_batch.py``, which reads the state, sequence and word Parquet files directly,
derives the weak-reference registry
``src/predictions/Model_D/candidate_incident_registry_refined.csv``, which *is*
versioned, so this step can be skipped to reproduce the paper tables.

Step 2 – matched M0–M3 ablation
-------------------------------

``python src/run_model_d_ablation.py`` loads July–September 2022 telemetry,
fits the frozen command observer, the behavioural encoder and the fourth-order
references on declared chronological partitions, solves every context operator
exactly and with the three-qubit VQE, and evaluates the four evidence levels.
Outputs: ``reports/four_level_ablation/2022_focused``. ``--reuse-indicators``
replays the evaluation from the stored indicators.

Step 3 – multiscale sensitivity
-------------------------------

``python src/run_model_d_multiscale.py`` restores the frozen primary models and
repeats the study at 15 min, 1 h, 6 h and 24 h on a common 15-minute grid.
Outputs: ``reports/four_level_ablation/multiscale_sensitivity``. The script
verifies by hashing that the primary artifacts are unchanged.

Step 4 – extensions
-------------------

``python src/run_model_d_extensions.py --stage all`` (or one of
``model_a``, ``quantum``, ``telemetry``, ``fusion``) writes
``reports/extensions``: Model A class metrics, certificates, information
profiles, shot/noise and noisy-VQE studies, order statistics and fault
injection, conformal fusion, alternative transient slots and baselines. Only
the ``telemetry`` and ``model_a`` stages need the raw Parquet files.

Step 5 – figure data
--------------------

``python src/export_extension_figure_data.py`` writes the compact CSV tables
read by the pgfplots figures of the manuscript to
``reports/extensions/figure_data``.

Determinism
-----------

All random draws use fixed seeds recorded in the code and in the JSON
metadata files (VQE restarts, shot sampling, conformal tie-breaking,
injection placement). Re-running with the same inputs reproduces the tables
exactly; small floating-point differences (≈10⁻⁸) can occur in VQE directions
across NumPy builds.
