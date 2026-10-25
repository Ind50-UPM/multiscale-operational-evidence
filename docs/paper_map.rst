Manuscript-to-code map
======================

The table links each part of the manuscript to the code and artifacts that
produce it.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Manuscript element
     - Code
     - Artifacts
   * - Model A state inference, Table of class-level diagnostics
     - :mod:`iabm.models`, :func:`run_model_d_extensions.stage_model_a`
     - ``reports/extensions/model_a_class_metrics.csv``, ``model_a_period_metrics.csv``
   * - Duration-aware sequences and modes
     - :mod:`iabm_behavior.sequences`, :mod:`iabm_semantics.semantics`
     - ``data/secuencias.parquet``, ``src/predictions/Model_B``, ``Model_C`` (regenerated)
   * - Evidence vector, command-response residuals, fourth-order operator
     - :mod:`iabm_incidents.behavioral`, :mod:`iabm_incidents.electrical`
     - ``reports/four_level_ablation/2022_focused/raw_indicators.parquet``
   * - Temporal consolidation (242 → 127 episodes)
     - :mod:`iabm_incidents.episodes`, :mod:`iabm_incidents.segmentation`
     - ``src/predictions/episode_eval_full/comparison_report/comparison_summary.csv``
   * - Matched M0–M3 ablation
     - :mod:`iabm_incidents.ablation`, :mod:`run_model_d_ablation`
     - ``reports/four_level_ablation/2022_focused``
   * - Multiscale sensitivity
     - :mod:`iabm_incidents.multiscale`, :mod:`run_model_d_multiscale`
     - ``reports/four_level_ablation/multiscale_sensitivity``
   * - Density-operator reading, certificates, shot/noise study
     - :mod:`iabm_incidents.quantum`, :func:`run_model_d_extensions.stage_quantum`
     - ``certificate_summary.csv``, ``shot_noise_study.csv``, ``noisy_vqe_study.csv``
   * - Fault-injection benchmark
     - :mod:`iabm_incidents.injection`, :func:`run_model_d_extensions.stage_telemetry`
     - ``injection_summary.csv``, ``injection_windows.parquet``
   * - Conformal fusion, baselines, multiscale fusion
     - :mod:`iabm_incidents.conformal`, :func:`run_model_d_extensions.stage_fusion`
     - ``fusion_summary.csv``, ``fusion_attribution.csv``, ``conformal_primary_pvalues.csv``
   * - pgfplots figures of the manuscript
     - :mod:`export_extension_figure_data`
     - ``reports/extensions/figure_data/*.csv``
   * - Exploratory future-episode analysis
     - :mod:`iabm_incidents.anticipation`
     - ``src/predictions/anticipation_validation_table.csv``

Citation
--------

Please cite the manuscript and this software; see ``CITATION.cff`` in the
repository root.
