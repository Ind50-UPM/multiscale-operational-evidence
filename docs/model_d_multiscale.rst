One multiscale window-length sensitivity
========================================

Prespecified supplemental experiment
------------------------------------

This page documents one supplemental sensitivity analysis of the matched
M0--M3 experiment. Its only scale axis is

.. math::

   w\in\{15\;\mathrm{min},1\;\mathrm{h},6\;\mathrm{h},24\;\mathrm{h}\}.

All four existing evidence levels are retained at every scale. M3 replaces the
exact fourth-order coordinate with its VQE approximation; Bode interpretation
remains post hoc and never enters the reliability vector. No additional feature,
threshold, circuit or forecasting-model sweep is performed, and no window is
selected retrospectively from held-out results.

The original 24-hour/hourly study is retained as the primary experiment in
``reports/four_level_ablation/2022_focused``. Sensitivity outputs live in the
separate ``multiscale_sensitivity`` directory. An artifact integrity manifest
checks every primary file before and after execution; the script rejects nested
or identical output directories.

Decision cadence, shared endpoints and phase boundaries
-------------------------------------------------------

A fixed **15-minute decision cadence** applies to all sensitivity scales. This
keeps the decision grid constant while varying only the lookback length and
avoids the blind spots that an hourly cadence would create for 15-minute
windows. Consequently the supplemental 24-hour arm is not a replay of the
primary hourly-cadence arm. Its comparison with other sensitivity arms is
controlled; comparing it numerically with the primary run also changes cadence
and cohort support.

Every sensitivity arm uses identical asset, decision-endpoint and phase keys.
The widest 24-hour lookback determines common warmup and phase eligibility:
endpoints straddling a partition boundary for that widest window are excluded
from all arms. Reference, calibration and held-out partitions remain the
original UTC intervals in :doc:`model_d_ablation`:

* observer/behavioral fit: 1--22 July 2022;
* operator reference and nominal score scaling: 22 July--5 August;
* threshold calibration and risk training: 5--19 August;
* held-out evaluation: 19 August--11 September, exclusive at the source boundary.

A window contains source observations in ``[end-w, end)``. All scales use the
same labelled outcomes and right-censoring boundary. The forecast outcome-horizon
purge is fixed at **19 August 00:00 UTC**, rather than inferred separately from
each arm's shorter lookback start. This prevents window length from changing
the allowed risk-training rows.

Frozen models and scale-dependent calibration
---------------------------------------------

The script restores the primary command observer from its saved expected
phase levels/scales and restores the saved fourth-order reference. These
objects are not refitted to the sensitivity or test windows. The same raw
analogue/digital source hashes and retained Model A predictions must match the
primary input provenance.

Whitening, eight-coordinate causal embedding, nominal fourth-order scatter,
normalization, per-state support, gap/settling masks and observed 0.807-second
cadence are identical. Each supported state/window operator is constructed from
its actual samples; a long-window score is not interpolated or rescaled to
fabricate short-window evidence.

The behavioral encoder is fitted on the original observer-fit partition for
each window length. Sequence/run-duration references and the mode proxy retain
the same definitions. Occupancy 95th-percentile references use windows of that
arm's length on the common 15-minute grid. This is necessary because occupancy
variability depends on the integration interval.

Coordinate centers/scales are fitted on the common reference endpoints, and
thresholds use the common calibration endpoints, independently for each scale.
Onset/recovery quantiles remain 95%/80%. Onset confirmation retains two threshold-exceeding windows under the
existing gap/recovery policy; recovery confirmation remains three windows. Within the sensitivity,
these counts imply the same decision-cadence persistence policy in every arm;
they do not equal the original hourly experiment's clock-time confirmation
policy. Duration/gap policies, weights, family classifier and risk model remain
fixed.

M3 shares its arm's exact transient center, scale and thresholds. VQE uses the
original three RY/CX layers, six Rotosolve sweeps, two restarts and seed 1729.
Every supported operator receives the same budget, including difficult ones.
There is no exact eigenvector initialization, fallback or test-window selection.

Matched support and selection limits
------------------------------------

The primary analysis used matched support across evidence levels. The sensitivity
uses the stricter intersection across **all four scales and all four evidence
levels**, before fitting score distributions or evaluating risk/episodes.
Missing dimensions are never zero-imputed. Observation, electrical-safety,
reference and minimum prediction/vector counts are not relaxed for short
windows.

The availability table retains every candidate endpoint and identifies each
arm's own support, missing dimensions and cross-scale support. All reported arm
comparisons therefore have identical observed decision exposure and identical
future-target labels. Retrospective event matching uses one fixed cohort:
records whose event timestamp falls inside a common-supported shortest-window
observation interval. Longer lookbacks cannot enlarge that reference denominator.
Forecasting still uses the complete future-event registry, including events
outside predictor-support intervals. Rows excluded by another arm remain explicitly marked in
the per-scale ablation support table.

Short windows can have too few retained Model A predictions, excessive relative
settling exclusions, sparse state-specific delay vectors or missing nominal
state support. Removing their endpoints from every scale can concentrate the
cohort around active command regimes. The results are conditional on that
cohort, not representative of all calendar hours or each arm's independent
maximum coverage.

Excluded endpoints break episode and recovery blocks even when long lookbacks
overlap. Fragmentation caused by those support gaps is not a physical-failure
count. No scale is declared operationally preferable from this analysis alone.

Metrics and interpretation
--------------------------

The same per-level artifacts as :doc:`model_d_ablation` are exported:

* candidate episodes, lookback spans and exposure-normalized warning burden;
* fragmentation and timing against fixed weak signal-event references;
* raw family evidence and existing rule-based family labels;
* threshold-confirmed recovery and its support/study-boundary censoring;
* fixed-target 6/24/72/168-hour forecasting metrics and coefficients;
* exact/VQE scores, resource budget, directions and retrospective peak locations.

Decision exposure is measured on the 15-minute interval after window completion,
truncated at the study boundary. Overlapping lookbacks are not summed as warning
hours. Shorter windows can reduce lookback-based onset displacement and produce
narrower candidate spans, but this does not establish physical-onset accuracy
or true warning lead time.

The earlier maintenance-source audit still applies. Reported-maintenance
comparison records cover 2024--2026, their original Word files are absent, and
no independent healthy/onset/recovery records exist for this focused 2022
comparison. Missing independent validation metrics remain unavailable. Derived
signal events are exploratory references; their absence does not establish a
false warning. Single-class forecast horizons remain explicitly non-discriminative.

Raw sampling is not downsampled or replaced by synthetic telemetry. The retained
Model A time-split provenance is still unknown, and its sparse prediction
coverage remains a shared upstream limitation. The inherited valley-based
maximum-duration split remains a soft lookback-span target, audited in the
underlying ablation results.

Execution and artifacts
-----------------------

.. code-block:: bash

   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src/Model_D \
     python src/run_model_d_multiscale.py

   # Re-evaluate the completed sensitivity without rerunning signal extraction/VQE:
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src/Model_D \
     python src/run_model_d_multiscale.py --reuse-indicators

The default output is
``reports/four_level_ablation/multiscale_sensitivity``. The English
``sensitivity_report.html`` and ``assessment.md`` present this one supplemental
experiment. Aggregate summary/pairwise/risk tables, the full endpoint
availability audit, input provenance and the primary integrity manifest sit
alongside the four ``window_<minutes>min`` directories. Each arm retains raw
indicators, identical exact/VQE operators, VQE directions, context scores and
its complete M0--M3 outputs.

.. code-block:: python

   from iabm_incidents import (
       MultiscaleSensitivityConfig, MultiscaleSensitivityRunner,
   )

   sensitivity = MultiscaleSensitivityRunner(MultiscaleSensitivityConfig())
   result = sensitivity.run(
       indicators_by_scale={
           15: indicators_15min,
           60: indicators_1h,
           360: indicators_6h,
           1440: indicators_24h,
       },
       incidents=fixed_reference_records,
       independent_incidents=False,
       study_end=study_end,
   )
   result.export("reports/prespecified_sensitivity")

The class rejects mixed grids, mismatched phase labels, incorrect lengths and
unplanned/missing arms. Nominal/calibration support requirements remain those of
the four-level runner. See :doc:`model_d_multiscale_results` for the executed
real-data results and English assessment.

API reference
-------------

.. autosummary::
   :nosignatures:

   iabm_incidents.multiscale
