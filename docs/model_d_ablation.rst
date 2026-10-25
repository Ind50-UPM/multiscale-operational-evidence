Four-level Model D ablation
===========================

Evidence levels and scientific question
---------------------------------------

The reusable ``FourLevelAblationRunner`` compares four nested evidence sets:

.. list-table::
   :header-rows: 1
   :widths: 10 55 35

   * - Level
     - Indicator vector
     - Purpose
   * - M0
     - Seven behavioral Model D indicators
     - Behavioral comparator
   * - M1
     - M0 plus command-response residual magnitude
     - Contribution of command conditioning
   * - M2
     - M1 plus exact fourth-order transient concentration
     - Contribution of classical fourth-order evidence
   * - M3
     - M1 plus VQE transient concentration
     - Approximation of the identical fourth-order operator

M3 **replaces**, rather than supplements, the exact transient coordinate. The
post hoc frequency-response outputs are not features in any level. All levels
use the same supported windows, incident references, family classifier and
segmentation policy. The important comparisons are M1--M0, M2--M1 and M3--M2.

The runner accepts already materialized indicators so it can also compare a
validated seven-coordinate historical baseline with any frozen observer or
transient reference. It never silently replaces a missing indicator by zero.

Focused real-data reproduction
------------------------------

The executable ``src/run_model_d_ablation.py`` uses the original historical
analogue/digital Parquet files and the notebook's focused UTC partitions:

.. list-table::
   :header-rows: 1

   * - Stage
     - Start, inclusive
     - End, exclusive
   * - Command observer and behavioral reference fit
     - 2022-07-01
     - 2022-07-22
   * - Fourth-order reference and score-coordinate scaling
     - 2022-07-22
     - 2022-08-05
   * - Detection calibration and risk fitting
     - 2022-08-05
     - 2022-08-19
   * - Held-out comparison
     - 2022-08-19
     - 2022-09-11

Model D uses 24-hour rolling windows with a one-hour step. Windows straddling a
partition boundary are excluded. This is a longitudinal Model D ablation, not
an exact reproduction of the notebook's 256-sample detector experiment.

Commands are taken from ``digitales.parquet/estado`` and matched to electrical
samples by a causal backward as-of join with a two-second tolerance. Unknown
commands are not replaced by inferred Model A states. Duplicate timestamps,
source hashes, observed cadence, unknown-command count, fitted references and
all VQE resource budgets are retained. Sampling is not downsampled, resampled or
replaced with synthetic telemetry.

The historical saved baseline contains semantic scores rather than a complete
seven-indicator table. Therefore the focused run uses a newly fitted
``BehavioralWindowEncoder`` to materialize M0. It must not be described as a
numerical replay of that saved semantic baseline:

* ``sequence_divergence``: Jensen--Shannon divergence, in bits, of compressed
  actual-command run-transition distributions. Timestamp-gap transitions are
  excluded.
* ``duration_drift``: run-frequency-weighted absolute deviation of within-state
  median dwell time from its fitted median, divided by nominal MAD with a
  one-sample floor. Rare unseen states have no invented dwell reference; they
  remain in sequence/occupancy evidence, while dwell uses supported states and
  exports its nominal-state sample fraction. Insufficient support yields NaN.
* ``recurrence_excess``: maximum positive command-run-count excess relative to
  fitted observed-time rates, over supported states.
* ``persistence_excess``: maximum state-occupancy excess relative to nominal
  24-hour occupancy 95th percentiles, with a one-sample occupancy floor.
* ``consumption_deviation``: mean signed three-phase command-state residual.
  It retains its sign; aggregate deviation scoring uses the magnitude of its
  departure from the nominal center.
* ``state_error_rate``: disagreement of retained ``pred_estado`` with the
  actual command, on exactly timestamp-matched available Model A predictions.
  Its prediction count and coverage fraction are exported. Missing predictions
  are not counted as agreements.
* ``mode_divergence``: Jensen--Shannon divergence of active-control-bit-count
  distributions. This is a disclosed command-activity proxy, not the complete
  Model C semantic mode ontology.

The original Model A fit/holdout provenance is not reconstructed from the saved
predictions. Thus chronological claims apply to the newly fitted observer,
behavioral references, transient reference, scaling, calibration and risk fit;
they do not certify the temporal independence of the retained Model A stage.
The focused results are conditional on that shared upstream input.

The command observer is the implemented robust same-state median/MAD observer
with a settling guard, not the notebook's richer state-and-age observer. Its
reference-period phase-wise standardized errors are exported for review; these
are adequacy diagnostics, not independently certified healthy observations.

Matched support and score calibration
-------------------------------------

Every level uses the intersection of finite seven-coordinate behavioral,
command, exact and VQE evidence. Observation coverage must satisfy the Model D
policy, and the electrical extension must declare supported coverage. The
complete support audit retains excluded windows, missing dimensions and phase.
Results describe this common support, not uninterrupted calendar exposure or
each model's maximum standalone coverage.

Reference-only median centering and robust scaling use

.. math::

   s_j=\max\left(1.4826\,\operatorname{MAD}_0(z_j),
       \frac{Q_{0.95,0}(z_j)-Q_{0.05,0}(z_j)}{3.29},10^{-6}\right).

Directional excess is ``max((z-center)/scale, 0)``. Consumption uses absolute
centered deviation. The existing nonnegative ``DeviationWeights`` combine these
coordinates; all weights are one in the focused experiment. Raw values and
signed consumption remain in the exported tables. This calibrated ablation
scorer is distinct from the package's legacy raw weighted scorer.

Each level's onset threshold is its calibration score 95th percentile;
recovery uses the 80th percentile. These quantiles define a matched nominal
warning-budget target, not measured false-alarm rates. Equal-value ties are
retained and the actual calibration warning fractions are reported.

M3 shares M2's transient coordinate center, scale and, by default, both detection
thresholds. Its independently calibrated thresholds are exported as a secondary
audit. This primary protocol isolates solver approximation: under the
variational bound and nonnegative weights, M3's score cannot exceed M2's. To
study independently calibrated decision distributions instead, set
``share_exact_vqe_threshold=False`` and clearly identify that second protocol.

The focused VQE uses three real RY/CX layers, six Rotosolve sweeps, two restarts
and seed 1729. Batch vectorization gives each operator independent angles and
updates. Every supported operator receives the same budget. Classical
eigenvectors never initialize or replace a poor variational solution. Stored
operators and directions permit direct Rayleigh-bound audits; no hardware or
quantum advantage is claimed.

Detection, fragmentation, warning burden and recovery
-----------------------------------------------------

The same Model D episode builder, two-window onset requirement, three-window
recovery confirmation, duration limits and family classifier are used across
levels. ``semantic_status`` does not switch this experiment into the alternate
semantic segmentation route. Excluded decision windows break segmentation
blocks even when their 24-hour lookbacks overlap. Different assets cannot share
recovery evidence. Episode IDs are unique across those blocks.

The report distinguishes:

* warning windows and warning hours per 100 observed **decision** hours;
* episode counts, episode span and episodes per 100 observed decision hours;
* reference-event temporal overlap and episodes per hit reference;
* warning hours on independently healthy intervals, when supplied;
* onset decision error against timestamped references;
* family evidence and family agreement against those references;
* threshold-confirmed recovery versus independently documented recovery error.

Warning exposure is evaluated on each one-hour decision interval after window
completion, truncated at the supplied study end. Overlapping 24-hour lookbacks
are not summed as 24 hours of warning for each decision. Episode spans remain
lookback-based spans and should not be confused with operational alarm duration.

Fragmentation against point/interval references counts overlapping episodes
per hit reference and excess episodes beyond one. One episode may cover several
references; those records are not assumed independent physical failures.
Matching is asset-constrained and requires actual overlap, without forcing
agreement of incident family. An unmatched episode is not automatically a false
alarm when reference labels are incomplete.

The onset availability timestamp is the end of the onset-confirming window,
not the first lookback-window start. Within-window projected impulse peaks are
retrospective localization only; exact and stored-VQE projection peaks are retained in the raw
indicator artifact for subsequent post hoc examination. They are not adjudicated
physical onsets. Matched exact/VQE peak displacement is exported as a
projection-consistency diagnostic, not physical-onset accuracy. Recovery is censored at unsupported blocks and the study end.
A stable score recovery does not establish physical repair. The inherited
segmenter's valley-based duration split is not a strict hard bound on episode
lookback spans. The focused report explicitly counts episodes that exceed the
configured maximum duration; their fragmentation must not be interpreted as
independently confirmed physical failures.

Maintenance and incident source audit
-------------------------------------

A repository-wide search located the retained maintenance comparison under
``src/predictions/episode_eval_full/comparison_report`` and its extraction code
``src/generate_probe_incident_timeline_report.py``. That comparison concerns
February 2024--April 2026: 14 reported maintenance periods and 11 work-order
points. Its two original Word documents are referenced as
``/tmp/Tabla_Incidencias.docx`` and
``/tmp/OT - INCIDENCIAS LAVA RUEDAS.docx``; those files are absent in this session.
Only aggregate comparison counts and plots remain. They cannot provide new
M0--M3 interval matches and their dates do not overlap the 2022 focused study.
The old 14/14 overlap must not be copied into this experiment.

The candidate registry is automatically generated from electrical/state,
sequence-duration and gap evidence. It contains no populated maintenance or
recovery timestamps. It is used only as a **fixed weak-reference process**,
with ``weak_reference_`` metric names and explicit outcome provenance. The
ontology incident file is an example, not an operational registry.

Accordingly, independent maintenance overlap, false-warning burden, physical
onset/recovery accuracy and physical-failure risk validation remain explicitly
unavailable for the focused run. NaN is not zero, and unavailable is not failed
validation. The reusable API accepts independent incident, intervention and
healthy-interval tables when such evidence is available.

Risk model evaluation
---------------------

Every level uses the same fixed external event targets. Targets never come
from that level's detected episodes. The existing interpretable
``AnticipatoryMaintenanceForecaster`` is fitted using only the corresponding
level's transformed coordinates. In the focused experiment the event targets
are future candidate ``pump_abrupt_failure`` or ``process_saturation`` records,
so all reported performance is explicitly a weak-reference proxy.

The original 24-, 72- and 168-hour horizons are evaluated, plus a six-hour
short-horizon diagnostic. Calibration-period training rows
must finish their complete outcome horizon before the test start. Held-out
rows whose outcome horizon extends beyond the study boundary are censored.
The same labelled rows, outcome events and censoring apply to every level.
Precision, recall, F1, tie-aware ROC AUC and average precision, Brier score and a
training-prevalence Brier comparator are exported. The probability threshold is
selected by the existing forecaster on training only. A single-class target is
flagged explicitly; near-perfect metrics on an all-positive outcome are not
useful discrimination.

These point estimates do not establish statistical significance: rolling
predictors overlap and future outcomes are serially correlated. Prediction of
an automatically derived event process is not prediction of independent
physical failure. Sparse reported-event logs likewise require a completeness
audit before absence of a report can be treated as a negative physical outcome.

Execution and public API
------------------------

.. code-block:: bash

   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src/Model_D \
     python src/run_model_d_ablation.py

   # Re-evaluate completed materialized indicators without rerunning VQE:
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src/Model_D \
     python src/run_model_d_ablation.py --reuse-indicators

The default output is ``reports/four_level_ablation/2022_focused``. It contains
raw indicators, operator/context scores, all identical operators and VQE
directions, frozen electrical references, observer diagnostics, source hashes,
source audit, calibration, common support, pairwise decisions, per-level scores,
episodes, family evidence, fixed-target risk predictions, and English Markdown
and HTML assessments. Review the input/evaluation source hashes when reusing
artifacts; reuse does not refit or redefine previously materialized indicators.

.. code-block:: python

   from iabm_incidents import FourLevelAblationRunner, AblationConfig

   runner = FourLevelAblationRunner(AblationConfig())
   result = runner.run(
       indicators=materialized_indicators,  # reference/calibration/test phases
       incidents=reported_incidents,
       independent_incidents=True,          # explicit evidence provenance
       maintenance=reported_intervals,      # asset_id/start_time/end_time
       healthy_intervals=adjudicated_healthy_intervals,
       study_end=study_end,
   )
   result.export("reports/independent_ablation")

The focused assessment and numerical results are documented in
:doc:`model_d_ablation_results`.

Optional comparison figures use the package's ``analysis`` extra (Matplotlib).
The core ablation, tabular/HTML export and simulated VQE do not require it.

API reference
-------------

.. autosummary::
   :nosignatures:

   iabm_incidents.ablation

.. autosummary::
   :nosignatures:

   iabm_incidents.behavioral
