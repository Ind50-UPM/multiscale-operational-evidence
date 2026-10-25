Longitudinal Validation
=======================

The updated methodology requires validation beyond state classification accuracy
or isolated anomaly scores. The repository therefore treats longitudinal
validation as a first-class concern.

Validation targets
------------------

The key targets for longitudinal validation are:

- episode construction from synthetic and known incident scenarios,
- recurrence interval estimation,
- time-to-recovery estimation,
- partial recovery and regime-shift indicators,
- consistency between incident tables and derived temporal windows,
- reproducibility of family-oriented summary metrics.

Current test strategy
---------------------

The repository includes package-level tests that now cover:

- synthetic episode construction in ``Model_D``,
- reproducible recurrence and recovery metrics in ``Model_B``,
- rule-based incident-family interpretation in ``Model_B`` and ``Model_C``,
- incident-window consistency checks through the incident registry alignment
  utilities.

Why longitudinal validation differs
-----------------------------------

In a longitudinal setting, validation is not only about whether one time sample
or one sequence is correctly labeled. It is also about whether the pipeline
preserves temporal coherence across multiple levels:

- state -> sequence coherence,
- sequence -> family coherence,
- family -> episode coherence,
- episode -> known incident table coherence.

This means that the repository uses synthetic fixtures and expected metric tables
as reproducibility anchors, not just isolated unit assertions.

Future directions
-----------------

A future reliability-informed validation program may incorporate:

- annotated intervention logs,
- survival-like recovery analyses,
- family-specific occurrence models,
- uncertainty-aware episode boundaries,
- cross-campaign comparison of longitudinal motifs.
