Known issues and maintenance backlog
====================================

The items below were found while documenting the code for release 1.0.0.
They were **reported, not fixed**, because the release was required to keep
the code that produced the manuscript results unchanged. Each item still
needs to be confirmed with a test before it is changed; none of them affects
the numbers reported in the manuscript unless stated.

Confirmed
---------

* ``tests/test_ablation.py::test_behavioral_partial_nominal_support_and_gap_mask``
  fails under pandas 3 because ``BehavioralWindowEncoder.encode`` writes into
  a read-only array view (copy-on-write). The primary indicators were
  materialised before this pandas version, so reported results are unaffected.
* Episode confirmation time is the end of the second window of a run, which
  can be a tolerated interruption rather than a warning window (see
  :mod:`iabm_incidents.segmentation` and the manuscript methods).

Model A (``iabm``)
------------------

* Evaluate mode with two artifacts of the same algorithm overwrites plot files
  and reuses predictions for the second confusion-matrix plot.
* Missing feature columns at prediction time become ``NaN`` silently.
* ``--lang=es`` (single-token form) is ignored; only ``--lang es`` works.
* Predict mode requires ``--model-out``/``--pred-out`` although defaults exist.
* Cross-validation shuffles individual timestamps (optimistic for serially
  correlated data); imputation uses backward fill inside a window.
* ``pyproject.toml`` lists ``locales/messages.pot``, which does not exist
  (also in Model B and Model C).

Model B / Model C
-----------------

* Some Model C recovery branches are unreachable (``recovering`` regime).
* The partial recovery score is overwritten by later sequences instead of
  keeping the maximum; single-sample runs have zero duration.
* Comparison rows are aligned with sequences by position only.

Model D (``iabm_incidents``)
----------------------------

* ``EpisodeEvaluator`` counts zero-IoU matches as hits and filters candidates
  by family first, so family precision/recall are uninformative.
* ``main.py`` passes a non-normalised registry to the evaluator (possible
  ``KeyError`` on ``event_time``); ``--registry`` is ignored with ``--workbook``.
* ``exposure._collect_time_ranges`` collapses each table into one range per
  asset, so internal gaps are not detected.
* ``detection._infer_fallback_family`` has unreachable branches;
  ``detection._fit_behavioral_baseline`` is fitted in-sample.
* ``anticipation`` has no purge/embargo between train and test, and its
  rolling feature names count rows rather than hours.
* ``registry_builder._merge_nearby_candidates`` never merges recurrence
  candidates because ``NaN == NaN`` is false.
* ``injection.fit_second_order`` does not filter residuals by asset.
* ``quantum.ShotEstimator`` caches Pauli coefficients by array identity.

Scripts
-------

* ``generate_probe_incident_timeline_report.py`` reads two Word inputs from
  hard-coded ``/tmp`` paths that are not distributed.
* ``concat_monthly_csv_to_parquet.py`` adds ``source_month`` only with
  ``--include-source-month``, which downstream runners require.
* ``run_model_d_ablation.py`` hard-codes the Model A month filter separately
  from ``BOUNDARIES``.
