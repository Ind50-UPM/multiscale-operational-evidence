Model_D package guide
=====================

Model_D now exposes a phase-1 modular architecture for incident processing. The
current implementation already separates detection, segmentation, classification,
recovery assessment, registry evaluation, and exposure-aware occurrence summaries, while
supporting both true rolling-window detection from timed rows and backward-compatible
processing of pre-windowed indicator tables.

It also supports the expected operational input format for the next phase: a
single Excel workbook whose sheets contain sequence inputs, assignment inputs,
and the canonical incident registry.

Package ``iabm_incidents``
--------------------------

.. autosummary::
   :nosignatures:

   iabm_incidents

Episode Construction
--------------------

.. autosummary::
   :nosignatures:

   iabm_incidents.episodes

Detection
---------

.. autosummary::
   :nosignatures:

   iabm_incidents.detection

Segmentation
------------

.. autosummary::
   :nosignatures:

   iabm_incidents.segmentation

Classification
--------------

.. autosummary::
   :nosignatures:

   iabm_incidents.classification

Recovery
--------

.. autosummary::
   :nosignatures:

   iabm_incidents.recovery

Evaluation
----------

.. autosummary::
   :nosignatures:

   iabm_incidents.evaluation

Configuration
-------------

.. autosummary::
   :nosignatures:

   iabm_incidents.config

Incident Registry
-----------------

.. autosummary::
   :nosignatures:

   iabm_incidents.registry

Occurrence Modelling
--------------------

.. autosummary::
   :nosignatures:

   iabm_incidents.occurrence

Metrics
-------

.. autosummary::
   :nosignatures:

   iabm_incidents.metrics

Command-Line Interface
----------------------

.. autosummary::
   :nosignatures:

   iabm_incidents.main


Batch and Registry Utilities
----------------------------

.. autosummary::
   :nosignatures:

   iabm_incidents.registry_builder

Full-Horizon Reassessment
-------------------------

The repository includes a consolidated ``Model_D`` reassessment for the
requested horizon from **202201** to **202606**. Final executive artifacts are
written under ``src/predictions/Model_D/final_reassessment/`` and include an
HTML report, mode-level and month-level CSV summaries, metadata, and an
explicit caveat file.

The current reference run processed 50 available months and explicitly marks
missing months inside the requested horizon: ``202211``, ``202212``, ``202408``,
and ``202506``. It also flags ``202505`` as a low-data month.

For interpretation, prefer recall-oriented and temporal-overlap metrics over
raw precision-like fields, because the current registry alignment logic can
inflate precision through repeated matches against the same reference context.


Electrical evidence extension
-----------------------------

Model D accepts the additional command-response and fourth-order transient
components documented in ``src/Model_D/README.md``. The scientific transient
indicator is the positive largest eigenvalue of a reference-normalized
fourth-order scatter, raised to the power 1/4. VQE is an optional approximation
of the dominant direction, exported separately from the exact indicator.
Observers and references must be fitted on explicit nominal partitions before
pipeline evaluation. Missing command/reference support remains explicit.

.. autosummary::
   :nosignatures:

   iabm_incidents.electrical

Post hoc event interpretation
-----------------------------

See :doc:`../event_frequency_interpretation` for the optional event-conditioned
frequency-response stage, its FIR/Bode equations, scientific interpretation
limits and exported artifacts. This stage is separate from reliability scoring
and does not change detected episodes.

Four-level ablation
-------------------

See :doc:`../model_d_ablation` for the matched M0--M3 protocol and
:doc:`../model_d_ablation_results` for the executed focused-study results.
The comparison separates behavioral evidence, command residuals, exact
fourth-order evidence and its VQE approximation, with explicit validation
provenance and unavailable metrics.

Multiscale sensitivity
----------------------

The original ablation is retained as the primary experiment. See
:doc:`../model_d_multiscale` for the single prespecified 15-min/1-h/6-h/24-h
sensitivity, matched endpoint support and constant decision cadence, and
:doc:`../model_d_multiscale_results` for the executed English assessment.
