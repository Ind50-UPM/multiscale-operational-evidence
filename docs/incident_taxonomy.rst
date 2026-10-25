Incident Taxonomy
=================

The updated paper relies on an explicit incident-family layer instead of treating
all non-nominal patterns as undifferentiated anomalies. In this repository,
incident families are intended to act as **weak labels** that connect sequence
patterns, semantic interpretation, and longitudinal episodes.

Current family vocabulary
-------------------------

The current implementation exposes a lightweight but explicit vocabulary that can
be extended project by project:

- ``pump_abrupt_failure``
- ``float_recurrent_disturbance``
- ``process_saturation``
- ``post_intervention_recovery``
- ``external_ambiguous_disturbance``

These labels are currently assigned through heuristic or rule-based logic in the
semantic and longitudinal layers. They are therefore best understood as an
interpretable baseline rather than as a final supervised taxonomy.

Why families matter
-------------------

An anomaly score alone is often insufficient for longitudinal interpretation.
Family labels make it possible to ask richer questions such as:

- Which sequence signatures recur across incidents?
- Which families are associated with prolonged recovery windows?
- Which patterns suggest degradation acceleration versus transient disturbance?
- Which weak labels can seed future family-specific occurrence models?

Relation to the pipeline
------------------------

Incident families sit between sequence-level analytics and episode-level
summaries:

- ``Model_B`` derives family-oriented signatures from longitudinal metrics.
- ``Model_C`` assigns family labels and reliability-oriented interpretations.
- ``Model_D`` groups anomalous windows into episodes and occurrence summaries.

Ontology alignment
------------------

The ontology now includes explicit support for this layer through entities such
as ``IncidentFamily``, ``IncidentEpisode``, ``SequenceSignature``, and
``hasWeakLabel``. This keeps family labels inspectable and reusable across the
code and documentation.
