Semantic Schema
===============

This page describes the ontology-aligned contextualization schema used to support the behavioral monitoring framework implemented in this repository.

Purpose
-------

The goal of the semantic schema is to make explicit the relationships between:

- the physical structure of the monitored asset,
- the functional role of its subsystems and components,
- the observations collected from sensors and control signals,
- the inferred operational states,
- the behavioral sequences extracted from those states,
- the incident families and longitudinal episodes derived from those sequences,
- and the higher-level operating, working, life, and recovery regimes used for interpretation.

The schema is not intended as a complete ontology for industrial asset management or as a full ontology-driven reasoning system. Its role is more focused: to provide a lightweight and explicit contextualization layer that improves interpretability, traceability, and consistency across the analytical workflow.

Federated semantic basis
------------------------

The schema is designed as a lightweight federation of existing semantic resources plus a domain-specific extension.

Reused vocabularies
~~~~~~~~~~~~~~~~~~~

The current conceptual basis is aligned with the following semantic resources:

- **SOSA / SSN**

  Used for representing sensors, observations, observed properties, and features of interest.

- **OWL-Time**

  Used for representing time instants, intervals, durations, and temporal ordering relations.

- **QUDT**

  Used for representing quantity kinds and units associated with industrial measurements.

- **SAREF** *(optional alignment)*

  Can be used where asset-, device-, or function-oriented concepts require a more explicit IoT/device interpretation.

These vocabularies are not necessarily imported in full in every implementation layer. Instead, they provide the conceptual and semantic anchors for the domain-specific schema.

Domain-specific extension
-------------------------

On top of the federated core, the repository defines a minimal domain extension to represent the behavioral monitoring concepts that are specific to the industrial asset use case.

Core classes
~~~~~~~~~~~~

The following classes are central to the schema:

- ``IndustrialAsset``

  The monitored industrial asset as a whole.

- ``Subsystem``

  A structurally meaningful part of the asset, such as a pump group or a control-relevant section.

- ``Component``

  A concrete physical element within a subsystem.

- ``StructuralAnchor``

  A stable structural reference used to preserve traceability between monitored evidence and asset interpretation.

- ``FunctionalRole``

  A process-oriented role associated with a subsystem or component.

- ``OperationalState``

  An elementary inferred operating condition derived from synchronized industrial observations.

- ``BehavioralSequence``

  An ordered temporal succession of operational states.

- ``OperatingMode``

  A higher-level category summarizing recurrent behavioral regimes linked mainly to internal process organization.

- ``WorkingMode``

  A higher-level category summarizing regimes influenced more strongly by external or contextual activation.

- ``StatePrediction``

  A predicted operational state derived from the data-driven identification layer.

- ``AbnormalityEvidence``

  Evidence indicating that a behavioral sequence or operating regime deviates from nominal behavior.

- ``IncidentFamily``

  A weak-label category used to distinguish recurring types of non-nominal behavioral patterns.

- ``IncidentEpisode``

  A longitudinal grouping of pre-event, event, and recovery windows associated with an incident family.

- ``LifeRegime``

  A semantic proxy describing the interpreted evolution of asset behavior over time.

- ``RecoveryRegime``

  A semantic proxy describing post-incident stabilization or intervention effects.

- ``OccurrenceInterval``

  A temporal abstraction used to express recurrence behavior between episodes.

- ``SequenceSignature``

  A reusable representation of family-associated sequence motifs.

Core relations
~~~~~~~~~~~~~~

The following relations structure the contextualization schema:

- ``hasSubsystem``
- ``hasComponent``
- ``hasStructuralAnchor``
- ``hasFunctionalRole``
- ``derivedFromObservation``
- ``evidencesState``
- ``belongsToSequence``
- ``summarizedByMode``
- ``hasWorkingMode``
- ``deviatesFromNominal``
- ``belongsToIncidentFamily``
- ``precedesRecovery``
- ``signalsLifeRegime``
- ``hasWeakLabel``
- ``hasAnomalyScore``
- ``hasRecoveryTime``
- ``hasRecurrenceInterval``

These relations are intended to preserve a traceable link between measured evidence, inferred behavior, longitudinal episodes, and maintenance-relevant interpretation.

Three complementary views
-------------------------

The schema is organized around three complementary views of the monitored asset.

Structural view
~~~~~~~~~~~~~~~

The structural view describes what physically exists in the system.

Typical entities in this view include:

- asset
- subsystem
- component
- sensor
- electrical channel

This view provides the stable anchors required to avoid reducing the monitoring representation to transient telemetry identifiers only.

Functional view
~~~~~~~~~~~~~~~

The functional view describes the intended role of the structural elements in the industrial process.

Typical examples include:

- pumping
- recirculation
- dosing
- standby
- transition support

This view is important because the same component may contribute to different interpreted behaviors depending on the process stage.

Informational view
~~~~~~~~~~~~~~~~~~

The informational view describes how the asset is observed and represented.

Typical entities include:

- measurements
- digital signals
- observations
- state predictions
- behavioral sequences
- incident episodes
- operating, working, life, and recovery regimes

This view links raw monitored evidence to higher-level behavioral interpretation.

Repository alignment
--------------------

The semantic schema is aligned with the four main modeling layers of the repository:

``Model_A``
   Produces operational-state predictions from synchronized analog and digital signals.

``Model_B``
   Extracts contiguous runs, active sequences, nominal sequence references, anomaly-oriented comparison metrics, and longitudinal divergence metrics.

``Model_C``
   Interprets behavioral sequences through semantically contextualized operating modes, working modes, incident families, and reliability-informed summaries.

``Model_D``
   Groups anomalous windows into incident episodes and checks consistency with incident registries.

In this sense, the semantic schema does not replace the analytical workflow. Rather, it provides a formalized contextual frame through which the outputs of ``Model_A``, ``Model_B``, ``Model_C``, and ``Model_D`` can be related to structural and functional interpretations.

Conceptual flow
---------------

The semantic organization of the workflow can be summarized as follows:

1. **Observations** are collected from sensors and control signals.
2. Observations provide evidence for inferred **operational states**.
3. Operational states are aggregated into **behavioral sequences**.
4. Behavioral sequences are compared against nominal baselines and summarized into **operating** and **working modes**.
5. Recurrent or non-nominal patterns are assigned **incident families**.
6. Temporally coherent anomalous windows are grouped into **incident episodes**.
7. Recovery windows and recurrence intervals support **life** and **recovery regime** interpretation.

This progressive mapping helps preserve the transition from raw observations to actionable longitudinal knowledge.

Example conceptual mapping
--------------------------

A simplified conceptual interpretation might look as follows:

- an electrical observation is linked to a monitored channel,
- the channel is associated with a component or subsystem,
- the observation contributes evidence for an inferred operational state,
- the state belongs to a behavioral sequence,
- the sequence is interpreted as an operating mode or working mode,
- the same sequence may receive a weak incident-family label,
- repeated anomalous windows may be grouped into an incident episode,
- and episode timing contributes recovery and recurrence interpretation.

This type of mapping is the basis for contextual interpretation in the paper and in the repository outputs.

Implementation status
---------------------

The current schema should be understood as a **lightweight ontology-aligned contextualization scheme**.

It already supports:

- traceable conceptual alignment between structural, functional, and informational views,
- explicit naming of the main behavioral and longitudinal entities and relations,
- semantic interpretation of sequence-level outputs,
- and extensibility toward richer reliability-informed monitoring layers.

At the same time, it remains limited in several respects:

- it is not yet a full domain ontology for industrial asset management,
- it does not yet implement a complete reasoning layer,
- and some mappings remain heuristic or workflow-oriented rather than fully axiomatized.

This scope is intentional. The purpose of the current implementation is to make the contextualization layer explicit and reproducible without overcomplicating the monitoring framework.

Future extensions
-----------------

Possible next steps include:

- richer OWL/Turtle serialization patterns for incident families and episodes,
- SHACL constraints for validation,
- SPARQL-based querying of contextualized longitudinal outputs,
- explicit linkage between paper tables and semantic artifacts,
- and deeper integration with asset-management or digital-twin environments.

Summary
-------

The semantic schema provides a lightweight but explicit formalization of the contextualization layer used in the industrial asset behavioral monitoring framework. It federates existing semantic resources for sensing, time, and quantities, and extends them with domain-specific concepts for operational states, behavioral sequences, incident families, episodes, and reliability-informed regimes.

Its main purpose is to support interpretability, traceability, and future semantic extensibility across the repository and the associated research workflow.
