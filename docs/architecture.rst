Architecture
============

This page summarizes the software architecture of the repository and the main
runtime relationships between the modelling layers.

System Architecture
-------------------

.. mermaid::

   flowchart TD
       A[Telemetry parquet datasets] --> B[Model_A<br/>State inference]
       B --> C[Model_B<br/>Runs and active sequences]
       C --> D[Model_C<br/>Semantic interpretation]
       C --> E[Candidate registry batch]
       D --> F[Model_D<br/>Episodes and occurrence summaries]
       E --> F
       F --> G[Evaluation, recovery and final reports]

Class Collaboration Snapshot
----------------------------

.. mermaid::

   classDiagram
       class IndustrialDataProcessor
       class StateClassifier
       class BehavioralSequenceAnalyzer
       class SemanticModeInterpreter
       class CandidateIncidentRegistryBuilder
       class IncidentEpisodeBuilder
       class IndicatorPipeline
       class EpisodeEvaluator

       IndustrialDataProcessor --> StateClassifier : features and labels
       StateClassifier --> BehavioralSequenceAnalyzer : predicted states
       BehavioralSequenceAnalyzer --> SemanticModeInterpreter : active sequences
       CandidateIncidentRegistryBuilder --> IncidentEpisodeBuilder : weak registry input
       SemanticModeInterpreter --> IncidentEpisodeBuilder : semantic assignments
       IndicatorPipeline --> IncidentEpisodeBuilder : window scores
       IncidentEpisodeBuilder --> EpisodeEvaluator : detected episodes

Execution Layers
----------------

- ``Model_A`` transforms analog and digital telemetry into operating-state predictions.
- ``Model_B`` converts those state timelines into contiguous runs, active sequences, and nominal comparison artifacts.
- ``Model_C`` maps the observed sequence behavior into semantic and reliability-oriented labels.
- ``Model_D`` combines semantic evidence and registry knowledge to build longitudinal episodes and occurrence summaries.
- Batch orchestration utilities make the workflow tractable for multi-month industrial campaigns.
