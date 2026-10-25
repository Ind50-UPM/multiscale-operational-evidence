Automation and Batch Runs
=========================

The repository includes dedicated batch utilities to process month-wise parquet
campaigns without forcing monolithic runs over the full historical archive.

Batch Utilities
---------------

.. autosummary::
   :nosignatures:

   generate_model_bc_batch

.. autosummary::
   :nosignatures:

   generate_candidate_incident_registry_batch

.. autosummary::
   :nosignatures:

   concat_monthly_csv_to_parquet

Operational Sequence
--------------------

.. mermaid::

   sequenceDiagram
       participant User
       participant BatchBC as Batch Model_B/Model_C
       participant BatchReg as Candidate Registry Batch
       participant ModelD as Model_D CLI

       User->>BatchBC: launch month-wise sequence and semantic generation
       BatchBC-->>User: consolidated active sequences and semantic assignments
       User->>BatchReg: launch candidate incident registry generation
       BatchReg-->>User: consolidated registry candidates and manifests
       User->>ModelD: run final longitudinal incident processing
       ModelD-->>User: episodes, occurrence summaries, evaluation artifacts

Batch Artefacts
---------------

- Monthly outputs preserve partial progress and simplify restart after interruptions.
- Manifest files expose which months were processed, empty, reused, or failed.
- Consolidated outputs provide the flat files consumed by downstream models and reporting layers.
