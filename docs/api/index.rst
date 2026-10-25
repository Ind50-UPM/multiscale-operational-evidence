API reference
=============

The reference below is generated systematically from the Google-style
docstrings of every module. Each module page lists its classes and functions,
links to one page per object, and shows a class-inheritance (UML) diagram
generated from the code with Graphviz. Narrative guides for each layer are in
:doc:`../packages/index`.

Packages
--------

.. autosummary::
   :toctree: generated
   :recursive:

   iabm
   iabm_behavior
   iabm_semantics
   iabm_incidents

Reproduction scripts
--------------------

The scripts in ``src/`` orchestrate the batch runs and the experiments reported
in the manuscript. They are documented with the same conventions; see
:doc:`../reproducibility` for the execution order.

.. autosummary::
   :toctree: generated

   concat_monthly_csv_to_parquet
   generate_model_bc_batch
   generate_candidate_incident_registry_batch
   generate_probe_incident_timeline_report
   generate_anticipatory_maintenance_report
   generate_model_d_final_report
   run_model_d_ablation
   run_model_d_multiscale
   run_model_d_extensions
   export_extension_figure_data
