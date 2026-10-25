Multi-Scale Operational Evidence
================================

**Multi-Scale Operational Evidence** turns low-intrusion industrial telemetry
(three-phase electrical measurements plus PLC digital states) into auditable,
calibrated incident evidence for weakly instrumented assets. It accompanies the
manuscript *Calibrated Multi-Scale Incident Evidence from Low-Intrusion
Electrical Telemetry with Certified Fourth-Order Transient Analysis* (submitted to *Results in Engineering*).

The framework is organised as a hierarchy of four installable packages:

.. mermaid::

   flowchart LR
     X[Synchronised telemetry X_t] --> A["Model_A · iabm<br/>operational states S_t"]
     A --> B["Model_B · iabm_behavior<br/>duration-aware sequences Q_k"]
     B --> C["Model_C · iabm_semantics<br/>contextual modes M_k"]
     C --> D["Model_D · iabm_incidents<br/>evidence z_w → score A_w → episodes E_r"]
     X -. command-conditioned residuals .-> D

Highlights of the current release:

- **Fourth-order transient evidence** read as a FOBI kurtosis matrix and a
  three-qubit residual density operator (magnitude/shape decomposition).
- **Certified variational evaluation**: rigorous a-posteriori bounds
  (Kato--Temple with a measurable Frobenius gap, Frobenius cap) either
  certify each VQE decision or route it to the exact solver, with a
  finite-shot, depolarising and readout-error resource model.
- **Conformal Cauchy-combination fusion** of evidence dimensions and
  integration scales, with per-warning attribution.
- **Semi-synthetic fault injection** through the frozen observer, giving
  labelled ground truth for comparing first-, second- and fourth-order evidence.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting_started
   overview
   architecture

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   incident_taxonomy
   semantic_schema
   reliability_interpretation
   event_frequency_interpretation

.. toctree::
   :maxdepth: 2
   :caption: Package guides

   packages/index

.. toctree::
   :maxdepth: 2
   :caption: Experiments

   longitudinal_validation
   model_d_ablation
   model_d_ablation_results
   model_d_multiscale
   model_d_multiscale_results
   model_d_extensions
   model_d_extensions_results

.. toctree::
   :maxdepth: 2
   :caption: Reproducibility and design

   reproducibility
   automation
   paper_map
   uml
   development
   known_issues

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index
