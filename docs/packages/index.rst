Package guides
==============

Each layer of the hierarchy is an independently installable Python package
under ``src/Model_*``. The guides below summarise the role of each package and
link to the module pages of the :doc:`../api/index`; the package README files
in the repository give installation and command-line details.

.. list-table::
   :header-rows: 1
   :widths: 14 18 68

   * - Layer
     - Package
     - Responsibility
   * - Model_A
     - ``iabm``
     - Supervised operational-state inference from analogue and digital signals (Random Forest / XGBoost).
   * - Model_B
     - ``iabm_behavior``
     - State runs, duration-aware sequences, nominal references and longitudinal divergence metrics.
   * - Model_C
     - ``iabm_semantics``
     - Contextualised operating and working modes, incident-family and recovery semantics.
   * - Model_D
     - ``iabm_incidents``
     - Evidence vectors, command-conditioned electrical residuals, fourth-order operator, exact/VQE solvers, conformal fusion, episodes, ablation, multiscale sensitivity and fault injection.

.. toctree::
   :maxdepth: 1

   model_a
   model_b
   model_c
   model_d
