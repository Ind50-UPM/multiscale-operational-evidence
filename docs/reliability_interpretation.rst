Reliability-Informed Interpretation
===================================

The semantic layer in the updated repository no longer stops at contextualized
operating or working modes. It now supports a reliability-informed reading of
behavioral traces.

From modes to life proxies
--------------------------

The central idea is that repeated sequence patterns can act as weak but useful
proxies for asset-life evolution. Instead of claiming direct physical failure
models, the repository exposes interpretable intermediate concepts such as:

- ``incident_family``
- ``reliability_interpretation``
- ``life_regime``
- ``recovery_regime``

These concepts bridge behavior analytics and maintenance-oriented reasoning while
remaining grounded in observed telemetry and sequence evolution.

Examples of interpretation
--------------------------

The current heuristic layer supports readings such as:

- ``pump_abrupt_failure`` -> ``high_severity_failure_proxy``
- ``process_saturation`` -> ``load_or_capacity_stress_proxy``
- ``float_recurrent_disturbance`` -> ``recurrent_control_disturbance_proxy``
- ``post_intervention_recovery`` -> ``recovery_stabilization_proxy``

Likewise, life regimes such as ``degradation_acceleration`` or
``recurrent_instability`` provide an interpretable vocabulary for discussing how
sequence evolution may reflect changing operating conditions.

Limits of the current layer
---------------------------

The repository does not yet implement a full reliability model. The current
layer should therefore be read as:

- a structured interpretation scaffold,
- a transparent weak-labeling mechanism,
- a bridge toward future family-specific occurrence modelling.

This keeps the system scientifically honest while still supporting practical
reasoning about incidents, recovery, and longitudinal monitoring.
