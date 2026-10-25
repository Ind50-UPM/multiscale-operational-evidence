UML diagrams
============

Two kinds of diagrams are maintained. **Generated** class-inheritance diagrams
are rebuilt from the code on every documentation build (Graphviz, via
``sphinx.ext.inheritance_diagram``), so they cannot drift from the
implementation; each module page of the :doc:`api/index` also carries its own
diagram. **Hand-maintained** Mermaid diagrams describe components and run-time
interactions that cannot be inferred from inheritance; update them whenever a
public class gains or loses a collaborator.

Component view
--------------

.. mermaid::

   flowchart TB
     subgraph Inputs
       AN[analogicas_nonans.parquet<br/>Vrms, RP, Irms, PF per phase]
       DG[digitales.parquet<br/>PLC bits and controller state]
       PR[estados_nonans.parquet<br/>retained Model A predictions]
       REG[candidate_incident_registry_refined.csv<br/>weak references]
     end
     subgraph Model_A [iabm]
       DP[DataProcessor] --> MOD[RandomForest / XGBoost models]
     end
     subgraph Model_B [iabm_behavior]
       RUNS[run extraction] --> SEQ[active sequences and nominal comparison]
     end
     subgraph Model_C [iabm_semantics]
       SEM[semantic mode and family rules]
     end
     subgraph Model_D [iabm_incidents]
       BEH[BehavioralWindowEncoder]
       OBS[CommandResponseObserver]
       TC[TransientConcentration]
       SOLV[VQEDirectionSolver / NoisyVQEDirectionSolver]
       CF[ConformalEvidenceFusion]
       ABL[FourLevelAblationRunner]
       MS[MultiscaleSensitivityRunner]
       EP[IncidentEpisodeBuilder]
     end
     AN --> DP
     DG --> DP
     MOD --> RUNS
     SEQ --> SEM
     AN --> OBS
     DG --> OBS
     PR --> BEH
     OBS --> TC --> SOLV
     BEH --> ABL
     TC --> ABL
     ABL --> EP
     MS --> ABL
     CF --> EP
     REG --> ABL

Class view of the electrical and quantum evidence
-------------------------------------------------

.. mermaid::

   classDiagram
     class ElectricalConfig {
       +channels
       +state_column
       +minimum_samples
       +minimum_safe_fraction
       +shrinkage
     }
     class CommandResponseObserver {
       +profiles
       +fit(nominal)
       +residuals(samples)
     }
     class TransientReference {
       +center
       +whitening
       +scatter
       +scale
     }
     class TransientConcentration {
       +references
       +fit(nominal_residuals)
       +operator(x, reference)$
       +score_windows(residuals, windows)
     }
     class VQEDirectionSolver {
       +layers
       +sweeps
       +restarts
       +solve(operator)
       +solve_batch(operators)
       #_energy(psi, h)
     }
     class NoisyVQEDirectionSolver {
       +estimator
       #_energy(psi, h)
     }
     class ShotEstimator {
       +shots
       +noise
       +settings
       +expectation(states, hamiltonians)
     }
     class NoiseModel {
       +single_qubit_error
       +two_qubit_error
       +readout_error
       +mitigate
       +fidelity(layers)
     }
     CommandResponseObserver --> ElectricalConfig
     TransientConcentration --> ElectricalConfig
     TransientConcentration "1" o-- "*" TransientReference
     TransientConcentration --> VQEDirectionSolver : optional solver
     VQEDirectionSolver <|-- NoisyVQEDirectionSolver
     NoisyVQEDirectionSolver --> ShotEstimator
     ShotEstimator --> NoiseModel

Sequence of one extension run
-----------------------------

.. mermaid::

   sequenceDiagram
     participant U as User
     participant X as run_model_d_extensions
     participant P as primary / multiscale artifacts
     participant Q as quantum
     participant C as conformal
     participant I as injection
     U->>X: --stage all
     X->>P: hash all files (read-only guard)
     X->>X: model_a class metrics
     X->>Q: certificates, information profiles, shot and noisy-VQE study
     X->>I: residuals, second-order reference, injected windows
     X->>C: conformal fusion, alternative slots, baselines
     X->>P: re-hash and compare
     X-->>U: reports/extensions/*

Generated inheritance diagrams
------------------------------

Model D incident layer:

.. inheritance-diagram:: iabm_incidents.electrical iabm_incidents.quantum iabm_incidents.conformal iabm_incidents.injection iabm_incidents.ablation iabm_incidents.multiscale iabm_incidents.config
   :parts: 1

Model A state inference:

.. inheritance-diagram:: iabm.models iabm.data_processor
   :parts: 1

Model B and Model C:

.. inheritance-diagram:: iabm_behavior.sequences iabm_semantics.semantics
   :parts: 1
