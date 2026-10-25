# Changelog

All notable changes to this project are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## 1.0.1 – pandas 3 compatibility

- `BehavioralWindowEncoder.encode` copies the gap mask before updating it in
  place, so it no longer fails under pandas copy-on-write (`ValueError: output
  array is read-only`). The values are unchanged, so no reported result is
  affected; `test_behavioral_partial_nominal_support_and_gap_mask` now passes.

## 1.0.0 – Initial public release

First release of **Multi-Scale Operational Evidence**, published with a fresh
history. It accompanies the *Results in Engineering* manuscript *Calibrated Multi-Scale Incident Evidence from Low-Intrusion Electrical Telemetry with Certified Fourth-Order Transient Analysis*.

### Packages
- `iabm` (Model A): Random Forest / XGBoost operational-state inference from
  analogue and digital signals, with persistence, CLI and i18n messages.
- `iabm_behavior` (Model B): state runs, duration-aware active sequences,
  nominal references and divergence/recovery metrics.
- `iabm_semantics` (Model C): contextual operating/working modes and
  incident-family semantics.
- `iabm_incidents` (Model D): seven behavioural indicators, command-conditioned
  electrical residuals, the fourth-order transient operator with exact and
  three-qubit VQE evaluation, episode segmentation with hysteresis, taxonomy
  interpretation, registry evaluation, exposure-aware occurrence summaries and
  exploratory future-episode analysis.

### Experiments
- Matched M0–M3 evidence ablation on the 2022 focused study.
- Prespecified 15 min / 1 h / 6 h / 24 h multiscale sensitivity.
- `iabm_incidents.quantum`: FOBI / density-operator reading of the transient
  operator (kurtosis ratio, quantum Jensen–Shannon divergence, trace distance),
  rigorous Kato–Temple certificates with a measurable Frobenius gap,
  qubit-wise-commuting Pauli measurement, finite-shot, depolarising and
  readout-error estimation, and a shot-based noisy VQE.
- `iabm_incidents.conformal`: smoothed conformal p-values and Cauchy
  combination fusion of evidence dimensions and scales with attribution.
- `iabm_incidents.injection`: second-order operator and semi-synthetic fault
  injection (under-response, power dropouts, bursts) through the frozen observer.
- Class-level Model A diagnostics, Hotelling T² and Isolation Forest baselines.

### Data
- Raw 2022–2026 telemetry (`data/*.parquet`) and all derived artifacts
  (incident registry, ablation, multiscale and extension outputs) are
  distributed with Git LFS.

### Documentation
- Google-style docstrings across all packages and scripts, Sphinx site with a
  recursive API reference, generated inheritance (UML) diagrams, Mermaid
  component/class/sequence diagrams, and a manuscript-to-code map.

