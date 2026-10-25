# Four-level Model D ablation assessment

This is a matched, chronologically calibrated retrospective ablation. M3 substitutes VQE for exact transient evidence; it does not add both coordinates.

Outcome provenance: weak derived signal references. Derived signal events are exploratory references, not independent physical-fault labels.

- M0: 2 episodes; 8 warning windows of 271; 2.95 warning hours per 100 observed decision hours; median episode span 6.25 h.
- M1: 2 episodes; 8 warning windows of 271; 2.95 warning hours per 100 observed decision hours; median episode span 6.25 h.
- M2: 2 episodes; 8 warning windows of 271; 2.95 warning hours per 100 observed decision hours; median episode span 6.25 h.
- M3: 2 episodes; 8 warning windows of 271; 2.95 warning hours per 100 observed decision hours; median episode span 6.25 h.

M2/M3 window decision agreement is 100.00%; M3-only warnings: 0; M2-only warnings: 0. A matching decision set does not establish identical eigenvectors or a quantum advantage.

Episode count and union-based warning exposure describe candidate workload. More warnings are not automatically better detection, and fewer warnings are not automatically fewer false alarms. Fragmentation against derived references is reported separately from physical-event fragmentation.

The original family classifier is held fixed. Family counts and raw indicator features describe evidence; additional electrical evidence does not independently validate family labels.

Episode onsets are window-based and retrospective. Detection availability is recorded at the end of the onset-confirming window; apparent lead from a lookback-window start is not online early warning. Stable recovery means threshold-confirmed return of the score, not confirmed physical repair.

Risk targets are identical fixed external reference events for every level, never each level's own episodes. Newly fitted stages use declared chronological partitions; the original Model A fit provenance must be audited separately when retained predictions are supplied. Training uses calibration-period predictors with a full-horizon purge; held-out labels are right-censored at the supplied study boundary. Weak-reference risk results measure prediction of that derived event process, not calibrated probability of physical failure.

## Fixed weak-reference risk assessment

- M0, 6 h: ROC AUC 0.6010; Brier 0.1668; training-prevalence Brier 0.1586.
- M1, 6 h: ROC AUC 0.5185; Brier 0.2816; training-prevalence Brier 0.1586.
- M2, 6 h: ROC AUC 0.5219; Brier 0.2547; training-prevalence Brier 0.1586.
- M3, 6 h: ROC AUC 0.5215; Brier 0.2539; training-prevalence Brier 0.1586.

Every level has a worse Brier score than the constant training-prevalence comparator. These results do not support useful calibrated risk prediction at this horizon.

Single-class horizons are marked in risk_metrics.csv. Perfect precision/recall on an all-positive target is not useful evidence of risk discrimination.

- maintenance overlap: unavailable: no temporally aligned intervention intervals.
- false warning burden: unavailable: no independently healthy intervals; total warning burden is reported.
- physical onset/recovery accuracy: unavailable: independent timestamped onset/recovery truth absent.
- physical risk performance: unavailable: only fixed weak-reference risk performance can be evaluated.

The ablation cannot establish superior physical-fault detection unless independent health labels and temporally aligned intervention/onset/recovery records support that conclusion.
