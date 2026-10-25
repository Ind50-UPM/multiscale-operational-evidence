# Multiscale window-length sensitivity

This is one prespecified sensitivity at 15 min, 1 h, 6 h and 24 h. The primary 24-hour/hourly experiment is retained unchanged.

Every sensitivity scale uses the same 15-minute decision endpoints and the same cross-scale/M0–M3 support intersection. The sensitivity 24-hour arm is not a replay of the primary hourly-cadence arm.

Electrical observer, state whitening and fourth-order reference are frozen. Behavioral occupancy references, coordinate scaling and thresholds are estimated per scale on the original fit/reference/calibration partitions. The risk outcome-horizon purge boundary is fixed across scales.

- 15 min / M0: 271 matched decisions, 35 candidate episodes, 60.89 warning h per 100 decision h.
- 15 min / M1: 271 matched decisions, 32 candidate episodes, 57.93 warning h per 100 decision h.
- 15 min / M2: 271 matched decisions, 31 candidate episodes, 56.46 warning h per 100 decision h.
- 15 min / M3: 271 matched decisions, 31 candidate episodes, 56.46 warning h per 100 decision h.
- 60 min / M0: 271 matched decisions, 3 candidate episodes, 9.96 warning h per 100 decision h.
- 60 min / M1: 271 matched decisions, 3 candidate episodes, 9.96 warning h per 100 decision h.
- 60 min / M2: 271 matched decisions, 3 candidate episodes, 9.96 warning h per 100 decision h.
- 60 min / M3: 271 matched decisions, 3 candidate episodes, 9.96 warning h per 100 decision h.
- 360 min / M0: 271 matched decisions, 2 candidate episodes, 2.95 warning h per 100 decision h.
- 360 min / M1: 271 matched decisions, 2 candidate episodes, 2.95 warning h per 100 decision h.
- 360 min / M2: 271 matched decisions, 2 candidate episodes, 2.95 warning h per 100 decision h.
- 360 min / M3: 271 matched decisions, 2 candidate episodes, 2.95 warning h per 100 decision h.
- 1440 min / M0: 271 matched decisions, 45 candidate episodes, 66.79 warning h per 100 decision h.
- 1440 min / M1: 271 matched decisions, 47 candidate episodes, 69.74 warning h per 100 decision h.
- 1440 min / M2: 271 matched decisions, 48 candidate episodes, 71.96 warning h per 100 decision h.
- 1440 min / M3: 271 matched decisions, 48 candidate episodes, 71.96 warning h per 100 decision h.

## Executed comparison

The common cohort retains 271/2113 candidate test endpoints (12.83%). This coverage loss conditions every result.
M2 and M3 have identical warning decisions at every prespecified window length; no VQE-only warning or detection gain is observed.
Every six-hour weak-reference forecast has a worse Brier score than the constant training-prevalence comparator. Window sensitivity does not establish a calibrated forecasting benefit.

These are results conditional on common support. Short windows can lack enough retained Model A predictions or safe reference-supported vectors. Removing those endpoints from all scales also splits episode/recovery blocks; compare the availability audit before attributing fragmentation to faults.

Counts and durations at different scales are correlated candidate descriptions, not independent physical-failure counts. A shorter lookback can reduce apparent localization delay without proving earlier detection of a physical onset.

Reported maintenance and healthy/onset/recovery truth remain unavailable in the focused 2022 interval. False warnings are not inferred from unmatched weak records. Fixed derived-event risk targets remain exploratory, and single-class horizons are explicitly marked.

No held-out results select the window, threshold, circuit budget or risk model. Any claim of a preferred operational scale requires independent adjudication.
