# Data

The raw telemetry of the monitored wheel-washing facility (Port of Gijón,
2022–2026) is openly distributed in this folder through **Git LFS**
(≈1.8 GB). Run `git lfs install` before cloning, or `git lfs pull` after
cloning, to download the actual files instead of LFS pointers:

| File | Rows (approx.) | Content | Key columns |
|---|---|---|---|
| `analogicas_nonans.parquet` | 33.2 M | 1-s electrical measurements after NaN treatment | `0` (UTC timestamp), `Vrms1–4`, `RP1–4` (active power per phase), `Irms1–4`, `PF1–4` |
| `digitales.parquet` | 52.5 M | PLC digital signals | `0` (UTC timestamp), `6`–`11` (bits), `estado` (controller state code) |
| `estados_nonans.parquet` | 16.1 M | Retained Model A predictions on non-idle samples | `source_month`, `date`, electrical columns, `estado` (reference), `pred_estado` (prediction) |
| `model_b_state_timeline.parquet` | 16.1 M | Predicted state timeline used by Model B | `Time`, `Predicted_State`, `source_month` |
| `secuencias.parquet` | 2.2 M | Duration-aware behavioural sequences | `source_month`, `date`, `Runs`, `Values`, `span` |
| `palabras.parquet` | – | Sequence-word inventory | – |

Conventions:

* Timestamps are UTC; analysis windows are half-open `[start, end)`.
* Controller states are integer codes formed from the PLC activation bits
  (e.g. `0` idle, `32` water-tank pump, `12` feedback pumps 1 and 2).
* Digital commands are aligned to analogue samples causally (as-of backward
  join, two-second tolerance) in `src/run_model_d_ablation.py`.

All derived artifacts needed to reproduce the manuscript tables are also
versioned (Git LFS) under `reports/` and `src/predictions/Model_D/`, so most
results can be regenerated without the raw files; only the primary
indicators, the fault-injection benchmark and the Model A class-level
diagnostics need them.
