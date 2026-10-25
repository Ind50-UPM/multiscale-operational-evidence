"""Export compact CSV tables behind the manuscript's extension figures (pgfplots).

Purpose:
    Last step of the reproduction chain. It converts the (larger, long-format)
    result tables written by ``run_model_d_extensions.py`` into small
    wide-format CSV files that LaTeX/pgfplots can plot directly, and it
    recomputes one timeline (equal-weight versus conformal M2 evidence on the
    primary 24-hour test windows) from the frozen primary ablation artifacts.
    No model is refitted except the lightweight conformal calibration, which
    is deterministic given the stored indicators.

Command line (from the repository root, after ``run_model_d_extensions.py``)::

    PYTHONPATH=src:src/Model_D python src/export_extension_figure_data.py

``src`` is needed to import ``run_model_d_extensions`` (and, transitively,
``run_model_d_ablation``/``run_model_d_multiscale``); ``src/Model_D`` is
needed for ``iabm_incidents``. The script takes no command-line options and
all paths are relative to the current working directory.

Inputs:
    * ``reports/extensions/injection_summary.csv`` (telemetry stage): columns
      ``scenario``, ``statistic``, ``detection_power``. The ``scenario``
      label must follow ``<kind>|m=<float>|rate=<float>|len=<int>|phases=<digit>``.
    * ``reports/extensions/shot_noise_study.csv`` (quantum stage): columns
      ``shots_per_setting``, ``noise``, ``median_relative_score_error``.
    * ``reports/extensions/fusion_summary.csv`` (fusion stage): columns
      ``cadence``, ``scale``, ``method``,
      ``warning_hours_per_100_observed_hours``,
      ``weak_reference_event_overlap_fraction``.
    * Primary ablation artifacts loaded through
      :func:`run_model_d_extensions.load_scale` (``'primary'``):
      ``reports/four_level_ablation/2022_focused/raw_indicators.parquet``,
      ``ablation_metadata.json``, ``calibration.csv`` and ``support.csv``.

Outputs (all in ``reports/extensions/figure_data/``):
    * ``injection_dropouts.csv``: detection power versus dropout rate
      (single-sample dropouts), one column per statistic in :data:`STATS`.
    * ``injection_under_response.csv``: detection power versus
      under-response magnitude (in percent) when all three phases are
      affected.
    * ``shot_error.csv``: median relative score error (percent) versus shots
      per measurement setting, one column per noise model (the infinite-shot
      reference row is dropped).
    * ``shot_agreement_15min.csv``: mean agreement (percent) of 15-min
      transient-only decisions with the exact solver versus shots, one column
      per noise model (from ``shot_noise_T4_decisions_15min.csv``).
    * ``timeline_ratio.csv``: per test window, days since 2022-08-19 (the
      calibration/test boundary) and the equal-weight and conformal M2
      scores divided by their respective onset thresholds; NaN rows break
      the line at coverage gaps.
    * ``multiscale_calibration.csv``: per window length (15-minute cadence),
      warning burden and weak-reference overlap for each fusion method,
      plus an integer ``position`` column for evenly spaced x-axes.

Script dependencies:
    ``run_model_d_ablation.py`` -> ``run_model_d_multiscale.py`` ->
    ``run_model_d_extensions.py --stage all`` (at least the ``quantum``,
    ``telemetry`` and ``fusion`` stages) -> this script.

.. mermaid::

    flowchart LR
        P[reports/four_level_ablation/2022_focused] --> X[export_extension_figure_data.py]
        I[reports/extensions/injection_summary.csv] --> X
        S[reports/extensions/shot_noise_study.csv] --> X
        F[reports/extensions/fusion_summary.csv] --> X
        X --> D[reports/extensions/figure_data/*.csv]
        D --> L[LaTeX / pgfplots figures]

Maintenance notes:
    The date ``2022-08-19`` used as timeline origin duplicates
    ``BOUNDARIES['calibration_end']`` of ``run_model_d_ablation.py``; keep
    both in sync if the study partitions change. The column names of
    ``multiscale_calibration.csv`` are derived from the first underscore token
    of each metric (``warning_*`` and ``weak_*``) followed by the method
    name; renaming metrics upstream silently renames these columns. The
    script performs no telemetry processing or VQE, so it is expected to be
    short-running.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import run_model_d_extensions as ext
from iabm_incidents.config import ModelDConfig
from iabm_incidents.conformal import ConformalEvidenceFusion

#: Output folder for the pgfplots tables (relative to the repository root).
OUT = Path('reports/extensions/figure_data')
#: Statistics kept (and column order) in the fault-injection tables.
STATS = ['mean_residual_norm', 'mean_shift', 'second_order', 'fourth_order']


def main():
    """Build every figure-data CSV in :data:`OUT`.

    The function reads the extension result tables, reshapes them into wide
    format, recomputes the primary equal-weight/conformal M2 test timeline
    and finally prints the row count of each CSV found in :data:`OUT`.

    Raises:
        FileNotFoundError: If one of the extension tables or primary
            ablation artifacts listed in the module docstring is missing.
        KeyError: If an expected statistic, level (``M2``) or method column
            is absent from the inputs.

    Notes:
        Outputs are overwritten on every run. The conformal fusion is fitted
        on the calibration windows inside common support and applied to the
        test windows inside common support, exactly as the
        ``conformal_ACAT_M2`` arm of ``run_model_d_extensions.stage_fusion``.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    # Note: the scenario label encodes the injection parameters; split it into
    # typed columns (kind, m, rate, len, ph) for filtering and pivoting.
    inj = pd.read_csv('reports/extensions/injection_summary.csv')
    parts = inj.scenario.str.extract(r'(?P<kind>\w+)\|m=(?P<m>[\d.]+)\|rate=(?P<rate>[\d.]+)\|len=(?P<len>[\d.]+)s?\|phases=(?P<ph>\d)')
    inj = pd.concat([inj, parts.astype({'m': float, 'rate': float, 'len': float, 'ph': int})], axis=1)
    pivot = lambda q, key: inj[q].pivot_table(index=key, columns='statistic', values='detection_power')[STATS].reset_index()
    pivot((inj.kind == 'dropouts') & (inj.len == 1), 'rate').to_csv(OUT/'injection_dropouts.csv', index=False)
    under = pivot((inj.kind == 'under_response') & (inj.ph == 3), 'm'); under['m'] *= 100
    under.to_csv(OUT/'injection_under_response.csv', index=False)
    shots = pd.read_csv('reports/extensions/shot_noise_study.csv')
    # Why: the exact-readout reference is stored with shots_per_setting=inf and
    # cannot be placed on a logarithmic shots axis.
    shots = shots[np.isfinite(shots.shots_per_setting)]
    shots.pivot(index='shots_per_setting', columns='noise', values='median_relative_score_error').mul(100).reset_index().to_csv(OUT/'shot_error.csv', index=False)
    # Note: the 15-min transient-only study comes from the robustness stage.
    agreement = pd.read_csv('reports/extensions/shot_noise_T4_decisions_15min.csv')
    agreement = agreement[np.isfinite(agreement.shots_per_setting)]
    agreement.pivot(index='shots_per_setting', columns='noise', values='mean_agreement').mul(100).reset_index().to_csv(OUT/'shot_agreement_15min.csv', index=False)
    frame, meta, calibration, common = ext.load_scale('primary')
    e = ext.excess(frame, meta, ext.NINE)
    cal, test = common & frame.phase.eq('calibration').to_numpy(), common & frame.phase.eq('test').to_numpy()
    fusion = ConformalEvidenceFusion(ext.NINE).fit(e[cal])
    p = fusion.transform(e[test])
    tau = calibration.loc['M2', 'onset_threshold']
    # Invariant: origin equals BOUNDARIES['calibration_end'] (first test day).
    start = pd.Timestamp('2022-08-19')
    timeline = pd.DataFrame(dict(day=(frame.end_time[test]-start).dt.total_seconds().to_numpy()/86400,
        equal_weight=e[test].sum(axis=1).to_numpy()/tau, conformal=p.fused_score.to_numpy()/fusion.thresholds[0]))
    # Blank rows break the plotted line across unsupported (excluded) windows.
    gap = timeline.day.diff().gt(1.5/24)
    blank = pd.DataFrame(dict(day=timeline.day[gap].to_numpy()-1/48, equal_weight=np.nan, conformal=np.nan))
    timeline = pd.concat([timeline, blank]).sort_values('day')
    timeline.to_csv(OUT/'timeline_ratio.csv', index=False, na_rep='nan')
    # Note: only the 15-minute-cadence fixed-scale rows are tabulated; the
    # 'adaptive' multiscale row has a non-integer scale label.
    fused = pd.read_csv('reports/extensions/fusion_summary.csv')
    multi = fused[fused.cadence.eq('15min') & fused.scale.ne('adaptive')].copy()
    multi['scale'] = multi.scale.astype(int)
    table = multi.pivot(index='scale', columns='method', values=['warning_hours_per_100_observed_hours', 'weak_reference_event_overlap_fraction'])
    table.columns = [f'{a.split("_")[0]}_{b}' for a, b in table.columns]
    table = table.reset_index(); table['position'] = range(len(table))
    table.to_csv(OUT/'multiscale_calibration.csv', index=False)
    print(*(f'{path.name}: {len(pd.read_csv(path))} rows' for path in sorted(OUT.glob('*.csv'))), sep='\n')


if __name__ == '__main__':
    main()
