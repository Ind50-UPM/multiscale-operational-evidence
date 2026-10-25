"""One prespecified window-length sensitivity of the matched M0--M3 experiment.

The primary hourly-updated 24-hour study is untouched. This supplemental study
uses one 15-minute decision grid, the same frozen electrical models and outcome
records, and matched support across four window lengths. No best scale is chosen.

Role in the pipeline:
    The module wraps :class:`~iabm_incidents.ablation.FourLevelAblationRunner`
    (M0 behavioural, M1 + command residual, M2 + exact Tconc, M3 VQE Tconc)
    and executes it once per prespecified window length (15 min, 1 h, 6 h,
    24 h). It is driven by ``src/run_model_d_multiscale.py`` and its results
    are exported to ``reports/four_level_ablation/multiscale_sensitivity``.

Main public objects:
    * :class:`MultiscaleSensitivityConfig` - prespecified lengths, cadence and
      fixed risk purge boundary.
    * :class:`MultiscaleSensitivityRunner` - validation, common-support
      matching and per-scale ablation.
    * :class:`MultiscaleSensitivityResult` - per-scale results, stacked
      summaries, availability audit, Markdown/HTML report and export.

.. mermaid::

   flowchart TD
       IN["indicators_by_scale {15, 60, 360, 1440}"] --> V[validate fields, durations, unique endpoints, 15-min grid]
       V --> K[identical asset/end_time/phase keys across scales]
       K --> S[per-scale support: finite 10 columns, coverage, electrical_status, phase]
       S --> C[common support = AND over scales]
       C --> R[retrospective cohort: events inside common shortest test windows]
       C --> X[mark unsupported endpoints excluded_cross_scale_support]
       R --> A[FourLevelAblationRunner per scale]
       X --> A
       A --> OUT[MultiscaleSensitivityResult]

Data contract (per-scale input frames):
    Required columns: ``asset_id``, ``window_id``, ``start_time``,
    ``end_time``, ``phase`` (``reference``/``calibration``/``test``; any other
    value is unsupported), the seven behavioural columns,
    ``command_response_deviation``, ``transient_concentration`` and
    ``transient_concentration_vqe``. Optional: ``data_coverage`` (compared
    with ``ModelDConfig.window.min_coverage``) and ``electrical_status``
    (must equal ``"supported"``).

Notes:
    Let :math:`\\mathcal{S}_m` be the supported endpoints of scale :math:`m`.
    Every scale is evaluated on the common cohort
    :math:`\\mathcal{C} = \\bigcap_m \\mathcal{S}_m` of decision endpoints, so
    labels, train/test membership and exposure denominators are identical
    across scales. Endpoints outside :math:`\\mathcal{C}` are kept in the
    frame but flagged ``electrical_status = "excluded_cross_scale_support"``.

Maintenance notes:
    * **Leakage guards:** no held-out data selects the window, threshold,
      circuit budget or risk model; the risk holdout start is fixed across
      scales; the retrospective cohort is fixed from the shortest windows and
      reused at every scale; the full registry is passed unchanged as
      future-target registry.
    * Behavioural references, coordinate scaling and thresholds are refitted
      per scale by the ablation runner on the original partitions; the
      electrical observer, whitening and fourth-order reference are frozen
      upstream.
    * The retrospective-eligibility loop writes ``eligible[records.index]``
      and therefore assumes ``_normalize_incidents`` returns a frame with a
      default ``RangeIndex`` (true when the supplied registry has one).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from html import escape
import json

import numpy as np
import pandas as pd

from .ablation import (AblationConfig, FourLevelAblationRunner, FourLevelAblationResult,
                       BEHAVIORAL_COLUMNS, COMMAND_COLUMN, EXACT_COLUMN, VQE_COLUMN, _normalize_incidents)
from .config import ModelDConfig


@dataclass(frozen=True)
class MultiscaleSensitivityConfig:
    """Prespecified lengths, constant decision cadence and fixed risk purge time.

    All durations are integer minutes. The default decision cadence is short
    enough to observe contiguous 15-minute windows. Shared endpoint selection
    and phase membership must be assigned by the caller before evaluation.
    Threshold quantiles, VQE budgets and onset/recovery counts are not tuned
    independently by scale; nominal score distributions are calibrated per scale.

    Attributes:
        windows_minutes: Unique positive integer window lengths in minutes
            (default ``(15, 60, 360, 1440)``). Iteration order is preserved
            in the outputs.
        decision_step_minutes: Common decision cadence in minutes; must be a
            positive integer not larger than the shortest window so that the
            shortest windows tile the timeline without gaps (default 15).
        risk_holdout_start: Fixed boundary passed to every ablation run as
            ``risk_holdout_start`` (default 2022-08-19).

    Raises:
        ValueError: On empty or duplicated lengths, non-positive or
            non-integer lengths, or an invalid decision cadence.
    """

    windows_minutes: tuple[int, ...] = (15, 60, 360, 1440)
    decision_step_minutes: int = 15
    risk_holdout_start: pd.Timestamp = pd.Timestamp('2022-08-19')

    def __post_init__(self) -> None:
        """Validate the prespecified protocol.

        Raises:
            ValueError: See the class docstring.
        """
        if not self.windows_minutes or len(set(self.windows_minutes)) != len(self.windows_minutes):
            raise ValueError('Unique nonempty window lengths required')
        if any(not isinstance(value, int) or value <= 0 for value in self.windows_minutes):
            raise ValueError('Positive integer window lengths required')
        if not isinstance(self.decision_step_minutes, int) or not 0 < self.decision_step_minutes <= min(self.windows_minutes):
            raise ValueError('Decision cadence must not leave gaps in the shortest window')


@dataclass(frozen=True)
class MultiscaleSensitivityResult:
    """One sensitivity report plus per-scale four-level artifacts and support audit.

    Attributes:
        scales: ``window_minutes -> FourLevelAblationResult``.
        summary: Concatenated per-level summaries with ``window_minutes``.
        pairwise: Concatenated pairwise level comparisons (``left``,
            ``right``, ``decision_agreement``, ...) with ``window_minutes``.
        risk_metrics: Concatenated risk metrics with ``level`` and
            ``window_minutes``.
        availability: One row per scale and endpoint with ``own_support``,
            ``cross_scale_support`` and comma-separated ``missing_dimensions``.
        metadata: Protocol description, common endpoint counts per phase,
            cohort sizes and outcome provenance (JSON-serialisable).
    """

    scales: dict[int, FourLevelAblationResult]
    summary: pd.DataFrame
    pairwise: pd.DataFrame
    risk_metrics: pd.DataFrame
    availability: pd.DataFrame
    metadata: dict = field(default_factory=dict)

    def assessment(self) -> str:
        """Summarize conditional results without choosing a retrospectively best scale.

        Returns:
            str: English Markdown report: protocol statement, per-scale/per-level
            decision counts, episode counts and warning burden, common-cohort
            coverage, M2-versus-M3 decision agreement, six-hour Brier comparison
            with the prevalence comparator, and interpretation caveats.

        Raises:
            KeyError: If ``metadata`` lacks ``common_phase_endpoints``.
            IndexError: If ``availability`` has no ``test`` rows.
        """
        lines = ['# Multiscale window-length sensitivity', '',
            'This is one prespecified sensitivity at 15 min, 1 h, 6 h and 24 h. '
            'The primary 24-hour/hourly experiment is retained unchanged.', '',
            'Every sensitivity scale uses the same 15-minute decision endpoints and '
            'the same cross-scale/M0–M3 support intersection. The sensitivity 24-hour '
            'arm is not a replay of the primary hourly-cadence arm.', '',
            'Electrical observer, state whitening and fourth-order reference are frozen. '
            'Behavioral occupancy references, coordinate scaling and thresholds are '
            'estimated per scale on the original fit/reference/calibration partitions. '
            'The risk outcome-horizon purge boundary is fixed across scales.', '']
        for minutes, result in self.scales.items():
            for row in result.summary.itertuples():
                lines.append(f'- {minutes} min / {row.level}: {row.test_windows} matched decisions, '
                    f'{row.episode_count} candidate episodes, '
                    f'{row.warning_hours_per_100_observed_hours:.2f} warning h per 100 decision h.')
        lines.extend(['', '## Executed comparison', ''])
        test_rows = self.availability[self.availability.phase.eq('test')]
        candidate_count = int(test_rows.groupby('window_minutes').size().iloc[0])
        common_count = self.metadata['common_phase_endpoints']['test']
        lines.append(f'The common cohort retains {common_count}/{candidate_count} candidate test endpoints '
                     f'({100*common_count/candidate_count:.2f}%). This coverage loss conditions every result.')
        exact_vqe = self.pairwise.query("left == 'M2' and right == 'M3'")
        if exact_vqe.decision_agreement.eq(1).all():
            lines.append('M2 and M3 have identical warning decisions at every prespecified window length; '
                         'no VQE-only warning or detection gain is observed.')
        risk6 = self.risk_metrics[self.risk_metrics.horizon_hours.eq(6)]
        if not risk6.empty and 'brier_score' in risk6 and (risk6.brier_score > risk6.prevalence_baseline_brier).all():
            lines.append('Every six-hour weak-reference forecast has a worse Brier score than the '
                         'constant training-prevalence comparator. Window sensitivity does not '
                         'establish a calibrated forecasting benefit.')
        lines.extend(['',
            'These are results conditional on common support. Short windows can lack '
            'enough retained Model A predictions or safe reference-supported vectors. '
            'Removing those endpoints from all scales also splits episode/recovery blocks; '
            'compare the availability audit before attributing fragmentation to faults.', '',
            'Counts and durations at different scales are correlated candidate descriptions, '
            'not independent physical-failure counts. A shorter lookback can reduce apparent '
            'localization delay without proving earlier detection of a physical onset.', '',
            'Reported maintenance and healthy/onset/recovery truth remain unavailable '
            'in the focused 2022 interval. False warnings are not inferred from unmatched '
            'weak records. Fixed derived-event risk targets remain exploratory, and '
            'single-class horizons are explicitly marked.', '',
            'No held-out results select the window, threshold, circuit budget or risk model. '
            'Any claim of a preferred operational scale requires independent adjudication.', ''])
        return '\n'.join(lines)

    def export(self, output_dir: str | Path) -> dict[str, Path]:
        """Export a single English sensitivity report and its scale-specific audits.

        Args:
            output_dir: Target directory, created if needed.

        Returns:
            dict[str, pathlib.Path]: Written files: ``summary.csv``,
            ``pairwise.csv``, ``risk_metrics.csv``, ``availability.csv``,
            per-scale ablation exports under ``window_<m>min/`` (keys
            ``<m>min_<name>``), ``sensitivity_metadata.json``,
            ``assessment.md`` and ``sensitivity_report.html``.

        Raises:
            ValueError: If ``metadata`` contains NaN or infinite floats
                (``allow_nan=False``).
        """
        output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name in ('summary', 'pairwise', 'risk_metrics', 'availability'):
            path = output/f'{name}.csv'; getattr(self, name).to_csv(path, index=False); paths[name] = path
        for minutes, result in self.scales.items():
            for name, path in result.export(output/f'window_{minutes}min').items():
                paths[f'{minutes}min_{name}'] = path
        path = output/'sensitivity_metadata.json'
        path.write_text(json.dumps(self.metadata, indent=2, default=str, allow_nan=False), encoding='utf-8'); paths['metadata'] = path
        path = output/'assessment.md'; path.write_text(self.assessment(), encoding='utf-8'); paths['assessment'] = path
        html = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>Multiscale Model D sensitivity</title>",
            "<style>body{font-family:sans-serif;margin:24px}pre{white-space:pre-wrap}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px}</style>",
            '<h1>One multiscale window-length sensitivity</h1>', '<pre>'+escape(self.assessment())+'</pre>']
        for name in ('summary', 'pairwise', 'risk_metrics'):
            html.extend([f'<h2>{name.replace("_", " ").title()}</h2>', getattr(self, name).to_html(index=False)])
        html.append('<h2>Availability by phase and scale</h2>')
        stats = self.availability.groupby(['window_minutes','phase']).agg(
            candidate_endpoints=('own_support','size'), own_supported=('own_support','sum'),
            cross_scale_supported=('cross_scale_support','sum')).reset_index()
        html.append(stats.to_html(index=False))
        path = output/'sensitivity_report.html'; path.write_text(''.join(html)+'</html>', encoding='utf-8'); paths['html'] = path
        return paths


class MultiscaleSensitivityRunner:
    """Compare four window lengths on identical phase-labelled decision endpoints.

    Raw frames must contain identical asset/end_time/phase keys, scale-appropriate
    start times, all nine dimensions plus VQE, and coverage/safety evidence.
    Match support before fitting any score or evaluating episodes/risk. This
    preserves labels, train/test rows and exposure denominators across scales.

    Args:
        config: Prespecified multiscale protocol.
        ablation_config: Decision policy shared by every scale.
        model_d_config: Base Model_D configuration. Its window ``length`` and
            ``step`` are replaced per scale; ``window.min_coverage`` is the
            coverage support threshold.

    Examples:
        Typical driver usage (see ``src/run_model_d_multiscale.py``)::

            runner = MultiscaleSensitivityRunner()
            result = runner.run(indicators_by_scale=frames, incidents=registry)
            result.export("reports/four_level_ablation/multiscale_sensitivity")
    """

    def __init__(self, config: MultiscaleSensitivityConfig = MultiscaleSensitivityConfig(), *,
                 ablation_config: AblationConfig = AblationConfig(),
                 model_d_config: ModelDConfig | None = None) -> None:
        """Store the protocol and configurations.

        Args:
            config: Multiscale protocol.
            ablation_config: Four-level decision policy.
            model_d_config: Optional base configuration (defaults to
                :class:`~iabm_incidents.config.ModelDConfig`).
        """
        self.config, self.ablation_config = config, ablation_config
        self.model_config = model_d_config or ModelDConfig()

    def run(self, *, indicators_by_scale: dict[int, pd.DataFrame],
            incidents: pd.DataFrame | None = None, independent_incidents: bool = False,
            study_end: pd.Timestamp | None = None) -> MultiscaleSensitivityResult:
        """Run the single sensitivity; reject mixed grids or insufficient common support.

        Args:
            indicators_by_scale: ``window_minutes -> indicator frame``; keys
                must equal ``config.windows_minutes`` exactly.
            incidents: Outcome registry (``incident_id``, ``asset_id``,
                ``event_time``, ...). Its complete content is passed to every
                ablation run as future-target registry.
            independent_incidents: Whether the registry is independent truth
                (``True``) or a weak derived reference (``False``).
            study_end: Optional study end, forwarded unchanged to
                :meth:`~iabm_incidents.ablation.FourLevelAblationRunner.run`.

        Returns:
            MultiscaleSensitivityResult: Per-scale and stacked results.

        Raises:
            ValueError: If scale keys differ from the protocol, required
                fields are missing, a window duration differs from its scale,
                an ``(asset_id, end_time)`` endpoint is duplicated, a scale is
                not on the complete decision grid, endpoint/phase keys differ
                between scales, or the incident registry has duplicated
                ``(asset_id, incident_id)`` pairs. Insufficient common phase
                support and other ablation contract violations are raised
                (``ValueError``) by the ablation runner.

        Notes:
            The retrospective cohort keeps an event at time :math:`t` when a
            common-supported shortest *test* window satisfies
            :math:`s \\le t < e` (the first window whose end exceeds
            :math:`t` is checked).
        """
        if set(indicators_by_scale) != set(self.config.windows_minutes):
            raise ValueError('Supply exactly the prespecified window lengths')
        columns = [*BEHAVIORAL_COLUMNS, COMMAND_COLUMN, EXACT_COLUMN, VQE_COLUMN]
        frames, support_masks, keys = {}, {}, None
        available = []
        step = pd.Timedelta(minutes=self.config.decision_step_minutes)
        for minutes in self.config.windows_minutes:
            frame = indicators_by_scale[minutes].copy()
            required = {'asset_id','window_id','start_time','end_time','phase',*columns}
            if not required.issubset(frame):
                raise ValueError(f'Missing multiscale fields at {minutes} min: {sorted(required-set(frame))}')
            for column in ('start_time','end_time'):
                frame[column] = pd.to_datetime(frame[column], utc=True, format='mixed').dt.tz_convert(None)
            if not frame.end_time.sub(frame.start_time).eq(pd.Timedelta(minutes=minutes)).all():
                raise ValueError('Window duration disagrees with scale')
            if frame.duplicated(['asset_id','end_time']).any():
                raise ValueError('Duplicate decision endpoint')
            frame = frame.sort_values(['asset_id','end_time']).reset_index(drop=True)
            for _, group in frame.groupby('asset_id'):
                if not group.end_time.diff().dropna().eq(step).all():
                    raise ValueError('Every scale must use the same complete decision grid')
            current_keys = pd.MultiIndex.from_frame(frame[['asset_id','end_time','phase']])
            if keys is not None and not keys.equals(current_keys):
                raise ValueError('Scale endpoints and phase membership must match')
            keys = current_keys
            numeric = frame[columns].apply(pd.to_numeric, errors='coerce')
            support = np.isfinite(numeric).all(axis=1)
            if 'data_coverage' in frame:
                support &= frame.data_coverage.ge(self.model_config.window.min_coverage)
            if 'electrical_status' in frame:
                support &= frame.electrical_status.eq('supported')
            support &= frame.phase.isin(['reference','calibration','test'])
            frames[minutes], support_masks[minutes] = frame, support
        common = np.logical_and.reduce([mask.to_numpy() for mask in support_masks.values()])
        # Fix retrospective matching eligibility on the shortest observed windows.
        # Full future outcomes remain unchanged for every forecasting horizon.
        evaluation = _normalize_incidents(incidents)
        # Invariant: ``evaluation`` must have a 0..n-1 index, because its
        # labels are used as positions into ``eligible`` below.
        eligible = np.zeros(len(evaluation), bool)
        shortest = frames[min(self.config.windows_minutes)]
        observed = shortest.loc[common & shortest.phase.eq('test')]
        for asset, records in evaluation.groupby('asset_id'):
            windows = observed[observed.asset_id == asset]
            if windows.empty:
                continue
            times = records.event_time.to_numpy(dtype='datetime64[ns]')
            ends = windows.end_time.to_numpy(dtype='datetime64[ns]')
            starts = windows.start_time.to_numpy(dtype='datetime64[ns]')
            positions = np.searchsorted(ends,times,side='right')
            contained = positions < len(ends)
            contained &= starts[np.minimum(positions,len(starts)-1)] <= times
            eligible[records.index] = contained
        evaluation = evaluation.loc[eligible].copy()
        scales, summaries, pairs, risk = {}, [], [], []
        for minutes, frame in frames.items():
            audit = frame[['asset_id','window_id','start_time','end_time','phase']].copy()
            audit['window_minutes'] = minutes
            audit['own_support'], audit['cross_scale_support'] = support_masks[minutes], common
            numeric = frame[columns].apply(pd.to_numeric, errors='coerce')
            audit['missing_dimensions'] = numeric.apply(lambda row: ','.join(row.index[~np.isfinite(row)]), axis=1)
            available.append(audit)
            frame['electrical_status'] = frame.get('electrical_status', pd.Series('supported',index=frame.index))
            # Why: unsupported endpoints stay in the frame (availability audit)
            # but the ablation runner only scores ``electrical_status ==
            # "supported"`` rows, so this flag enforces the common cohort.
            frame.loc[~common,'electrical_status'] = 'excluded_cross_scale_support'
            config = replace(self.model_config, window=replace(self.model_config.window,
                length=pd.Timedelta(minutes=minutes), step=step))
            result = FourLevelAblationRunner(self.ablation_config, config).run(
                indicators=frame, incidents=incidents, independent_incidents=independent_incidents,
                study_end=study_end, risk_holdout_start=self.config.risk_holdout_start,
                evaluation_incidents=evaluation)
            result.metadata['sensitivity_window_minutes'] = minutes
            scales[minutes] = result
            summaries.append(result.summary.assign(window_minutes=minutes))
            pairs.append(result.pairwise.assign(window_minutes=minutes))
            risk.extend(level.risk_metrics.assign(level=name,window_minutes=minutes)
                        for name,level in result.levels.items())
        metadata = dict(protocol='one prespecified multiscale sensitivity; no best-scale selection',
            windows_minutes=self.config.windows_minutes,decision_step_minutes=self.config.decision_step_minutes,
            fixed_risk_holdout_start=self.config.risk_holdout_start,
            primary_experiment='retained 24-hour/hourly experiment; no files modified',
            common_phase_endpoints={phase:int((common & frames[self.config.windows_minutes[0]].phase.eq(phase)).sum())
                                    for phase in ('reference','calibration','test')},
            retrospective_reference_cohort_count=len(evaluation),
            retrospective_reference_cohort='events observed in shortest supported test windows; fixed across scales',
            future_target_registry='complete supplied registry, unchanged across scales/levels',
            outcome_provenance=scales[self.config.windows_minutes[0]].metadata['reference_status'])
        return MultiscaleSensitivityResult(scales, pd.concat(summaries,ignore_index=True),
            pd.concat(pairs,ignore_index=True),pd.concat(risk,ignore_index=True),
            pd.concat(available,ignore_index=True),metadata)
