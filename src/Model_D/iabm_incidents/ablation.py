r"""Matched M0--M3 ablation with chronological calibration and evidence provenance.

M0 uses seven behavioral indicators; M1 adds command residual magnitude; M2
adds exact fourth-order evidence; M3 replaces that exact coordinate with VQE.
No level's own detections are used as outcome labels. Missing independent
maintenance, healthy intervals or recovery truth are reported as unavailable.

Role in the pipeline:
    Quantifies what each additional evidence layer (command-response
    residuals, exact fourth-order transient concentration Tconc, its VQE
    approximation) changes in Model_D warnings, episodes, family evidence
    and weak-reference risk prediction, under one common-support window set
    and one fixed decision policy.

Main public objects:
    * :data:`LEVEL_COLUMNS` -- evidence coordinates of each level.
    * :class:`AblationConfig` -- quantile policy, phase support, horizons.
    * :class:`FourLevelAblationRunner` -- runs all four levels.
    * :class:`FourLevelAblationResult`, :class:`AblationLevelResult`.

Data contract (``indicators`` passed to :meth:`FourLevelAblationRunner.run`):
    One row per decision window with ``asset_id``, ``window_id`` (unique per
    asset), ``start_time``/``end_time`` (timezone-aware or naive; converted
    to naive UTC), ``phase`` in ``{"reference", "calibration", "test"}``,
    the seven :data:`BEHAVIORAL_COLUMNS`, :data:`COMMAND_COLUMN`,
    :data:`EXACT_COLUMN` and :data:`VQE_COLUMN`; optionally
    ``data_coverage`` and ``electrical_status``. Window starts of an asset
    must be at least ``ModelDConfig.window.step`` apart.

Chronological partition and leakage guards:
    * Robust centres/scales use the **reference** phase only.
    * Onset/recovery thresholds use **calibration** scores only.
    * Episodes, warnings and summary metrics are computed on **test** only.
    * Phases must be chronologically disjoint (checked globally across
      assets: ``max(reference.end) <= min(calibration.start)`` and
      ``max(calibration.end) <= min(test.start)``).
    * Risk models train on calibration windows whose full horizon ends
      before the holdout boundary (purge) and are evaluated on test windows
      whose horizon ends before ``study_end`` (right-censoring).
    * Risk targets come from supplied reference incidents, never from any
      level's own episodes.

Equations:
    For coordinate :math:`k` with reference values :math:`x^{ref}_k`,

    .. math::

       c_k = \operatorname{median}(x^{ref}_k), \qquad
       s_k = \max\Big(1.4826\,\operatorname{MAD}(x^{ref}_k),\;
                      \frac{q_{0.95}(x^{ref}_k) - q_{0.05}(x^{ref}_k)}{3.29},\;
                      10^{-6}\Big),

    .. math::

       z_k = \begin{cases} |x_k - c_k| / s_k & k = \text{consumption\_deviation} \\
                           \max(0, (x_k - c_k)/s_k) & \text{otherwise,} \end{cases}

    where the VQE coordinate reuses :math:`c, s` of the exact Tconc. The
    level score is :math:`S_L = \sum_{k \in L} \omega_k z_k` and, on the
    calibration scores, :math:`\tau^{on}_L = Q_{q_{on}}(S_L)`,
    :math:`\tau^{rec}_L = \min(Q_{q_{rec}}(S_L), \mathrm{nextafter}(\tau^{on}_L, -\infty))`.

.. mermaid::

   flowchart TD
       I[indicators] --> V[common support filter]
       V --> R[reference: robust centre/scale]
       R --> Z[directional excess z]
       Z --> S[level scores M0..M3]
       S --> C[calibration: quantile thresholds]
       C --> T[test: warnings + episodes]
       T --> M[summary / pairwise / reference metrics]
       Z --> K[_risk: purged forecaster per horizon]
       K --> M

.. mermaid::

   classDiagram
       class FourLevelAblationRunner {
           +AblationConfig config
           +ModelDConfig model_config
           +run(...) FourLevelAblationResult
           -_episodes(test, config) DataFrame
           -_risk(...) tuple
       }
       class FourLevelAblationResult {
           +dict levels
           +DataFrame summary
           +DataFrame pairwise
           +DataFrame calibration
           +DataFrame support
           +dict metadata
           +export(output_dir) dict
           +assessment() str
       }
       class AblationLevelResult
       FourLevelAblationRunner ..> FourLevelAblationResult
       FourLevelAblationResult o-- AblationLevelResult

Maintenance notes:
    Keep the order of :data:`LEVEL_COLUMNS` (``M0``..``M3``): summary rows
    index calibration rows by level position, and the M3 shared-threshold
    policy requires M2 to be calibrated first. The VQE coordinate must never
    exceed exact Tconc (variational bound); this is enforced with a
    ``1e-7`` tolerance. Exposure accounting attributes to each test window
    the decision interval ``[end_time, end_time + step)`` clipped at
    ``study_end``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, field
from pathlib import Path
import json
import itertools

import numpy as np
import pandas as pd

from .anticipation import AnticipationConfig, AnticipatoryMaintenanceForecaster, _binary_metrics
from .config import ModelDConfig
from .episodes import IncidentEpisodeBuilder
from .features import build_episode_features
from .metrics import summarize_episode_metrics

BEHAVIORAL_COLUMNS = (
    "sequence_divergence", "duration_drift", "recurrence_excess", "persistence_excess",
    "consumption_deviation", "state_error_rate", "mode_divergence",
)
COMMAND_COLUMN = "command_response_deviation"
EXACT_COLUMN = "transient_concentration"
VQE_COLUMN = "transient_concentration_vqe"
LEVEL_COLUMNS = {
    "M0": BEHAVIORAL_COLUMNS,
    "M1": (*BEHAVIORAL_COLUMNS, COMMAND_COLUMN),
    "M2": (*BEHAVIORAL_COLUMNS, COMMAND_COLUMN, EXACT_COLUMN),
    "M3": (*BEHAVIORAL_COLUMNS, COMMAND_COLUMN, VQE_COLUMN),
}


@dataclass(frozen=True)
class AblationConfig:
    """Fixed decision policy and risk horizons; no held-out threshold tuning.

    ``reference``, ``calibration`` and ``test`` phases must be supplied by the
    caller. Centers/scales use reference only; thresholds use calibration only.
    Exact and VQE use the same coordinate transform. By default M3 also shares
    M2 thresholds, isolating solver approximation; an independently calibrated
    M3 threshold is retained as a secondary audit.

    Attributes:
        onset_quantile: Calibration quantile level of the onset threshold.
        recovery_quantile: Calibration quantile level of the recovery
            threshold; must be below ``onset_quantile``.
        share_exact_vqe_threshold: If ``True``, M3 uses M2's thresholds.
        minimum_phase_windows: Minimum common-support windows required in
            each of the three phases.
        risk_horizons_hours: Positive risk horizons in hours.
        risk_target_families: Reference incident families used as risk
            targets.

    Raises:
        ValueError: If ``0 < recovery_quantile < onset_quantile < 1`` does
            not hold, ``minimum_phase_windows < 2`` or any horizon is not
            positive.
    """

    onset_quantile: float = .95
    recovery_quantile: float = .80
    share_exact_vqe_threshold: bool = True
    minimum_phase_windows: int = 24
    risk_horizons_hours: tuple[int, ...] = (6, 24, 72, 168)
    risk_target_families: tuple[str, ...] = ("pump_abrupt_failure", "process_saturation")

    def __post_init__(self) -> None:
        """Validate quantile ordering, phase support and horizons.

        Raises:
            ValueError: On any invalid setting (see class docstring).
        """
        if not 0 < self.recovery_quantile < self.onset_quantile < 1:
            raise ValueError("Require 0 < recovery quantile < onset quantile < 1")
        if self.minimum_phase_windows < 2 or any(h <= 0 for h in self.risk_horizons_hours):
            raise ValueError("Invalid phase support or risk horizon")


@dataclass(frozen=True)
class AblationLevelResult:
    """One level's held-out windows, episodes, family evidence and risk outputs.

    Attributes:
        window_scores: Test-phase windows with the level's coordinates (VQE
            renamed to ``transient_concentration`` for M3),
            ``deviation_score`` and boolean ``warning``.
        episodes: Test-phase episodes with episode features and
            ``onset_decision_time``.
        family_summary: Output of
            :func:`~iabm_incidents.metrics.summarize_episode_metrics`.
        risk_predictions: Test-phase risks per horizon.
        risk_metrics: One row per horizon (status, provenance, metrics).
        risk_coefficients: Fitted forecaster coefficients per horizon.
    """

    window_scores: pd.DataFrame
    episodes: pd.DataFrame
    family_summary: pd.DataFrame
    risk_predictions: pd.DataFrame
    risk_metrics: pd.DataFrame
    risk_coefficients: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class FourLevelAblationResult:
    """Auditable matched-window outputs, calibration, metrics and pairwise deltas.

    Attributes:
        levels: Mapping ``"M0".."M3"`` -> :class:`AblationLevelResult`.
        summary: One row per level (workload, episodes, recovery, healthy
            burden, maintenance overlap and reference metrics).
        pairwise: One row per level pair (decision agreement, Jaccard,
            exclusive warnings, episode and score deltas, optional
            M2/M3 peak displacement).
        calibration: Thresholds and calibration warning fraction per level.
        support: Availability audit of every input window
            (``common_support``, ``missing_dimensions``).
        metadata: JSON-serializable run provenance.
    """

    levels: dict[str, AblationLevelResult]
    summary: pd.DataFrame
    pairwise: pd.DataFrame
    calibration: pd.DataFrame
    support: pd.DataFrame
    metadata: dict

    def export(self, output_dir: str | Path) -> dict[str, Path]:
        """Export English CSV/JSON/Markdown/HTML results and level artifacts.

        Args:
            output_dir: Target directory (created if needed); one
                sub-directory per level is written.

        Returns:
            Mapping from artefact key (``summary``, ``pairwise``,
            ``calibration``, ``support``, ``<level>_<table>``, ``metadata``,
            ``assessment``, ``html``) to written path.

        Raises:
            ValueError: If metadata contains NaN/inf (``allow_nan=False``).
        """
        output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name in ("summary", "pairwise", "calibration", "support"):
            path = output / f"{name}.csv"; getattr(self, name).to_csv(path, index=False); paths[name] = path
        for level, result in self.levels.items():
            folder = output / level; folder.mkdir(exist_ok=True)
            for name in ("window_scores", "episodes", "family_summary", "risk_predictions", "risk_metrics", "risk_coefficients"):
                path = folder / f"{name}.csv"; getattr(result, name).to_csv(path, index=False); paths[f"{level}_{name}"] = path
        path = output / "ablation_metadata.json"
        path.write_text(json.dumps(self.metadata, indent=2, default=str, allow_nan=False), encoding="utf-8"); paths["metadata"] = path
        assessment = self.assessment()
        path = output / "assessment.md"; path.write_text(assessment, encoding="utf-8"); paths["assessment"] = path
        html = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>Four-level Model D ablation</title>",
            "<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:6px}pre{white-space:pre-wrap}</style>",
            "<h1>Four-level Model D ablation</h1>"]
        from html import escape
        html.append("<pre>"+escape(assessment)+"</pre>")
        for name in ("summary", "pairwise", "calibration", "support"):
            html.extend([f"<h2>{name.title()}</h2>", getattr(self, name).to_html(index=False)])
        html.append("<h2>Risk model performance</h2>")
        for level, result in self.levels.items():
            html.extend([f"<h3>{level}</h3>", result.risk_metrics.to_html(index=False)])
        path = output / "ablation_report.html"; path.write_text("".join(html)+"</html>", encoding="utf-8"); paths["html"] = path
        return paths

    def assessment(self) -> str:
        """Generate a data-dependent interpretation without claiming unmeasured truth.

        Returns:
            Markdown text with per-level workload lines, M2/M3 decision
            agreement, fixed interpretation caveats, the 6 h risk assessment
            (when a 6 h horizon was evaluated) and validation availability.

        Raises:
            IndexError: If ``pairwise`` has no ``M2``/``M3`` row.
        """
        lines = ["# Four-level Model D ablation assessment", "",
            "This is a matched, chronologically calibrated retrospective ablation. "
            "M3 substitutes VQE for exact transient evidence; it does not add both coordinates.", "",
            f"Outcome provenance: {self.metadata['reference_status']}. "
            "Derived signal events are exploratory references, not independent physical-fault labels.", ""]
        for row in self.summary.itertuples():
            lines.append(f"- {row.level}: {row.episode_count} episodes; {row.warning_windows} warning windows "
                f"of {row.test_windows}; {row.warning_hours_per_100_observed_hours:.2f} warning hours "
                f"per 100 observed decision hours; median episode span {row.median_episode_hours:.2f} h.")
        solver = self.pairwise[(self.pairwise.left == "M2") & (self.pairwise.right == "M3")].iloc[0]
        lines.extend(["", f"M2/M3 window decision agreement is {solver.decision_agreement:.2%}; "
            f"M3-only warnings: {solver.right_only_warnings}; M2-only warnings: {solver.left_only_warnings}. "
            "A matching decision set does not establish identical eigenvectors or a quantum advantage.", "",
            "Episode count and union-based warning exposure describe candidate workload. "
            "More warnings are not automatically better detection, and fewer warnings are not automatically fewer false alarms. "
            "Fragmentation against derived references is reported separately from physical-event fragmentation.", "",
            "The original family classifier is held fixed. Family counts and raw indicator features describe evidence; "
            "additional electrical evidence does not independently validate family labels.", "",
            "Episode onsets are window-based and retrospective. Detection availability is recorded at the end "
            "of the onset-confirming window; apparent lead from a lookback-window start is not online early warning. "
            "Stable recovery means threshold-confirmed return of the score, not confirmed physical repair.", "",
            "Risk targets are identical fixed external reference events for every level, never each level's own episodes. "
            "Newly fitted stages use declared chronological partitions; the original Model A fit provenance "
            "must be audited separately when retained predictions are supplied. "
            "Training uses calibration-period predictors with a full-horizon purge; held-out labels are right-censored "
            "at the supplied study boundary. Weak-reference risk results measure prediction of that derived event process, "
            "not calibrated probability of physical failure.", ""])
        risk6 = [(level, result.risk_metrics[result.risk_metrics.horizon_hours.eq(6)]) for level, result in self.levels.items()]
        evaluated = [(level, part.iloc[0]) for level, part in risk6 if not part.empty and 'brier_score' in part]
        if evaluated:
            lines.extend(["## Fixed weak-reference risk assessment", ""])
            for level, row in evaluated:
                lines.append(f"- {level}, 6 h: ROC AUC {row.roc_auc:.4f}; Brier {row.brier_score:.4f}; "
                             f"training-prevalence Brier {row.prevalence_baseline_brier:.4f}.")
            if all(row.brier_score > row.prevalence_baseline_brier for _, row in evaluated):
                lines.extend(["", "Every level has a worse Brier score than the constant training-prevalence "
                    "comparator. These results do not support useful calibrated risk prediction at this horizon."])
            lines.extend(["", "Single-class horizons are marked in risk_metrics.csv. Perfect precision/recall on "
                "an all-positive target is not useful evidence of risk discrimination.", ""])
        for metric, status in self.metadata["validation_availability"].items():
            lines.append(f"- {metric}: {status}.")
        lines.extend(["", "The ablation cannot establish superior physical-fault detection unless independent "
            "health labels and temporally aligned intervention/onset/recovery records support that conclusion.", ""])
        return "\n".join(lines)


class FourLevelAblationRunner:
    """Run four nested evidence levels on identical supported windows and labels.

    Input columns include all seven behavioral dimensions, command residual,
    exact and VQE Tconc, window/asset IDs, timestamps and declared phase. Missing
    dimensions are not zero-imputed. Common support is used for every level,
    and the complete availability audit retains excluded windows.

    Attributes:
        config: Ablation decision policy.
        model_config: Model_D configuration (window step/coverage, weights,
            detection defaults; thresholds are overridden per level).

    See Also:
        :class:`~iabm_incidents.behavioral.BehavioralWindowEncoder`,
        :class:`~iabm_incidents.electrical.TransientConcentration`,
        :class:`~iabm_incidents.conformal.ConformalEvidenceFusion`.
    """

    def __init__(self, config: AblationConfig = AblationConfig(), model_d_config: ModelDConfig | None = None) -> None:
        """Create a runner.

        Args:
            config: Ablation policy.
            model_d_config: Model_D configuration; defaults to
                :class:`~iabm_incidents.config.ModelDConfig`.
        """
        self.config, self.model_config = config, model_d_config or ModelDConfig()

    def run(self, *, indicators: pd.DataFrame, incidents: pd.DataFrame | None = None,
            independent_incidents: bool = False, maintenance: pd.DataFrame | None = None,
            healthy_intervals: pd.DataFrame | None = None, study_end: pd.Timestamp | None = None,
            risk_holdout_start: pd.Timestamp | None = None,
            evaluation_incidents: pd.DataFrame | None = None) -> FourLevelAblationResult:
        """Evaluate without self-labelled outcomes or held-out calibration.

        ``incidents`` use incident_id, asset_id, event_time, incident_family and
        optional documented_start/end, recovery_time. Independent provenance is
        an explicit caller declaration. ``maintenance`` and ``healthy_intervals``
        use asset_id/start_time/end_time. Absence of records is not a negative label.
        ``risk_holdout_start`` optionally fixes the outcome-horizon purge boundary
        across different window lengths; omitted preserves the original policy.
        ``evaluation_incidents`` optionally fixes the retrospective matching
        cohort separately from the complete future-target registry.

        Args:
            indicators: Window table (see the module data contract).
            incidents: Reference incident registry used for risk targets
                and, unless ``evaluation_incidents`` is given, matching.
            independent_incidents: Caller declaration that ``incidents``
                are independent of the pipeline; changes metric prefixes,
                provenance labels and availability statements.
            maintenance: Intervention intervals.
            healthy_intervals: Independently healthy intervals.
            study_end: End of the observation period; defaults to the
                latest window end. Timezone-aware values are converted to
                naive UTC.
            risk_holdout_start: Optional purge boundary for risk training;
                defaults to the first test window start.
            evaluation_incidents: Optional separate matching cohort.

        Returns:
            A :class:`FourLevelAblationResult`.

        Raises:
            ValueError: If interval tables lack columns or have
                non-positive durations; required indicator columns are
                missing; asset/window keys are duplicated or windows have
                non-positive duration; window starts of an asset are closer
                than ``window.step``; any phase has fewer than
                ``minimum_phase_windows`` supported windows; phases overlap
                chronologically; VQE exceeds exact Tconc; weights are
                negative or non-finite; incident IDs repeat within an
                asset; or the risk holdout boundary does not separate
                calibration and test.

        Notes:
            Common support requires every coordinate finite, and, when the
            columns exist, ``data_coverage >= window.min_coverage`` and
            ``electrical_status == "supported"``. ``semantic_status`` is
            dropped before episode building so Model_C labels cannot open
            episodes.
        """
        for name, table in (("maintenance", maintenance), ("healthy intervals", healthy_intervals)):
            if table is not None and not table.empty:
                if not {"asset_id", "start_time", "end_time"}.issubset(table):
                    raise ValueError(f"{name} require asset_id/start_time/end_time")
                table = table.copy()
                table["start_time"], table["end_time"] = _timestamps(table.start_time), _timestamps(table.end_time)
                if (table.end_time <= table.start_time).any():
                    raise ValueError(f"{name} require positive interval durations")
                if name == "maintenance":
                    maintenance = table
                else:
                    healthy_intervals = table
        required = ["asset_id", "window_id", "start_time", "end_time", "phase", *BEHAVIORAL_COLUMNS, COMMAND_COLUMN, EXACT_COLUMN, VQE_COLUMN]
        missing = set(required)-set(indicators)
        if missing:
            raise ValueError(f"Missing ablation columns: {sorted(missing)}")
        frame = indicators.copy()
        for column in ("start_time", "end_time"):
            frame[column] = _timestamps(frame[column])
        if frame.duplicated(["asset_id", "window_id"]).any() or (frame.end_time <= frame.start_time).any():
            raise ValueError("Unique asset/window keys and positive durations required")
        for _, asset_frame in frame.sort_values('start_time').groupby('asset_id'):
            if asset_frame.start_time.diff().dropna().lt(self.model_config.window.step).any():
                raise ValueError("Decision windows must not have overlapping step intervals")
        columns = [*BEHAVIORAL_COLUMNS, COMMAND_COLUMN, EXACT_COLUMN, VQE_COLUMN]
        numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
        valid = pd.Series(np.isfinite(numeric).all(axis=1), index=frame.index)
        if "data_coverage" in frame:
            valid &= pd.to_numeric(frame.data_coverage, errors="coerce").ge(self.model_config.window.min_coverage)
        if "electrical_status" in frame:
            valid &= frame.electrical_status.eq("supported")
        support = frame[["asset_id", "window_id", "phase", "start_time", "end_time"]].copy()
        support["common_support"] = valid
        support["missing_dimensions"] = numeric.apply(lambda row: ",".join(row.index[~np.isfinite(row)]), axis=1)
        frame[columns] = numeric
        frame = frame.loc[valid].sort_values(["asset_id", "start_time"]).reset_index(drop=True)
        phases = {phase: frame[frame.phase.eq(phase)].copy() for phase in ("reference", "calibration", "test")}
        if any(len(part) < self.config.minimum_phase_windows for part in phases.values()):
            raise ValueError("Insufficient matched reference/calibration/test support; see raw availability")
        if phases['reference'].end_time.max() > phases['calibration'].start_time.min() or phases['calibration'].end_time.max() > phases['test'].start_time.min():
            raise ValueError("Reference/calibration/test intervals must be chronologically disjoint")
        if (frame[VQE_COLUMN] > frame[EXACT_COLUMN]+1e-7).any():
            raise ValueError("VQE violates the same-operator variational bound")
        center, scale, transformed = {}, {}, pd.DataFrame(index=frame.index)
        # Invariant: centres/scales come from the reference phase only; the
        # VQE column (last) is handled below with the exact Tconc transform.
        for column in columns[:-1]:
            values = phases['reference'][column]
            center[column] = float(values.median())
            mad = float(1.4826*(values-center[column]).abs().median())
            scale[column] = max(mad, float(values.quantile(.95)-values.quantile(.05))/3.29, 1e-6)
            delta = (frame[column]-center[column])/scale[column]
            transformed[column] = delta.abs() if column == 'consumption_deviation' else delta.clip(lower=0)
        center[VQE_COLUMN], scale[VQE_COLUMN] = center[EXACT_COLUMN], scale[EXACT_COLUMN]
        transformed[VQE_COLUMN] = ((frame[VQE_COLUMN]-center[EXACT_COLUMN])/scale[EXACT_COLUMN]).clip(lower=0)
        weight_names = ("sequence", "duration", "recurrence", "persistence", "consumption", "state_error", "mode", "command_response", "transient_concentration")
        # Note: the trailing name maps the VQE column onto the Tconc weight.
        weights = {c: getattr(self.model_config.weights, n) for c, n in zip(columns, (*weight_names, "transient_concentration"))}
        if any(weight < 0 or not np.isfinite(weight) for weight in weights.values()):
            raise ValueError("Finite nonnegative weights required for matched variational scoring")
        all_scores = {level: sum((transformed[c]*weights[c] for c in cols), pd.Series(0., index=frame.index)) for level, cols in LEVEL_COLUMNS.items()}
        thresholds, calibration_rows = {}, []
        for level, score in all_scores.items():
            cal = score[frame.phase.eq('calibration')]
            own_onset = float(cal.quantile(self.config.onset_quantile))
            own_recovery = float(cal.quantile(self.config.recovery_quantile))
            onset, recovery = (thresholds['M2'] if level == 'M3' and self.config.share_exact_vqe_threshold else (own_onset, own_recovery))
            # Avoid equal thresholds for a completely tied calibration distribution.
            recovery = min(recovery, np.nextafter(onset, -np.inf))
            thresholds[level] = onset, recovery
            calibration_rows.append(dict(level=level, onset_threshold=onset, recovery_threshold=recovery,
                independently_calibrated_onset=own_onset, independently_calibrated_recovery=own_recovery,
                calibration_warning_fraction=float(cal.ge(onset).mean()), calibration_windows=len(cal)))
        references = _normalize_incidents(incidents)
        matching_references = references if evaluation_incidents is None else _normalize_incidents(evaluation_incidents)
        end = pd.Timestamp(study_end) if study_end is not None else frame.end_time.max()
        if end.tzinfo is not None:
            end = end.tz_convert('UTC').tz_localize(None)
        levels, summary_rows = {}, []
        for level, cols in LEVEL_COLUMNS.items():
            full = frame.copy()
            # Excluded coordinates are absent, so family features cannot leak ablated evidence.
            full = full.drop(columns=[c for c in columns if c not in cols])
            if level == 'M3':
                full = full.rename(columns={VQE_COLUMN: EXACT_COLUMN})
            full['deviation_score'] = all_scores[level]
            full['warning'] = full.deviation_score.ge(thresholds[level][0])
            test = full[full.phase.eq('test')].copy().reset_index(drop=True)
            detection = replace(self.model_config.detection, onset_threshold=thresholds[level][0], recovery_threshold=thresholds[level][1])
            model_config = replace(self.model_config, detection=detection)
            episodes = self._episodes(test, model_config)
            family = summarize_episode_metrics(episodes)
            risk_predictions, risk_metrics, risk_coefficients = self._risk(frame, transformed, cols, references, independent_incidents, end, risk_holdout_start)
            levels[level] = AblationLevelResult(test, episodes, family, risk_predictions, risk_metrics, risk_coefficients)
            # Why: each window owns the decision interval [end, end + step),
            # so overlapping lookback spans are not double counted.
            exposure = (test.end_time+self.model_config.window.step).clip(upper=end)-test.end_time
            exposure_hours = exposure.dt.total_seconds().clip(lower=0)/3600
            observed = float(exposure_hours.sum())
            warnings = int(test.warning.sum())
            warning_hours = float(exposure_hours[test.warning].sum())
            metrics = _reference_metrics(test, episodes, matching_references, independent_incidents)
            healthy_hours, false_hours = _healthy_warning_hours(test, healthy_intervals, self.model_config.window.step, end)
            summary_rows.append(dict(level=level, test_windows=len(test), observed_decision_hours=observed,
                warning_windows=warnings, warning_hours_per_100_observed_hours=100*warning_hours/observed if observed else np.nan,
                episode_count=len(episodes),
                independently_calibrated_warning_windows=int(all_scores[level][frame.phase.eq("test")].ge(calibration_rows[list(LEVEL_COLUMNS).index(level)]["independently_calibrated_onset"]).sum()),
                episodes_exceeding_configured_maximum_duration=int(episodes.duration_seconds.gt(self.model_config.detection.maximum_duration.total_seconds()).sum()) if not episodes.empty else 0, episodes_per_100_observed_hours=100*len(episodes)/observed if observed else np.nan,
                median_episode_hours=float(episodes.duration_seconds.median()/3600) if not episodes.empty else 0.,
                score_confirmed_recovery_fraction=float(episodes.recovery_status.eq('stable').mean()) if not episodes.empty else np.nan,
                median_score_recovery_hours=float(episodes.time_to_recovery_seconds.dropna().median()/3600) if not episodes.empty and episodes.time_to_recovery_seconds.notna().any() else np.nan,
                independently_healthy_hours=healthy_hours, false_warning_hours=false_hours,
                false_warning_hours_per_100_healthy_hours=100*false_hours/healthy_hours if healthy_hours > 0 else np.nan,
                **_maintenance_metrics(episodes, maintenance, test), **metrics))
        pairwise_rows = []
        for left, right in itertools.combinations(levels, 2):
            a, b = levels[left].window_scores, levels[right].window_scores
            x, y = a.warning.to_numpy(bool), b.warning.to_numpy(bool)
            peak_delta = np.array([])
            if left == 'M2' and right == 'M3' and 'exact_projection_peak_time' in a and 'vqe_projection_peak_time' in b:
                peak_delta = (pd.to_datetime(b.vqe_projection_peak_time)-pd.to_datetime(a.exact_projection_peak_time)).dt.total_seconds().abs().dropna().to_numpy()
            pairwise_rows.append(dict(left=left, right=right,
                retrospective_peak_pairs=len(peak_delta),
                median_absolute_retrospective_peak_displacement_seconds=float(np.median(peak_delta)) if len(peak_delta) else np.nan,
                p95_absolute_retrospective_peak_displacement_seconds=float(np.quantile(peak_delta,.95)) if len(peak_delta) else np.nan,
                decision_agreement=float((x == y).mean()),
                shared_warnings=int((x & y).sum()), left_only_warnings=int((x & ~y).sum()), right_only_warnings=int((~x & y).sum()),
                warning_jaccard=float((x & y).sum()/(x | y).sum()) if (x | y).any() else 1.,
                episode_count_delta=len(levels[right].episodes)-len(levels[left].episodes),
                median_raw_score_delta=float((b.deviation_score-a.deviation_score).median())))
        reference_status = "independent reported incidents" if independent_incidents and not references.empty else "weak derived signal references" if not references.empty else "no incident references"
        metadata = dict(level_columns={k: list(v) for k, v in LEVEL_COLUMNS.items()},
            config=self.config.__dict__, model_d_config=self.model_config.to_dict(), centers=center, scales=scale,
            common_support_windows=len(frame), reference_status=reference_status,
            phase_windows={k:len(v) for k,v in phases.items()}, study_end=end,
            risk_holdout_start=risk_holdout_start, evaluation_incident_count=len(matching_references),
            threshold_policy="M3 shares M2" if self.config.share_exact_vqe_threshold else "independent level calibration",
            validation_availability={
                "maintenance overlap": "available from supplied intervention intervals" if maintenance is not None and not maintenance.empty else "unavailable: no temporally aligned intervention intervals",
                "false warning burden": "available on supplied independently healthy intervals" if healthy_intervals is not None and not healthy_intervals.empty else "unavailable: no independently healthy intervals; total warning burden is reported",
                "physical onset/recovery accuracy": "available where supplied independent records contain timestamps" if independent_incidents else "unavailable: independent timestamped onset/recovery truth absent",
                "physical risk performance": "reported-event prediction available; physical-failure validation still requires adjudicated record completeness" if independent_incidents else "unavailable: only fixed weak-reference risk performance can be evaluated",
            })
        return FourLevelAblationResult(levels, pd.DataFrame(summary_rows), pd.DataFrame(pairwise_rows), pd.DataFrame(calibration_rows), support, metadata)

    def _episodes(self, test: pd.DataFrame, config: ModelDConfig) -> pd.DataFrame:
        """Do not bridge excluded windows or use another asset's recovery evidence.

        Test windows are split per asset into contiguous blocks (a new block
        starts when consecutive starts are more than ``window.step`` apart)
        and episodes are built independently per block.

        Args:
            test: Test-phase windows of one level with ``deviation_score``.
            config: Model_D configuration carrying the level's calibrated
                thresholds.

        Returns:
            Episodes with merged episode features, ``onset_decision_time``
            (end time of the window that confirms onset, i.e. the
            ``onset_windows``-th window of the episode, or its last window
            if shorter) and ``episode_id`` prefixed ``<asset>_<block>_``.
            An empty canonical episode frame when none exist.
        """
        results = []
        for asset, group in test.groupby('asset_id', sort=False):
            block = group.start_time.diff().gt(config.window.step).cumsum()
            for number, part in group.groupby(block, sort=False):
                part = part.drop(columns=['semantic_status'], errors='ignore').reset_index(drop=True)
                episodes = IncidentEpisodeBuilder(config).build_episodes_from_window_scores(part)
                if episodes.empty:
                    continue
                features = build_episode_features(episodes, part)
                keys = ['episode_id', 'asset_id']
                features = features.drop(columns=[c for c in features if c in episodes and c not in keys])
                episodes = episodes.merge(features, on=keys, how='left')
                for i, episode in episodes.iterrows():
                    # Note: source_window_ids are positional labels into ``part``.
                    ids = [int(value) for value in episode.source_window_ids]
                    confirmation = ids[min(config.detection.onset_windows-1, len(ids)-1)]
                    episodes.loc[i, 'onset_decision_time'] = part.loc[confirmation, 'end_time']
                episodes['episode_id'] = str(asset)+'_'+str(number)+'_'+episodes.episode_id
                results.append(episodes)
        return pd.concat(results, ignore_index=True) if results else IncidentEpisodeBuilder(config)._empty_episode_frame()

    def _risk(self, frame, transformed, columns, references, independent, end, holdout_start=None):
        """Fit and evaluate the purged weak-reference risk model of one level.

        For each horizon :math:`h`, the predictors are the level's
        directional excesses (prefixed ``feat_``) and the target of a window
        is 1 when a reference incident of a target family on the same asset
        has ``event_time`` in ``(window_end, window_end + h]``.

        Args:
            frame: Common-support windows of all phases.
            transformed: Directional excesses aligned with ``frame``.
            columns: The level's coordinates.
            references: Normalized reference incidents.
            independent: Whether references are declared independent.
            end: Study end (naive UTC).
            holdout_start: Optional purge boundary; defaults to the first
                test window start.

        Returns:
            Tuple ``(predictions, metrics, coefficients)`` of DataFrames.
            Each metrics row has ``status`` ``evaluated``,
            ``single_class_target`` or ``unavailable_targets_or_support``.

        Raises:
            ValueError: If the boundary is after the first test start or
                before the last calibration end.

        Notes:
            Training rows: calibration windows with
            ``window_end + h <= cutoff`` (full-horizon purge). Test rows:
            test windows with ``window_end + h <= study_end``
            (right-censoring). The prevalence baseline Brier uses the
            training positive rate as a constant forecast.
        """
        rows, predictions, coefficients = [], [], []
        cutoff = pd.Timestamp(holdout_start) if holdout_start is not None else frame.loc[frame.phase.eq('test'), 'start_time'].min()
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert('UTC').tz_localize(None)
        if cutoff > frame.loc[frame.phase.eq('test'), 'start_time'].min() or cutoff < frame.loc[frame.phase.eq('calibration'), 'end_time'].max():
            raise ValueError('Risk holdout boundary must separate calibration and test evidence')
        for horizon in self.config.risk_horizons_hours:
            predictors = transformed[list(columns)].rename(columns=lambda c: 'feat_'+c)
            data = pd.concat([frame[['asset_id', 'window_id', 'end_time', 'phase']].rename(columns={'end_time':'window_end'}), predictors], axis=1)
            target = np.zeros(len(data))
            relevant = references[references.incident_family.isin(self.config.risk_target_families)]
            for asset, group in data.groupby('asset_id'):
                times = relevant.loc[relevant.asset_id == asset, 'event_time'].sort_values().to_numpy(dtype='datetime64[ns]')
                ends = group.window_end.to_numpy(dtype='datetime64[ns]')
                target[group.index] = np.searchsorted(times, ends+np.timedelta64(horizon, 'h'), side='right') > np.searchsorted(times, ends, side='right')
            data[f'target_h{horizon}'] = target
            train = data[data.phase.eq('calibration') & (data.window_end+pd.Timedelta(hours=horizon) <= cutoff)]
            test = data[data.phase.eq('test') & (data.window_end+pd.Timedelta(hours=horizon) <= end)]
            provenance = 'independent_event_target' if independent else 'weak_reference_proxy_not_physical_failure'
            if references.empty or train.empty or test.empty:
                rows.append(dict(horizon_hours=horizon, status='unavailable_targets_or_support', target_provenance=provenance))
                continue
            forecaster = AnticipatoryMaintenanceForecaster(AnticipationConfig(horizons_hours=(horizon,)))
            forecaster.fit(train)
            coefficients.append(forecaster.horizon_coefficients())
            scored = forecaster.score(test)
            threshold = forecaster._models[horizon].decision_threshold
            stats = _binary_metrics(scored[f'target_h{horizon}'], scored[f'risk_h{horizon}'], threshold)
            probability = scored[f'risk_h{horizon}'].to_numpy(float)
            truth = scored[f'target_h{horizon}'].to_numpy(float)
            rows.append(dict(horizon_hours=horizon, status='evaluated' if train[f'target_h{horizon}'].nunique()==2 and np.unique(truth).size==2 else 'single_class_target',
                target_provenance=provenance, train_windows=len(train), test_windows=len(test),
                train_positive_rate=float(train[f'target_h{horizon}'].mean()), test_positive_rate=float(truth.mean()),
                prevalence_baseline_brier=float(np.mean((float(train[f'target_h{horizon}'].mean())-truth)**2)),
                roc_auc=_roc_auc(truth, probability),
                average_precision=_average_precision(truth, probability),
                **stats))
            predictions.append(scored[['asset_id','window_id','window_end', f'target_h{horizon}', f'risk_h{horizon}']].assign(horizon_hours=horizon))
        return (pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(columns=['asset_id','window_id','horizon_hours']),
                pd.DataFrame(rows), pd.concat(coefficients, ignore_index=True) if coefficients else pd.DataFrame(columns=['horizon_hours','feature']))


def _timestamps(values):
    """Parse timestamps as UTC and return them timezone-naive.

    Args:
        values: Series of timestamps or strings (mixed formats allowed).

    Returns:
        ``datetime64[ns]`` series in naive UTC.

    Raises:
        ValueError: If a value cannot be parsed.
    """
    return pd.to_datetime(values, utc=True, format='mixed', errors='raise').dt.tz_convert(None)


def _normalize_incidents(incidents):
    """Normalize a reference incident table.

    Args:
        incidents: Table with ``incident_id``, ``event_time``,
            ``incident_family`` and optional ``asset_id``,
            ``documented_start``, ``documented_end``, ``recovery_time``;
            or ``None``.

    Returns:
        Copy with all time columns parsed to naive UTC (missing ones as
        ``NaT``), ``asset_id`` defaulted to ``"asset-unknown"`` and
        documented bounds defaulted to ``event_time``. An empty table with
        the canonical columns for ``None``/empty input.

    Raises:
        ValueError: If ``(asset_id, incident_id)`` is not unique.
    """
    if incidents is None or incidents.empty:
        return pd.DataFrame(columns=['incident_id','asset_id','event_time','incident_family','documented_start','documented_end','recovery_time'])
    frame = incidents.copy()
    if 'asset_id' not in frame:
        frame['asset_id'] = 'asset-unknown'
    for c in ('event_time','documented_start','documented_end','recovery_time'):
        if c in frame:
            frame[c] = _timestamps(frame[c])
        else:
            frame[c] = pd.NaT
    frame['documented_start'] = frame.documented_start.fillna(frame.event_time)
    frame['documented_end'] = frame.documented_end.fillna(frame.event_time)
    if frame.duplicated(['asset_id','incident_id']).any():
        raise ValueError('Incident identifiers must be unique within asset')
    return frame


def _reference_metrics(test, episodes, references, independent):
    """Match test episodes to reference events observed in the test phase.

    A reference counts when its ``event_time`` lies inside some test window
    of its asset. Point events (documented start == end) match episodes that
    contain ``event_time``; interval events match overlapping episodes.

    Args:
        test: Test windows of one level.
        episodes: That level's test episodes.
        references: Normalized matching references.
        independent: If ``False``, metric names get the ``weak_reference_``
            prefix.

    Returns:
        Dictionary with ``reference_events``, ``event_overlap_fraction``,
        ``fragmentation_mean_episodes_per_hit``,
        ``fragmentation_excess_episodes``,
        ``median_absolute_onset_decision_error_hours`` (onset decision time
        minus documented start of the closest candidate),
        ``median_absolute_recovery_error_hours`` and ``family_agreement``
        (NaN when not computable).
    """
    prefix = '' if independent else 'weak_reference_'
    overlap_counts, onset, recovery, family = [], [], [], []
    relevant = references[(references.event_time >= test.start_time.min()) & (references.event_time < test.end_time.max())]
    for ref in relevant.itertuples():
        asset = test[test.asset_id == ref.asset_id]
        if asset.empty or not ((asset.start_time <= ref.event_time) & (asset.end_time > ref.event_time)).any():
            continue
        candidates = episodes[episodes.asset_id == ref.asset_id]
        if ref.documented_end == ref.documented_start:
            candidates = candidates[(candidates.event_start <= ref.event_time) & (candidates.event_end > ref.event_time)]
        else:
            candidates = candidates[(candidates.event_start < ref.documented_end) & (candidates.event_end > ref.documented_start)]
        overlap_counts.append(len(candidates))
        if not candidates.empty:
            # Localization uses closest confirming decision, never a family-constrained zero-overlap match.
            best = candidates.iloc[np.argmin(abs((candidates.onset_decision_time-ref.documented_start).dt.total_seconds()))]
            onset.append((best.onset_decision_time-ref.documented_start).total_seconds())
            family.append(best.primary_family == ref.incident_family)
            if pd.notna(ref.recovery_time) and pd.notna(best.recovery_end):
                recovery.append((best.recovery_end-ref.recovery_time).total_seconds())
    hits = [count for count in overlap_counts if count > 0]
    return {prefix+'reference_events':len(overlap_counts), prefix+'event_overlap_fraction':len(hits)/len(overlap_counts) if overlap_counts else np.nan,
        prefix+'fragmentation_mean_episodes_per_hit':float(np.mean(hits)) if hits else np.nan,
        prefix+'fragmentation_excess_episodes':int(sum(max(n-1,0) for n in hits)),
        prefix+'median_absolute_onset_decision_error_hours':float(np.median(abs(np.asarray(onset)))/3600) if onset else np.nan,
        prefix+'median_absolute_recovery_error_hours':float(np.median(abs(np.asarray(recovery)))/3600) if recovery else np.nan,
        prefix+'family_agreement':float(np.mean(family)) if family else np.nan}


def _maintenance_metrics(episodes, maintenance, test):
    """Measure how many observed maintenance periods overlap an episode.

    Args:
        episodes: Test episodes of one level.
        maintenance: Normalized intervention intervals or ``None``.
        test: Test windows (used to keep only periods overlapping observed
            windows of the same asset).

    Returns:
        ``maintenance_periods``, ``maintenance_period_hits`` and
        ``maintenance_overlap_fraction``; all NaN without eligible periods.
    """
    if maintenance is None or maintenance.empty:
        return dict(maintenance_periods=np.nan, maintenance_period_hits=np.nan, maintenance_overlap_fraction=np.nan)
    eligible = []
    for _, period in maintenance.iterrows():
        observed = test[test.asset_id == period.asset_id]
        if ((observed.start_time < period.end_time) & (observed.end_time > period.start_time)).any():
            eligible.append(period)
    if not eligible:
        return dict(maintenance_periods=np.nan, maintenance_period_hits=np.nan, maintenance_overlap_fraction=np.nan)
    maintenance = pd.DataFrame(eligible)
    hits = 0
    for row in maintenance.itertuples():
        hit = episodes[(episodes.asset_id == row.asset_id) & (episodes.event_start < pd.Timestamp(row.end_time)) & (episodes.event_end > pd.Timestamp(row.start_time))]
        hits += not hit.empty
    return dict(maintenance_periods=len(maintenance), maintenance_period_hits=hits, maintenance_overlap_fraction=hits/len(maintenance))


def _healthy_warning_hours(windows, healthy, step, study_end=None):
    """Compute observed and warning hours inside independently healthy time.

    Healthy intervals are merged per asset; each window contributes the
    intersection of its decision interval ``[end_time, end_time + step)``
    (clipped at ``study_end``) with the merged intervals.

    Args:
        windows: Test windows with ``asset_id``, ``end_time``, ``warning``.
        healthy: Normalized healthy intervals or ``None``.
        step: Decision-interval length.
        study_end: Optional clipping bound.

    Returns:
        ``(healthy_hours, false_warning_hours)``; ``(nan, nan)`` without
        healthy intervals.
    """
    if healthy is None or healthy.empty:
        return np.nan, np.nan
    # Count union intersections, not overlapping 24-hour lookback spans.
    observed = warning = 0.
    for asset, group in windows.groupby('asset_id'):
        spans = sorted((pd.Timestamp(r.start_time), pd.Timestamp(r.end_time)) for r in healthy[healthy.asset_id == asset].itertuples())
        merged = []
        for start, end in spans:
            if merged and start <= merged[-1][1]:
                merged[-1] = merged[-1][0], max(end,merged[-1][1])
            else:
                merged.append((start,end))
        for row in group.itertuples():
            for start, end in merged:
                limit = min(row.end_time+step, end, study_end) if study_end is not None else min(row.end_time+step,end)
                seconds = max((limit-max(row.end_time,start)).total_seconds(),0.)
                observed += seconds; warning += seconds*bool(row.warning)
    return observed/3600, warning/3600


def _roc_auc(truth, probability):
    r"""Tie-aware ROC AUC without assuming independent windows.

    Mann--Whitney form with average ranks :math:`R_i`:

    .. math:: \mathrm{AUC} = \frac{\sum_{i \in +} R_i - n_1(n_1+1)/2}{n_1 n_0}.

    Args:
        truth: Array of labels (> 0.5 is positive).
        probability: Array of scores.

    Returns:
        AUC in ``[0, 1]``; NaN when one class is absent. No variance or
        confidence interval is computed.
    """
    positives = truth > .5
    n1, n0 = int(positives.sum()), int((~positives).sum())
    if not n1 or not n0:
        return np.nan
    ranks = pd.Series(probability).rank(method='average').to_numpy()
    return float((ranks[positives].sum()-n1*(n1+1)/2)/(n1*n0))


def _average_precision(truth, probability):
    r"""Threshold-grouped average precision, handling equal probabilities together.

    .. math:: \mathrm{AP} = \sum_t (R_t - R_{t-1})\, P_t

    over distinct score thresholds :math:`t` in decreasing order, where tied
    scores enter the ranking together.

    Args:
        truth: Array of labels (> 0.5 is positive).
        probability: Array of scores.

    Returns:
        Average precision; NaN when there are no positives.
    """
    positives = int((truth > .5).sum())
    if not positives:
        return np.nan
    order = np.argsort(-probability,kind='stable')
    y,p = (truth[order] > .5).astype(int),probability[order]
    ends = np.r_[np.flatnonzero(p[:-1] != p[1:]),len(p)-1]
    cumulative = np.cumsum(y)[ends]
    increments = np.diff(np.r_[0,cumulative])
    return float(np.sum(increments/positives*cumulative/(ends+1)))
