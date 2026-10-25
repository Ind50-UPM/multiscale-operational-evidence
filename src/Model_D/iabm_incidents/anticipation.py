r"""Experimental anticipatory maintenance modeling for Model_D.

Role in the pipeline:
    Turns scored Model_D windows into per-horizon early-warning risks that a
    target-family episode (default ``process_saturation`` or
    ``pump_abrupt_failure``) will *start* within the next :math:`h` hours,
    and maps the latest risk per asset to a maintenance action. The matched
    ablation (:mod:`iabm_incidents.ablation`) reuses
    :class:`AnticipatoryMaintenanceForecaster` with fixed external
    reference targets.

Main public objects:
    * :class:`AnticipationConfig`, :class:`HorizonModel`,
      :class:`AnticipationResult` -- configuration and outputs.
    * :class:`AnticipatoryMaintenanceForecaster` -- fit/score/evaluate/export.
    * :func:`build_anticipation_frame` -- feature and target construction.
    * :func:`temporal_holdout_split`, :func:`walk_forward_validation`.
    * :func:`build_maintenance_actions`.

Data contract:
    ``window_scores``: one row per window with ``start_time``/``end_time``
    (or ``window_start``/``window_end``), optional ``asset_id``,
    ``semantic_status``, ``incident_family`` and any of
    :data:`DEFAULT_FEATURE_COLUMNS`. ``episodes``: ``event_start``,
    ``event_end``, ``primary_family`` and optional ``asset_id``. Feature
    columns are prefixed ``feat_``; targets are named ``target_h<h>`` and
    risks ``risk_h<h>``.

Model:
    Each horizon uses a transparent linear-logistic score, *not* a fitted
    maximum-likelihood logistic regression. With training prevalence
    :math:`\pi`, feature means/standard deviations :math:`\mu_j,\sigma_j`
    and class-conditional means :math:`\mu_j^{+},\mu_j^{-}`,

    .. math::

       w_j = \frac{\mu_j^{+} - \mu_j^{-}}{\sigma_j}, \qquad
       r(x) = \sigma\Big(\operatorname{logit}(\pi)
              + \sum_j w_j \frac{x_j - \mu_j}{\sigma_j}\Big).

    The decision threshold is chosen on the training scores from a grid in
    ``[0.15, 0.85]`` (see :func:`_select_decision_threshold`).

.. mermaid::

   flowchart LR
       W[window_scores] --> B[build_anticipation_frame]
       E[episodes] --> B
       B --> S[temporal_holdout_split]
       S -->|train| F[fit: HorizonModel per horizon]
       F --> SC[score: risk_h*, max_risk]
       S -->|train/test| EV[evaluate + walk_forward_validation]
       SC --> A[build_maintenance_actions]

Maintenance notes:
    Rolling feature names (``_delta_1h``, ``_mean_6h``, ``_24h``) count
    *rows*, so they equal hours only for a 1-hour window step. Rolling
    features are causal (backward-looking) and computed per asset.
    The chronological holdout sorts all assets jointly by ``window_end`` and
    applies **no purge/embargo**: training windows near the split have
    targets whose horizon reaches into the test period, so the train/test
    metrics of :meth:`AnticipatoryMaintenanceForecaster.run` are optimistic.
    The ablation performs its own full-horizon purge instead. Context
    columns (``current_*``, ``hour_of_day`` ...) without the ``feat_``
    prefix are kept for reporting but are not model inputs; ``hour_of_day``,
    ``day_of_week`` and ``is_night_shift`` are dropped from the returned
    feature frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_FEATURE_COLUMNS = [
    "deviation_score",
    "sequence_divergence",
    "duration_drift",
    "recurrence_excess",
    "persistence_excess",
    "consumption_deviation",
    "state_error_rate",
    "mode_divergence",
    "state_word_diversity",
    "dominant_state_word_fraction",
    "state_word_transition_rate",
    "nominal_state_word_match_fraction",
    "mean_state_distance",
    "mean_dtw_distance",
    "mean_nominal_anomaly_score",
    "rare_word_fraction",
    "rare_state_fraction",
    "state_entropy",
    "state_17_fraction",
    "off_nominal_state_fraction",
    "word_regime_shift_score",
]
ROLLING_WINDOWS = (6, 24)
DEFAULT_TARGET_FAMILIES = ("process_saturation", "pump_abrupt_failure")


@dataclass(frozen=True)
class AnticipationConfig:
    """Configuration of the anticipatory forecaster.

    Attributes:
        horizons_hours: Forecast horizons in hours; one model per horizon.
        target_families: Episode ``primary_family`` values that count as
            positive future events.
        holdout_fraction: Fraction of the latest windows (by
            ``window_end``) reserved for the test split.
        risk_threshold: ``max_risk`` at or above which the action is
            ``inspect_and_prepare_intervention`` (priority ``high``).
        warning_threshold: ``max_risk`` at or above which the action is
            ``increase_monitoring_and_schedule_check`` (priority
            ``medium``).
    """

    horizons_hours: tuple[int, ...] = (24, 72, 168)
    target_families: tuple[str, ...] = DEFAULT_TARGET_FAMILIES
    holdout_fraction: float = 0.30
    risk_threshold: float = 0.55
    warning_threshold: float = 0.35


@dataclass(frozen=True)
class HorizonModel:
    """Frozen linear-logistic risk model for one horizon.

    Attributes:
        horizon_hours: Forecast horizon in hours.
        intercept: :math:`\\operatorname{logit}` of the clipped training
            prevalence.
        feature_weights: Standardized mean difference per ``feat_`` column.
        feature_means: Training mean per feature (also the NaN fill value
            at scoring time).
        feature_stds: Training population standard deviation per feature
            (1.0 when zero).
        positive_rate: Training prevalence of the target.
        decision_threshold: Risk threshold selected on training scores.
    """

    horizon_hours: int
    intercept: float
    feature_weights: dict[str, float]
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    positive_rate: float
    decision_threshold: float


@dataclass(frozen=True)
class AnticipationResult:
    """Outputs of :meth:`AnticipatoryMaintenanceForecaster.run`.

    Attributes:
        config: Configuration used.
        feature_frame: Features and targets for all windows.
        scored_windows: ``feature_frame`` plus ``risk_h<h>``, ``max_risk``
            and ``dominant_horizon_hours``.
        validation_summary: Metrics per split (``train``, ``test``,
            ``walk_forward``) and horizon.
        horizon_coefficients: Intercepts and weights per horizon.
        maintenance_actions: Latest window per asset with action/priority.
    """

    config: AnticipationConfig
    feature_frame: pd.DataFrame
    scored_windows: pd.DataFrame
    validation_summary: pd.DataFrame
    horizon_coefficients: pd.DataFrame
    maintenance_actions: pd.DataFrame


class AnticipatoryMaintenanceForecaster:
    """Fit interpretable early-warning models on top of Model_D windows and episodes.

    Examples:
        >>> forecaster = AnticipatoryMaintenanceForecaster(AnticipationConfig(horizons_hours=(24,)))
        >>> forecaster.horizon_coefficients().empty
        True
    """

    def __init__(self, config: AnticipationConfig | None = None) -> None:
        """Create an unfitted forecaster.

        Args:
            config: Forecaster configuration; defaults to
                :class:`AnticipationConfig`.
        """
        self._config = config or AnticipationConfig()
        self._models: dict[int, HorizonModel] = {}
        self._feature_columns: list[str] = []

    def run(self, *, window_scores: pd.DataFrame, episodes: pd.DataFrame) -> AnticipationResult:
        """Build features, split chronologically, fit, score, evaluate and plan actions.

        Args:
            window_scores: Scored Model_D windows.
            episodes: Model_D episodes used to derive future targets.

        Returns:
            An :class:`AnticipationResult`.

        Notes:
            The models are fitted on the earliest
            ``1 - holdout_fraction`` of windows only, but scored on every
            window; ``scored_windows`` therefore includes in-sample rows.
            Targets are derived from the same pipeline's episodes (not
            independent outcomes). See the module maintenance notes about
            the missing purge between splits.
        """
        feature_frame = build_anticipation_frame(
            window_scores=window_scores,
            episodes=episodes,
            horizons_hours=self._config.horizons_hours,
            target_families=self._config.target_families,
        )
        train_frame, test_frame = temporal_holdout_split(feature_frame, self._config.holdout_fraction)
        self.fit(train_frame)
        scored_windows = self.score(feature_frame)
        validation_summary = self.evaluate(train_frame=train_frame, test_frame=test_frame)
        horizon_coefficients = self.horizon_coefficients()
        maintenance_actions = build_maintenance_actions(scored_windows, self._config)
        return AnticipationResult(
            config=self._config,
            feature_frame=feature_frame,
            scored_windows=scored_windows,
            validation_summary=validation_summary,
            horizon_coefficients=horizon_coefficients,
            maintenance_actions=maintenance_actions,
        )

    def fit(self, feature_frame: pd.DataFrame) -> None:
        """Fit one :class:`HorizonModel` per configured horizon.

        Args:
            feature_frame: Training rows with ``window_end``, every
                ``feat_*`` column and ``target_h<h>`` for each configured
                horizon.

        Raises:
            KeyError: If a configured target column is missing.

        Notes:
            The feature set is every column starting with ``feat_`` in the
            training frame and is reused verbatim at scoring time.
        """
        frame = feature_frame.sort_values("window_end").reset_index(drop=True)
        self._feature_columns = [column for column in frame.columns if column.startswith("feat_")]
        self._models = {}
        for horizon in self._config.horizons_hours:
            target_column = f"target_h{horizon}"
            self._models[horizon] = _fit_horizon_model(frame, self._feature_columns, target_column, horizon)

    def score(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        """Attach per-horizon risks to a feature frame.

        Args:
            feature_frame: Rows containing the fitted ``feat_*`` columns.

        Returns:
            Copy with ``risk_h<h>`` in ``[0, 1]`` per horizon, ``max_risk``
            and ``dominant_horizon_hours`` (horizon of the maximum risk,
            first on ties).

        Raises:
            ValueError: If the forecaster has not been fitted.
            KeyError: If a fitted feature column is missing.
        """
        if not self._models:
            raise ValueError("Forecaster must be fitted before calling score().")
        frame = feature_frame.copy()
        for horizon, model in self._models.items():
            frame[f"risk_h{horizon}"] = _score_horizon_model(frame, self._feature_columns, model)
        risk_columns = [f"risk_h{horizon}" for horizon in self._config.horizons_hours]
        frame["max_risk"] = frame[risk_columns].max(axis=1)
        frame["dominant_horizon_hours"] = frame[risk_columns].idxmax(axis=1).str.extract(r"(\d+)").astype(float)
        return frame

    def evaluate(self, *, train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> pd.DataFrame:
        """Compute binary metrics on train/test and walk-forward refits.

        Args:
            train_frame: Rows used for fitting.
            test_frame: Held-out later rows.

        Returns:
            One row per split and horizon with ``split``, ``horizon_hours``
            and the metrics of :func:`_binary_metrics`. Walk-forward rows
            (``split == "walk_forward"``) average refits on the
            concatenation of both frames.
        """
        rows: list[dict[str, Any]] = []
        for split_name, split_frame in (("train", train_frame), ("test", test_frame)):
            scored = self.score(split_frame)
            for horizon in self._config.horizons_hours:
                target_column = f"target_h{horizon}"
                risk_column = f"risk_h{horizon}"
                rows.append(
                    {
                        "split": split_name,
                        "horizon_hours": horizon,
                        **_binary_metrics(scored[target_column], scored[risk_column], self._models[horizon].decision_threshold),
                    }
                )
        walk_forward = walk_forward_validation(feature_frame=pd.concat([train_frame, test_frame], ignore_index=True), config=self._config)
        if not walk_forward.empty:
            walk_forward["split"] = "walk_forward"
            rows.extend(walk_forward.to_dict(orient="records"))
        return pd.DataFrame(rows)

    def horizon_coefficients(self) -> pd.DataFrame:
        """Tabulate fitted intercepts and weights.

        Returns:
            Rows ``horizon_hours``, ``feature`` (``"__intercept__"`` first,
            then features by decreasing absolute weight), ``weight`` and
            ``decision_threshold``. Empty when unfitted.
        """
        rows: list[dict[str, Any]] = []
        for horizon, model in self._models.items():
            rows.append(
                {
                    "horizon_hours": horizon,
                    "feature": "__intercept__",
                    "weight": model.intercept,
                    "decision_threshold": model.decision_threshold,
                }
            )
            for feature, weight in sorted(model.feature_weights.items(), key=lambda item: abs(item[1]), reverse=True):
                rows.append(
                    {
                        "horizon_hours": horizon,
                        "feature": feature,
                        "weight": weight,
                        "decision_threshold": model.decision_threshold,
                    }
                )
        return pd.DataFrame(rows)

    def export_report(self, result: AnticipationResult, output_dir: str | Path) -> dict[str, Path]:
        """Write CSV tables, an HTML report and JSON metadata.

        Args:
            result: Result to export.
            output_dir: Target directory (created if needed).

        Returns:
            Mapping from artefact label (``feature_frame``,
            ``scored_windows``, ``validation_summary``,
            ``horizon_coefficients``, ``maintenance_actions``,
            ``html_report``, ``metadata``) to written path.
        """
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for label, frame in {
            "feature_frame": result.feature_frame,
            "scored_windows": result.scored_windows,
            "validation_summary": result.validation_summary,
            "horizon_coefficients": result.horizon_coefficients,
            "maintenance_actions": result.maintenance_actions,
        }.items():
            target = path / f"{label}.csv"
            frame.to_csv(target, index=False)
            written[label] = target
        report_path = path / "anticipation_report.html"
        report_path.write_text(_build_html_report(result), encoding="utf-8")
        written["html_report"] = report_path
        metadata_path = path / "anticipation_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "config": {
                        "horizons_hours": list(result.config.horizons_hours),
                        "target_families": list(result.config.target_families),
                        "holdout_fraction": result.config.holdout_fraction,
                        "risk_threshold": result.config.risk_threshold,
                        "warning_threshold": result.config.warning_threshold,
                    },
                    "artifacts": {key: str(value) for key, value in written.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        written["metadata"] = metadata_path
        return written


def build_anticipation_frame(
    *,
    window_scores: pd.DataFrame,
    episodes: pd.DataFrame,
    horizons_hours: tuple[int, ...],
    target_families: tuple[str, ...],
) -> pd.DataFrame:
    """Build causal per-asset features and future-event targets.

    Args:
        window_scores: Scored windows (see the module data contract).
        episodes: Episodes with ``event_start``, ``event_end``,
            ``primary_family`` and optional ``asset_id``.
        horizons_hours: Target horizons in hours.
        target_families: Families counted as positive events.

    Returns:
        Frame with ``asset_id``, ``window_start``, ``window_end``,
        ``semantic_status``, ``incident_family``, the ``current_*`` context
        flags, all ``feat_*`` columns and ``target_h<h>`` per horizon. An
        empty input yields an empty frame with the key and target columns.

    Raises:
        KeyError: If ``semantic_status`` or ``incident_family`` are absent
            from ``window_scores`` (they are selected unconditionally).

    Notes:
        ``target_h<h>`` is 1.0 when a target-family episode of the same
        asset starts in ``(window_end, window_end + h]``; the window's own
        end time is excluded so only strictly future onsets count.
    """
    frame = window_scores.copy()
    if frame.empty:
        columns = ["window_start", "window_end", "asset_id"]
        for horizon in horizons_hours:
            columns.append(f"target_h{horizon}")
        return pd.DataFrame(columns=columns)
    frame["window_start"] = pd.to_datetime(frame["start_time"] if "start_time" in frame.columns else frame["window_start"])
    frame["window_end"] = pd.to_datetime(frame["end_time"] if "end_time" in frame.columns else frame["window_end"])
    frame["asset_id"] = frame.get("asset_id", pd.Series("asset-unknown", index=frame.index)).fillna("asset-unknown")
    frame = frame.sort_values(["asset_id", "window_start", "window_end"]).reset_index(drop=True)

    enriched_rows: list[pd.DataFrame] = []
    for asset_id, asset_frame in frame.groupby("asset_id", sort=False):
        enriched_rows.append(_build_asset_features(asset_frame.reset_index(drop=True)))
    enriched = pd.concat(enriched_rows, ignore_index=True)

    family_series = enriched.get("incident_family", pd.Series("unclassified_incident", index=enriched.index)).fillna("unclassified_incident")
    context_payload = pd.DataFrame(
        {
            "current_semantic_anomalous": enriched.get("semantic_status", pd.Series("NORMAL", index=enriched.index)).eq("ANOMALOUS").astype(float),
            "current_process_saturation": family_series.eq("process_saturation").astype(float),
            "current_pump_abrupt_failure": family_series.eq("pump_abrupt_failure").astype(float),
            "hour_of_day": enriched["window_start"].dt.hour.astype(float),
            "day_of_week": enriched["window_start"].dt.dayofweek.astype(float),
            "is_night_shift": ((enriched["window_start"].dt.hour < 7) | (enriched["window_start"].dt.hour >= 22)).astype(float),
        },
        index=enriched.index,
    )
    enriched = pd.concat([enriched, context_payload], axis=1).copy()

    target_episodes = episodes.copy()
    if target_episodes.empty:
        target_episodes = pd.DataFrame(columns=["event_start", "event_end", "primary_family", "asset_id"])
    else:
        target_episodes["event_start"] = pd.to_datetime(target_episodes["event_start"])
        target_episodes["event_end"] = pd.to_datetime(target_episodes["event_end"])
        target_episodes["asset_id"] = target_episodes.get("asset_id", pd.Series("asset-unknown", index=target_episodes.index)).fillna("asset-unknown")
    target_episodes = target_episodes[target_episodes.get("primary_family", pd.Series(dtype=str)).isin(target_families)].copy()

    target_payload = {f"target_h{horizon}": _build_future_target(enriched, target_episodes, horizon) for horizon in horizons_hours}
    enriched = pd.concat([enriched, pd.DataFrame(target_payload, index=enriched.index)], axis=1).copy()

    keep_columns = [
        "asset_id",
        "window_start",
        "window_end",
        "semantic_status",
        "incident_family",
        "current_semantic_anomalous",
        "current_process_saturation",
        "current_pump_abrupt_failure",
    ]
    keep_columns += [column for column in enriched.columns if column.startswith("feat_")]
    keep_columns += [f"target_h{horizon}" for horizon in horizons_hours]
    return enriched.loc[:, keep_columns].reset_index(drop=True)


def temporal_holdout_split(feature_frame: pd.DataFrame, holdout_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows chronologically by ``window_end``.

    Args:
        feature_frame: Feature rows.
        holdout_fraction: Fraction of latest rows assigned to the test set.

    Returns:
        ``(train, test)`` copies. With at least two rows, both parts are
        non-empty; a single row goes entirely to train.

    Notes:
        No purge gap is applied between the parts (see module notes).
    """
    if feature_frame.empty:
        return feature_frame.copy(), feature_frame.copy()
    ordered = feature_frame.sort_values("window_end").reset_index(drop=True)
    split_index = max(int(len(ordered) * (1.0 - holdout_fraction)), 1)
    split_index = min(split_index, len(ordered) - 1) if len(ordered) > 1 else len(ordered)
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def walk_forward_validation(*, feature_frame: pd.DataFrame, config: AnticipationConfig, min_train_fraction: float = 0.5, steps: int = 3) -> pd.DataFrame:
    """Average expanding-window refit metrics over several chronological splits.

    Split indices are ``linspace(max(int(n * min_train_fraction), 4), n - 2,
    steps)`` (deduplicated); for each split a fresh forecaster is fitted on
    the earlier rows and evaluated on all later rows.

    Args:
        feature_frame: Feature rows with targets.
        config: Forecaster configuration.
        min_train_fraction: Fraction of rows in the first training set.
        steps: Number of candidate split points.

    Returns:
        Mean metrics per ``horizon_hours``; an empty frame with the metric
        columns when fewer than 12 rows are available.
    """
    if feature_frame.empty or len(feature_frame) < 12:
        return pd.DataFrame(columns=["horizon_hours", "precision", "recall", "f1", "positive_rate", "alert_rate", "brier_score"])
    ordered = feature_frame.sort_values("window_end").reset_index(drop=True)
    start_index = max(int(len(ordered) * min_train_fraction), 4)
    candidate_splits = np.linspace(start_index, len(ordered) - 2, num=steps, dtype=int)
    rows: list[dict[str, Any]] = []
    for split_index in sorted(set(int(item) for item in candidate_splits if item > 1)):
        train = ordered.iloc[:split_index].copy()
        test = ordered.iloc[split_index:].copy()
        if train.empty or test.empty:
            continue
        runner = AnticipatoryMaintenanceForecaster(config)
        runner.fit(train)
        scored = runner.score(test)
        for horizon in config.horizons_hours:
            target_column = f"target_h{horizon}"
            risk_column = f"risk_h{horizon}"
            metrics = _binary_metrics(scored[target_column], scored[risk_column], runner._models[horizon].decision_threshold)
            rows.append({"horizon_hours": horizon, **metrics})
    if not rows:
        return pd.DataFrame(columns=["horizon_hours", "precision", "recall", "f1", "positive_rate", "alert_rate", "brier_score"])
    summary = pd.DataFrame(rows).groupby("horizon_hours", dropna=False).mean(numeric_only=True).reset_index()
    return summary


def build_maintenance_actions(scored_windows: pd.DataFrame, config: AnticipationConfig) -> pd.DataFrame:
    """Map the latest risk of each asset to an action and priority.

    Args:
        scored_windows: Output of :meth:`AnticipatoryMaintenanceForecaster.score`.
        config: Supplies ``risk_threshold`` and ``warning_threshold``.

    Returns:
        One row per asset (latest ``window_end``) with ``asset_id``,
        ``window_start``, ``window_end``, ``max_risk``,
        ``dominant_horizon_hours``, ``risk_h*``, ``maintenance_action`` and
        ``maintenance_priority``.
    """
    if scored_windows.empty:
        return pd.DataFrame(columns=["asset_id", "window_end", "max_risk", "dominant_horizon_hours", "maintenance_action", "maintenance_priority"])
    frame = scored_windows.copy()
    latest = frame.sort_values(["asset_id", "window_end"]).groupby("asset_id", dropna=False).tail(1).copy()
    latest["maintenance_action"] = latest.apply(lambda row: _action_label(row["max_risk"], config), axis=1)
    latest["maintenance_priority"] = latest.apply(lambda row: _priority_label(row["max_risk"], config), axis=1)
    action_columns = [column for column in latest.columns if column.startswith("risk_h")]
    return latest[["asset_id", "window_start", "window_end", "max_risk", "dominant_horizon_hours", *action_columns, "maintenance_action", "maintenance_priority"]].reset_index(drop=True)


def _build_asset_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create causal rolling features for one asset's time-ordered windows.

    For each column of :data:`DEFAULT_FEATURE_COLUMNS` (missing -> 0, NaN
    -> 0): the current value, its one-row difference and rolling mean/max/min
    over the previous 6 and 24 rows (``min_periods=1``). Also the semantic
    anomaly rate over 6 and 24 rows and its one-row difference.

    Args:
        frame: Windows of a single asset sorted by time.

    Returns:
        ``frame`` with the ``feat_*`` columns appended.
    """
    asset = frame.copy()
    feature_payload: dict[str, pd.Series] = {}
    for column in DEFAULT_FEATURE_COLUMNS:
        values = pd.to_numeric(asset[column], errors="coerce").fillna(0.0) if column in asset.columns else pd.Series(0.0, index=asset.index, dtype=float)
        feature_payload[f"feat_{column}"] = values
        feature_payload[f"feat_{column}_delta_1h"] = values.diff().fillna(0.0)
        for lookback in ROLLING_WINDOWS:
            rolling = values.rolling(window=lookback, min_periods=1)
            feature_payload[f"feat_{column}_mean_{lookback}h"] = rolling.mean()
            feature_payload[f"feat_{column}_max_{lookback}h"] = rolling.max()
            feature_payload[f"feat_{column}_min_{lookback}h"] = rolling.min()
    semantic = asset.get("semantic_status", pd.Series("NORMAL", index=asset.index)).eq("ANOMALOUS").astype(float)
    feature_payload["feat_semantic_anomaly_rate_6h"] = semantic.rolling(window=6, min_periods=1).mean()
    feature_payload["feat_semantic_anomaly_rate_24h"] = semantic.rolling(window=24, min_periods=1).mean()
    feature_payload["feat_semantic_anomaly_delta_1h"] = semantic.diff().fillna(0.0)
    return pd.concat([asset, pd.DataFrame(feature_payload, index=asset.index)], axis=1).copy()


def _build_future_target(frame: pd.DataFrame, episodes: pd.DataFrame, horizon_hours: int) -> pd.Series:
    """Label windows followed by a target episode onset within the horizon.

    Args:
        frame: Windows with ``window_end`` and ``asset_id`` and a default
            ``RangeIndex`` (positions are used for assignment).
        episodes: Target-family episodes with ``event_start``/``asset_id``.
        horizon_hours: Horizon in hours.

    Returns:
        Float series of 0/1 labels: 1.0 when an episode of the same asset
        starts in ``(window_end, window_end + horizon]``.
    """
    labels = pd.Series(0.0, index=frame.index, dtype=float)
    if episodes.empty:
        return labels
    horizon = pd.Timedelta(hours=horizon_hours)
    for idx, row in frame.iterrows():
        future_start = row["window_end"]
        future_end = future_start + horizon
        asset_episodes = episodes[episodes["asset_id"] == row["asset_id"]]
        hit = asset_episodes[asset_episodes["event_start"].gt(future_start) & asset_episodes["event_start"].le(future_end)]
        labels.iloc[idx] = 1.0 if not hit.empty else 0.0
    return labels


def _fit_horizon_model(frame: pd.DataFrame, feature_columns: list[str], target_column: str, horizon: int) -> HorizonModel:
    r"""Fit the standardized-mean-difference logistic score for one horizon.

    .. math::

       b = \operatorname{logit}(\operatorname{clip}(\pi, 10^{-4}, 1-10^{-4})),
       \qquad w_j = (\mu_j^{+} - \mu_j^{-}) / \sigma_j.

    Args:
        frame: Training rows.
        feature_columns: ``feat_*`` columns.
        target_column: ``target_h<h>`` column (values > 0.5 are positive).
        horizon: Horizon in hours.

    Returns:
        The fitted :class:`HorizonModel`.

    Notes:
        If one class is absent, its class mean defaults to the overall mean,
        giving zero weights. Features are not decorrelated, so correlated
        rolling variants of one indicator add up their weights.
    """
    target = pd.to_numeric(frame[target_column], errors="coerce").fillna(0.0)
    positive = frame[target.gt(0.5)]
    negative = frame[target.le(0.5)]
    positive_rate = float(target.mean()) if not target.empty else 0.0
    intercept = _logit(np.clip(positive_rate, 1e-4, 1 - 1e-4))
    feature_weights: dict[str, float] = {}
    feature_means: dict[str, float] = {}
    feature_stds: dict[str, float] = {}
    for feature in feature_columns:
        values = pd.to_numeric(frame[feature], errors="coerce").fillna(0.0)
        std = float(values.std(ddof=0)) or 1.0
        mean = float(values.mean())
        feature_means[feature] = mean
        feature_stds[feature] = std
        pos_mean = float(pd.to_numeric(positive.get(feature, pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean()) if not positive.empty else mean
        neg_mean = float(pd.to_numeric(negative.get(feature, pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean()) if not negative.empty else mean
        feature_weights[feature] = (pos_mean - neg_mean) / std
    train_scores = _score_horizon_model(frame, feature_columns, HorizonModel(
        horizon_hours=horizon,
        intercept=intercept,
        feature_weights=feature_weights,
        feature_means=feature_means,
        feature_stds=feature_stds,
        positive_rate=positive_rate,
        decision_threshold=0.5,
    ))
    decision_threshold = _select_decision_threshold(target, train_scores, horizon)
    return HorizonModel(
        horizon_hours=horizon,
        intercept=intercept,
        feature_weights=feature_weights,
        feature_means=feature_means,
        feature_stds=feature_stds,
        positive_rate=positive_rate,
        decision_threshold=decision_threshold,
    )


def _score_horizon_model(frame: pd.DataFrame, feature_columns: list[str], model: HorizonModel) -> pd.Series:
    """Evaluate a :class:`HorizonModel` on feature rows.

    Args:
        frame: Rows containing ``feature_columns``.
        feature_columns: Feature names; missing values are filled with the
            training mean (zero contribution).
        model: Fitted model.

    Returns:
        Risk probabilities in ``[0, 1]`` aligned with ``frame.index``.
    """
    raw = pd.Series(model.intercept, index=frame.index, dtype=float)
    for feature in feature_columns:
        values = pd.to_numeric(frame[feature], errors="coerce").fillna(model.feature_means.get(feature, 0.0))
        std = model.feature_stds.get(feature, 1.0) or 1.0
        centered = (values - model.feature_means.get(feature, 0.0)) / std
        raw = raw + model.feature_weights.get(feature, 0.0) * centered
    return raw.map(_sigmoid)


def _select_decision_threshold(target: pd.Series, risk: pd.Series, horizon_hours: int) -> float:
    """Select a risk threshold on training data.

    Scans 29 thresholds in ``[0.15, 0.85]``, skipping those whose alert rate
    falls below 5 % (horizons <= 24 h) or 2 % (longer horizons), and keeps the
    lexicographic maximum of ``(F1, recall, -|alert_rate - positive_rate|)``.

    Args:
        target: Training labels.
        risk: Training risks.
        horizon_hours: Horizon in hours (selects the minimum alert rate).

    Returns:
        The selected threshold; 0.5 when no candidate qualifies or the
        target is empty.
    """
    y_true = pd.to_numeric(target, errors="coerce").fillna(0.0)
    y_score = pd.to_numeric(risk, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if y_true.empty:
        return 0.5
    min_alert_rate = 0.05 if horizon_hours <= 24 else 0.02
    candidate_thresholds = np.linspace(0.15, 0.85, 29)
    best_threshold = 0.5
    best_score = (-1.0, -1.0, -1.0)
    for threshold in candidate_thresholds:
        metrics = _binary_metrics(y_true, y_score, float(threshold))
        if metrics["alert_rate"] < min_alert_rate:
            continue
        score = (metrics["f1"], metrics["recall"], -abs(metrics["alert_rate"] - metrics["positive_rate"]))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _binary_metrics(target: pd.Series, risk: pd.Series, threshold: float) -> dict[str, float]:
    """Compute thresholded classification metrics and the Brier score.

    Args:
        target: Labels (values > 0.5 are positive; NaN -> negative).
        risk: Risk scores, clipped to ``[0, 1]`` (NaN -> 0).
        threshold: Alert when ``risk >= threshold``.

    Returns:
        ``precision``, ``recall``, ``f1``, ``positive_rate``,
        ``alert_rate`` and ``brier_score``. Precision/recall use a floored
        denominator of 1, so they are 0.0 (not NaN) without alerts or
        positives.
    """
    y_true = pd.to_numeric(target, errors="coerce").fillna(0.0).gt(0.5)
    y_score = pd.to_numeric(risk, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    y_pred = y_score.ge(threshold)
    tp = float((y_true & y_pred).sum())
    fp = float((~y_true & y_pred).sum())
    fn = float((y_true & ~y_pred).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    brier = float(((y_score - y_true.astype(float)) ** 2).mean()) if len(y_true) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "alert_rate": float(y_pred.mean()) if len(y_pred) else 0.0,
        "brier_score": brier,
    }


def _action_label(max_risk: float, config: AnticipationConfig) -> str:
    """Map a risk to a maintenance action label.

    Args:
        max_risk: Maximum risk over horizons.
        config: Supplies the risk and warning thresholds.

    Returns:
        ``"inspect_and_prepare_intervention"``,
        ``"increase_monitoring_and_schedule_check"`` or
        ``"routine_follow_up"``.
    """
    if max_risk >= config.risk_threshold:
        return "inspect_and_prepare_intervention"
    if max_risk >= config.warning_threshold:
        return "increase_monitoring_and_schedule_check"
    return "routine_follow_up"


def _priority_label(max_risk: float, config: AnticipationConfig) -> str:
    """Map a risk to a priority label.

    Args:
        max_risk: Maximum risk over horizons.
        config: Supplies the risk and warning thresholds.

    Returns:
        ``"high"``, ``"medium"`` or ``"low"``.
    """
    if max_risk >= config.risk_threshold:
        return "high"
    if max_risk >= config.warning_threshold:
        return "medium"
    return "low"


def _sigmoid(value: float) -> float:
    r"""Numerically stable logistic function :math:`1/(1+e^{-x})`.

    Args:
        value: Real input.

    Returns:
        Value in ``(0, 1)``; the branch on the sign avoids overflow.
    """
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _logit(value: float) -> float:
    r"""Log-odds :math:`\ln(p/(1-p))` with ``p`` clipped to ``[1e-6, 1-1e-6]``.

    Args:
        value: Probability.

    Returns:
        Log-odds.
    """
    value = min(max(float(value), 1e-6), 1 - 1e-6)
    return math.log(value / (1.0 - value))


def _build_html_report(result: AnticipationResult) -> str:
    """Render a self-contained HTML summary of an anticipation run.

    Args:
        result: Result to render.

    Returns:
        HTML document with the validation summary, recommended actions and
        the 12 first coefficient rows (intercept plus top features) per
        horizon.
    """
    summary_html = result.validation_summary.to_html(index=False, float_format=lambda value: f"{value:.3f}")
    actions_html = result.maintenance_actions.to_html(index=False, float_format=lambda value: f"{value:.3f}")
    coefficients = result.horizon_coefficients.copy()
    coeff_sections: list[str] = []
    if not coefficients.empty:
        for horizon, horizon_frame in coefficients.groupby("horizon_hours", sort=False):
            coeff_sections.append(f"<h3>Horizon {int(horizon)} h</h3>")
            coeff_sections.append(horizon_frame.head(12).to_html(index=False, float_format=lambda value: f"{value:.3f}"))
    return "\n".join(
        [
            "<html><head><meta charset='utf-8'><title>Anticipatory Maintenance Report</title>",
            "<style>body{font-family:Arial,sans-serif;margin:24px;}table{border-collapse:collapse;margin:16px 0;}th,td{border:1px solid #ccc;padding:6px 10px;}h1,h2,h3{margin-top:24px;}</style></head><body>",
            "<h1>Anticipatory Maintenance Experimental Report</h1>",
            f"<p>Horizons: {', '.join(str(item) + ' h' for item in result.config.horizons_hours)} | Target families: {', '.join(html.escape(item) for item in result.config.target_families)}</p>",
            "<h2>Validation summary</h2>",
            summary_html,
            "<h2>Recommended maintenance actions</h2>",
            actions_html,
            "<h2>Most influential features by horizon</h2>",
            *coeff_sections,
            "</body></html>",
        ]
    )
