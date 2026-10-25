r"""Rolling-window construction and deviation scoring for Model_D.

Role in the pipeline:
    Converts Model_B/Model_C sequence rows (each a timed behavioural
    sequence with optional indicator values, Model_C semantic status and
    incident family) into fixed-length rolling analysis windows, summarizes
    the state-word regime inside each window, applies a narrow rule-based
    fallback to flag stable localized regime shifts, and attaches a composite
    ``deviation_score`` consumed by the episode builder
    (:mod:`iabm_incidents.episodes`).

Main public objects:
    * :data:`INDICATOR_COLUMNS` -- indicator column -> weight attribute name.
    * :class:`WindowIndicators` -- typed record of one window profile.
    * :func:`compute_deviation_score` -- weighted composite score.
    * :func:`build_window_scores` -- entry point; accepts raw timed rows or
      pre-windowed tables.

Data contract:
    Input rows (``sequences`` and ``assignments`` are aligned row-wise and
    concatenated column-wise) may contain:

    * ``start_time`` / ``end_time`` (or aliases ``window_start`` /
      ``window_end``): interval bounds, parsed as UTC and converted to naive
      UTC timestamps. Windows are half-open ``[start, start + length)``.
    * Indicator columns from :data:`INDICATOR_COLUMNS` (numeric).
    * ``semantic_status`` (``"ANOMALOUS"`` or ``"NORMAL"``),
      ``incident_family`` (string), ``asset_id`` (string).
    * ``states`` (state word, e.g. ``"(1, 17, 3)"`` or a tuple/list),
      ``nominal_match`` (nearest nominal word), ``state_distance``,
      ``dtw_distance``, ``anomaly_score`` (numeric).
    * ``run_count`` / ``active_sequence_count``, ``sequence_count``,
      ``data_coverage`` for pre-windowed tables.

    The output has one row per emitted window with ``asset_id``,
    ``start_time``, ``end_time``, ``data_coverage``, ``sequence_count``,
    ``active_sequence_count``, ``semantic_status``, ``incident_family``, the
    available indicator columns, the state-word summary columns and
    ``deviation_score``, sorted by ``start_time``.

Composite score:
    .. math::

       S_w = \sum_{k \in \text{indicators}} \omega_k \, x_{w,k}
       \quad (\text{missing } x_{w,k} := 0).

    Without any indicator column, the score is the binary
    :math:`\max(\mathbb{1}[\text{ANOMALOUS}], \mathbb{1}[\text{fallback family}])`.

.. mermaid::

   flowchart TD
       A[sequences + assignments] --> N[_normalize_window_input]
       N --> P{_is_prewindowed?}
       P -- yes --> F[_finalize_window_frame]
       P -- no --> B[_build_rolling_windows]
       B --> BL[_fit_behavioral_baseline]
       B --> I{indicator columns?}
       I -- no --> SF[_build_semantic_windows_fast]
       I -- yes --> SW[_summarize_window per window]
       SF --> W[_summarize_state_words + _infer_fallback_family]
       SW --> W
       W --> F
       F --> S[compute_deviation_score]

Maintenance notes:
    The state-word baseline (:func:`_fit_behavioral_baseline`) is fitted on
    the same frame that is being windowed, i.e. over all phases supplied by
    the caller; rare-word thresholds and word frequencies therefore are not
    chronologically frozen. Callers that need a leakage-free evaluation
    should window reference and evaluation partitions with that in mind, or
    use the matched ablation, which rescales indicators on the reference
    phase only. ``state_17_fraction`` hard-codes controller state 17 as a
    known saturation-related state. Missing indicator values become 0 both
    in weighted window means and in the composite score.
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
import math

import numpy as np
import pandas as pd

from .config import DeviationWeights, WindowConfig


INDICATOR_COLUMNS = {
    "sequence_divergence": "sequence",
    "duration_drift": "duration",
    "recurrence_excess": "recurrence",
    "persistence_excess": "persistence",
    "consumption_deviation": "consumption",
    "state_error_rate": "state_error",
    "mode_divergence": "mode",
    "command_response_deviation": "command_response",
    "transient_concentration": "transient_concentration",
}
PREWINDOWED_COLUMNS = {"window_start", "window_end", "data_coverage", "sequence_count", "active_sequence_count"}
TIME_COLUMN_ALIASES = {
    "window_start": "start_time",
    "window_end": "end_time",
}
COUNT_COLUMN_ALIASES = {
    "active_sequence_count": "run_count",
}


@dataclass(frozen=True)
class WindowIndicators:
    """Container describing a single window-level anomaly profile.

    Attributes:
        start_time: Inclusive window start.
        end_time: Exclusive window end.
        sequence_divergence: Behavioural sequence divergence.
        duration_drift: Robust dwell-duration drift.
        recurrence_excess: Positive run-count excess.
        persistence_excess: Positive occupancy excess.
        consumption_deviation: Signed electrical consumption deviation.
        state_error_rate: Disagreement rate between commanded and predicted
            states.
        mode_divergence: Operating-mode distribution divergence.
        command_response_deviation: Mean command-response residual norm.
        transient_concentration: Fourth-order transient concentration
            (Tconc).
    """

    start_time: pd.Timestamp
    end_time: pd.Timestamp
    sequence_divergence: float
    duration_drift: float
    recurrence_excess: float
    persistence_excess: float
    consumption_deviation: float
    state_error_rate: float
    mode_divergence: float
    command_response_deviation: float = 0.0
    transient_concentration: float = 0.0


def compute_deviation_score(frame: pd.DataFrame, weights: DeviationWeights) -> pd.Series:
    r"""Compute a composite deviation score from available indicators.

    If at least one column of :data:`INDICATOR_COLUMNS` exists, the score is
    :math:`\sum_k \omega_k x_k` over all nine indicators, where absent
    columns and non-numeric/missing values contribute 0. Otherwise, if
    ``semantic_status`` exists, the score is 1.0 for ``"ANOMALOUS"`` rows
    and, when ``word_regime_shift_score`` is present, also 1.0 for rows that
    :func:`_infer_fallback_family` would flag; else 0.0.

    Args:
        frame: Window table.
        weights: Indicator weights.

    Returns:
        Float series aligned with ``frame.index``.

    Raises:
        ValueError: If neither ``semantic_status`` nor any indicator column
            is present.
    """

    if any(column in frame.columns for column in INDICATOR_COLUMNS):
        score = pd.Series(0.0, index=frame.index, dtype=float)
        for column, weight_name in INDICATOR_COLUMNS.items():
            values = pd.to_numeric(
                frame[column] if column in frame.columns else pd.Series(0.0, index=frame.index),
                errors="coerce",
            ).fillna(0.0)
            score = score + getattr(weights, weight_name) * values
        return score

    if "semantic_status" in frame.columns:
        semantic_score = frame["semantic_status"].eq("ANOMALOUS").astype(float)
        fallback_score = pd.Series(0.0, index=frame.index, dtype=float)
        if "word_regime_shift_score" in frame.columns:
            fallback_score = frame.apply(
                lambda row: float(_infer_fallback_family(row.to_dict()) is not None),
                axis=1,
            )
        return pd.concat([semantic_score, fallback_score], axis=1).max(axis=1)

    raise ValueError(
        "Unable to compute deviation score: expected semantic_status or at least one deviation indicator column."
    )


def build_window_scores(
    sequences: pd.DataFrame,
    assignments: pd.DataFrame,
    weights: DeviationWeights,
    window_config: WindowConfig,
) -> pd.DataFrame:
    """Build scored windows from raw timed rows or accept pre-windowed inputs.

    Args:
        sequences: Timed sequence rows or an already windowed table.
        assignments: Row-aligned semantic assignments (e.g. Model_C
            ``semantic_status``/``incident_family``); concatenated with
            ``sequences`` by position after resetting both indices.
        weights: Indicator weights for :func:`compute_deviation_score`.
        window_config: Rolling-window policy (used only for raw rows).

    Returns:
        Scored window table (see the module data contract).

    Notes:
        Inputs are considered pre-windowed when any of
        :data:`PREWINDOWED_COLUMNS` is present in ``sequences``; then
        assignment columns are only added when missing and no rolling is
        performed. For raw rows, overlapping assignment columns are dropped
        except ``semantic_status``, ``incident_family`` and
        ``anomaly_score``; for those protected names the ``sequences``
        column wins because duplicated labels keep their first occurrence.
    """

    sequence_frame = _normalize_window_input(sequences).reset_index(drop=True).copy()
    assignment_frame = _normalize_window_input(assignments).reset_index(drop=True).copy()

    if _is_prewindowed(sequence_frame):
        merged = sequence_frame.copy()
        for column in assignment_frame.columns:
            if column not in merged.columns:
                merged[column] = assignment_frame[column]
        return _finalize_window_frame(merged, weights)

    overlapping = [column for column in assignment_frame.columns if column in sequence_frame.columns]
    protected = {'semantic_status', 'incident_family', 'anomaly_score'}
    drop_from_assignment = [column for column in overlapping if column not in protected]
    if drop_from_assignment:
        assignment_frame = assignment_frame.drop(columns=drop_from_assignment)

    merged = pd.concat([sequence_frame, assignment_frame], axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()].copy()
    return _build_rolling_windows(merged, weights, window_config)


def _build_rolling_windows(
    merged: pd.DataFrame,
    weights: DeviationWeights,
    window_config: WindowConfig,
) -> pd.DataFrame:
    """Construct explicit rolling windows from timed sequence rows.

    Window starts run from ``floor(min(start_time), step)`` while they do not
    exceed ``max(end_time)``. A row belongs to a window when its interval
    intersects ``[start, start + length)``.

    Args:
        merged: Concatenated sequence/assignment rows.
        weights: Indicator weights.
        window_config: Rolling-window policy.

    Returns:
        Finalized and scored window table.

    Notes:
        Rows with unparsable ``start_time``/``end_time`` are dropped. When no
        indicator column exists, the vectorized semantic path
        :func:`_build_semantic_windows_fast` is used. Windows are built
        across all assets jointly: rows of different assets may share a
        window, and the reported ``asset_id`` is that of the first active
        row.
    """

    frame = merged.loc[:, ~merged.columns.duplicated()].copy()
    frame["start_time"] = pd.to_datetime(frame["start_time"], errors="coerce", utc=True).dt.tz_convert(None)
    frame["end_time"] = pd.to_datetime(frame["end_time"], errors="coerce", utc=True).dt.tz_convert(None)
    frame = frame.dropna(subset=["start_time", "end_time"]).reset_index(drop=True)
    if frame.empty:
        return _finalize_window_frame(frame, weights)

    baseline_profiles = _fit_behavioral_baseline(frame)

    if not any(column in frame.columns for column in INDICATOR_COLUMNS):
        return _finalize_window_frame(_build_semantic_windows_fast(frame, window_config, baseline_profiles), weights)

    min_start = frame["start_time"].min().floor(window_config.step)
    max_end = frame["end_time"].max()
    rows: list[dict[str, object]] = []
    current_start = min_start
    while current_start <= max_end:
        current_end = current_start + window_config.length
        window_rows = frame[(frame["start_time"] < current_end) & (frame["end_time"] > current_start)]
        window_payload = _summarize_window(window_rows, current_start, current_end, window_config, baseline_profiles)
        if window_payload is not None:
            rows.append(window_payload)
        current_start = current_start + window_config.step

    return _finalize_window_frame(pd.DataFrame(rows), weights)


def _build_semantic_windows_fast(
    frame: pd.DataFrame,
    window_config: WindowConfig,
    baseline_profiles: dict[str, object],
) -> pd.DataFrame:
    """Build semantic-only windows using array operations instead of DataFrame slicing.

    Args:
        frame: Timed rows with parsed ``start_time``/``end_time``.
        window_config: Rolling-window policy.
        baseline_profiles: State-word profiles from
            :func:`_fit_behavioral_baseline`.

    Returns:
        Unscored window table (one row per supported window); empty frame
        when ``frame`` is empty.

    Notes:
        Coverage is ``min(sum(overlap_seconds) / length_seconds, 1)``.
        Windows failing ``min_coverage`` or ``min_active_sequences`` are
        skipped. The window is ``ANOMALOUS`` when any active row is; its
        family is the mode over anomalous rows (all active rows otherwise).
        A ``NORMAL`` window can be promoted to ``ANOMALOUS`` by
        :func:`_infer_fallback_family`.
    """

    if frame.empty:
        return pd.DataFrame()

    starts = frame["start_time"].to_numpy(dtype="datetime64[ns]")
    ends = frame["end_time"].to_numpy(dtype="datetime64[ns]")
    semantic_values = (
        frame["semantic_status"].fillna("NORMAL").astype(str).to_numpy()
        if "semantic_status" in frame.columns
        else np.full(len(frame), "NORMAL", dtype=object)
    )
    family_values = (
        frame["incident_family"].fillna("unclassified_incident").astype(str).to_numpy()
        if "incident_family" in frame.columns
        else np.full(len(frame), "unclassified_incident", dtype=object)
    )
    asset_values = (
        frame["asset_id"].fillna("asset-unknown").astype(str).to_numpy()
        if "asset_id" in frame.columns
        else np.full(len(frame), "asset-unknown", dtype=object)
    )

    starts_ns = starts.astype("int64")
    ends_ns = ends.astype("int64")
    length_seconds = window_config.length.total_seconds()
    step = window_config.step
    min_start = frame["start_time"].min().floor(step)
    max_end = frame["end_time"].max()

    rows: list[dict[str, object]] = []
    current_start = min_start
    while current_start <= max_end:
        current_end = current_start + window_config.length
        current_start_ns = current_start.to_datetime64().astype("datetime64[ns]").astype("int64")
        current_end_ns = current_end.to_datetime64().astype("datetime64[ns]").astype("int64")
        mask = (starts_ns < current_end_ns) & (ends_ns > current_start_ns)
        if not bool(mask.any()):
            current_start = current_start + step
            continue

        overlap_start = np.maximum(starts_ns[mask], current_start_ns)
        overlap_end = np.minimum(ends_ns[mask], current_end_ns)
        overlap_seconds = np.maximum((overlap_end - overlap_start) / 1_000_000_000.0, 0.0)
        active_mask = overlap_seconds > 0.0
        if not bool(active_mask.any()):
            current_start = current_start + step
            continue

        overlap_seconds = overlap_seconds[active_mask]
        coverage = min(float(overlap_seconds.sum()) / length_seconds, 1.0)
        if coverage < window_config.min_coverage:
            current_start = current_start + step
            continue
        active_count = int(active_mask.sum())
        if active_count < window_config.min_active_sequences:
            current_start = current_start + step
            continue

        active_semantic = semantic_values[mask][active_mask]
        active_family = family_values[mask][active_mask]
        anomalous_mask = active_semantic == "ANOMALOUS"
        family_source = active_family[anomalous_mask] if anomalous_mask.any() else active_family
        if family_source.size:
            family_mode = pd.Series(family_source).mode(dropna=True)
            dominant_family = str(family_mode.iloc[0]) if not family_mode.empty else "unclassified_incident"
        else:
            dominant_family = "unclassified_incident"

        semantic_window = {
                "asset_id": str(asset_values[mask][active_mask][0]) if active_count else None,
                "start_time": current_start,
                "end_time": current_end,
                "data_coverage": coverage,
                "sequence_count": active_count,
                "active_sequence_count": active_count,
                "semantic_status": "ANOMALOUS" if anomalous_mask.any() else "NORMAL",
                "incident_family": dominant_family,
            }
        active_rows = frame.loc[mask].iloc[np.flatnonzero(active_mask)].copy()
        word_summary = _summarize_state_words(active_rows, baseline_profiles)
        semantic_window.update(word_summary)
        fallback_family = _infer_fallback_family(semantic_window)
        if semantic_window["semantic_status"] == "NORMAL" and fallback_family is not None:
            semantic_window["semantic_status"] = "ANOMALOUS"
            semantic_window["incident_family"] = fallback_family
        rows.append(semantic_window)
        current_start = current_start + step

    return pd.DataFrame(rows)


def _summarize_window(
    window_rows: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    window_config: WindowConfig,
    baseline_profiles: dict[str, object],
) -> dict[str, object] | None:
    """Summarize a single rolling window.

    Args:
        window_rows: Rows whose interval intersects the window.
        window_start: Inclusive window start.
        window_end: Exclusive window end.
        window_config: Rolling-window policy (coverage and activity gates).
        baseline_profiles: State-word profiles.

    Returns:
        Window payload dictionary, or ``None`` when no row has positive
        overlap or a support gate fails.

    Notes:
        Indicators are overlap-duration weighted means
        (:func:`_weighted_mean`), except ``transient_concentration``, which
        keeps the maximum over contributing rows so that a short strong
        transient is not diluted. Semantic status/family follow the same
        rules (and fallback promotion) as
        :func:`_build_semantic_windows_fast`.
    """

    if window_rows.empty:
        return None

    start_clip = window_rows["start_time"].where(window_rows["start_time"] > window_start, window_start)
    end_clip = window_rows["end_time"].where(window_rows["end_time"] < window_end, window_end)
    overlap_seconds = (end_clip - start_clip).dt.total_seconds().clip(lower=0.0)
    active_mask = overlap_seconds.gt(0)
    if not bool(active_mask.any()):
        return None

    active_rows = window_rows.loc[active_mask]
    overlap_seconds = overlap_seconds.loc[active_mask]
    coverage = min(float(overlap_seconds.sum()) / window_config.length.total_seconds(), 1.0)
    if coverage < window_config.min_coverage:
        return None
    if len(active_rows) < window_config.min_active_sequences:
        return None

    weighted = {
        column: _weighted_mean(active_rows[column], overlap_seconds)
        for column in INDICATOR_COLUMNS
        if column in active_rows.columns
    }

    # Precomputed transient subwindows retain their strongest concentration;
    # command-response magnitudes retain duration-weighted aggregation.
    if "transient_concentration" in active_rows:
        weighted["transient_concentration"] = float(pd.to_numeric(active_rows["transient_concentration"], errors="coerce").max())

    if "semantic_status" in active_rows.columns:
        semantic_series = active_rows["semantic_status"].fillna("NORMAL")
    else:
        semantic_series = pd.Series("NORMAL", index=active_rows.index)
    is_anomalous = semantic_series.eq("ANOMALOUS")

    dominant_family = None
    if "incident_family" in active_rows.columns:
        family_source = active_rows.loc[is_anomalous, "incident_family"] if bool(is_anomalous.any()) else active_rows["incident_family"]
        family_mode = family_source.dropna().mode()
        dominant_family = family_mode.iloc[0] if not family_mode.empty else None

    word_summary = _summarize_state_words(active_rows, baseline_profiles)
    semantic_status = "ANOMALOUS" if bool(is_anomalous.any()) else "NORMAL"
    incident_family = dominant_family or "unclassified_incident"
    fallback_context = {
        **weighted,
        **word_summary,
        "semantic_status": semantic_status,
        "incident_family": incident_family,
    }
    fallback_family = _infer_fallback_family(fallback_context)
    if semantic_status == "NORMAL" and fallback_family is not None:
        semantic_status = "ANOMALOUS"
        incident_family = fallback_family

    return {
        "asset_id": active_rows["asset_id"].iloc[0] if "asset_id" in active_rows.columns else None,
        "start_time": window_start,
        "end_time": window_end,
        "data_coverage": coverage,
        "sequence_count": int(len(active_rows)),
        "active_sequence_count": int(len(active_rows)),
        "semantic_status": semantic_status,
        "incident_family": incident_family,
        **weighted,
        **word_summary,
    }


def _finalize_window_frame(frame: pd.DataFrame, weights: DeviationWeights) -> pd.DataFrame:
    """Normalize and score a window frame regardless of its source.

    Args:
        frame: Window rows from the rolling builders or a pre-windowed
            input.
        weights: Indicator weights.

    Returns:
        Copy with parsed ``start_time``/``end_time``, numeric
        ``data_coverage`` (missing -> 1.0), integer ``sequence_count`` and
        ``active_sequence_count`` (missing -> 1), ``deviation_score``, sorted
        by ``start_time`` with a fresh index. An empty input returns an empty
        frame with the canonical column set.

    Notes:
        ``active_sequence_count`` is taken from ``run_count`` when that
        column exists and otherwise defaults to 1; an existing
        ``active_sequence_count`` column without ``run_count`` (as produced
        by the rolling builders) is overwritten with 1.
    """

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "asset_id",
                "start_time",
                "end_time",
                "data_coverage",
                "sequence_count",
                "active_sequence_count",
                "semantic_status",
                "incident_family",
                *INDICATOR_COLUMNS.keys(),
                "dominant_state_word",
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
                "deviation_score",
            ]
        )

    window_frame = frame.copy()
    window_frame["start_time"] = pd.to_datetime(window_frame["start_time"])
    window_frame["end_time"] = pd.to_datetime(window_frame["end_time"])
    # Note: ``active_sequence_count`` is sourced from ``run_count`` only (the
    # alias created by _normalize_window_input); otherwise it defaults to 1.
    data_coverage = window_frame["data_coverage"] if "data_coverage" in window_frame.columns else pd.Series(1.0, index=window_frame.index)
    active_sequence_count = window_frame["run_count"] if "run_count" in window_frame.columns else pd.Series(1, index=window_frame.index)
    sequence_count = window_frame["sequence_count"] if "sequence_count" in window_frame.columns else pd.Series(1, index=window_frame.index)
    window_frame["data_coverage"] = pd.to_numeric(data_coverage, errors="coerce").fillna(1.0)
    window_frame["sequence_count"] = pd.to_numeric(sequence_count, errors="coerce").fillna(1).astype(int)
    window_frame["active_sequence_count"] = pd.to_numeric(active_sequence_count, errors="coerce").fillna(1).astype(int)
    window_frame["deviation_score"] = compute_deviation_score(window_frame, weights)
    return window_frame.sort_values("start_time").reset_index(drop=True)


def _normalize_window_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize supported window input column conventions.

    Copies ``window_start``/``window_end`` into ``start_time``/``end_time``
    and ``active_sequence_count`` into ``run_count`` when the target column
    is absent. Source columns are kept.

    Args:
        frame: Raw input table.

    Returns:
        Normalized copy.
    """

    normalized = frame.copy()
    for source, target in TIME_COLUMN_ALIASES.items():
        if source in normalized.columns and target not in normalized.columns:
            normalized[target] = normalized[source]
    for source, target in COUNT_COLUMN_ALIASES.items():
        if source in normalized.columns and target not in normalized.columns:
            normalized[target] = normalized[source]
    return normalized


def _is_prewindowed(frame: pd.DataFrame) -> bool:
    """Return whether the input already represents analysis windows.

    Args:
        frame: Normalized input table.

    Returns:
        ``True`` if any of :data:`PREWINDOWED_COLUMNS` is present.
    """

    return bool(PREWINDOWED_COLUMNS.intersection(frame.columns))


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    r"""Compute a weighted mean with safe fallbacks for empty or invalid data.

    .. math::

       \bar x = \frac{\sum_i w_i x_i}{\sum_i w_i}
       \quad \text{over } i \text{ with finite } x_i \text{ and } w_i > 0.

    Args:
        values: Values to average (coerced to numeric).
        weights: Non-negative weights, e.g. overlap seconds.

    Returns:
        The weighted mean, or ``0.0`` when no valid pair exists.
    """

    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = numeric_values.notna() & numeric_weights.gt(0)
    if not bool(mask.any()):
        return 0.0
    return float((numeric_values[mask] * numeric_weights[mask]).sum() / numeric_weights[mask].sum())




def _fit_behavioral_baseline(frame: pd.DataFrame) -> dict[str, object]:
    """Fit global and per-regime state-word frequency profiles.

    Args:
        frame: Timed rows, optionally with a ``states`` column.

    Returns:
        Dictionary ``{"global": profile, "by_regime": {regime: profile}}``
        where each profile comes from :func:`_fit_single_behavioral_profile`
        and regimes come from :func:`_infer_baseline_regime`. Regimes with
        fewer than two rows are omitted.

    Notes:
        Fitted on the full frame passed in (no chronological partition).
    """
    profiles = {
        "global": _fit_single_behavioral_profile(frame),
        "by_regime": {},
    }
    if frame.empty:
        return profiles

    enriched = frame.copy()
    enriched["_baseline_regime"] = _infer_baseline_regime(enriched)
    regime_profiles: dict[str, dict[str, object]] = {}
    for regime, regime_frame in enriched.groupby("_baseline_regime", dropna=False, sort=False):
        if len(regime_frame) < 2:
            continue
        regime_profiles[str(regime)] = _fit_single_behavioral_profile(regime_frame)
    profiles["by_regime"] = regime_profiles
    return profiles


def _fit_single_behavioral_profile(frame: pd.DataFrame) -> dict[str, object]:
    """Fit word and state frequency tables plus rarity thresholds.

    Args:
        frame: Rows with an optional ``states`` column.

    Returns:
        Dictionary with ``word_frequency`` (word -> relative frequency),
        ``state_frequency`` (state -> relative frequency over all parsed
        tokens), ``rare_word_threshold`` and ``rare_state_threshold`` (the
        20th percentile of the respective frequency tables). All empty/zero
        when no state words exist.
    """
    state_words = frame["states"].dropna().astype(str) if "states" in frame.columns else pd.Series(dtype=str)
    if state_words.empty:
        return {
            "word_frequency": {},
            "state_frequency": {},
            "rare_word_threshold": 0.0,
            "rare_state_threshold": 0.0,
        }

    word_counts = state_words.value_counts(dropna=True)
    word_frequency = (word_counts / max(int(word_counts.sum()), 1)).to_dict()
    state_counts: dict[int, int] = {}
    for word in state_words:
        for state in _parse_state_word(word):
            state_counts[state] = state_counts.get(state, 0) + 1
    total_states = max(sum(state_counts.values()), 1)
    state_frequency = {state: count / total_states for state, count in state_counts.items()}
    rare_word_threshold = float((word_counts / max(int(word_counts.sum()), 1)).quantile(0.2)) if not word_counts.empty else 0.0
    rare_state_threshold = float(pd.Series(state_frequency).quantile(0.2)) if state_frequency else 0.0
    return {
        "word_frequency": word_frequency,
        "state_frequency": state_frequency,
        "rare_word_threshold": rare_word_threshold,
        "rare_state_threshold": rare_state_threshold,
    }


def _summarize_state_words(active_rows: pd.DataFrame, baseline_profiles: dict[str, object]) -> dict[str, object]:
    """Summarize the local sequence-word regime inside one window.

    Args:
        active_rows: Rows with positive overlap in the window.
        baseline_profiles: Output of :func:`_fit_behavioral_baseline`; the
            profile of the window's modal regime is used when available.

    Returns:
        Dictionary with ``dominant_state_word``, ``state_word_diversity``
        (number of distinct words), ``dominant_state_word_fraction``,
        ``state_word_transition_rate`` (word changes per adjacent pair),
        ``nominal_state_word_match_fraction``, ``mean_state_distance``,
        ``mean_dtw_distance``, ``mean_nominal_anomaly_score``,
        ``rare_word_fraction``, ``rare_state_fraction``, ``state_entropy``
        (nats), ``state_17_fraction``, ``off_nominal_state_fraction`` and
        ``word_regime_shift_score`` (total-variation distance to the
        baseline word distribution). All zero/empty without ``states``.
    """

    default = {
        "dominant_state_word": "",
        "state_word_diversity": 0.0,
        "dominant_state_word_fraction": 0.0,
        "state_word_transition_rate": 0.0,
        "nominal_state_word_match_fraction": 0.0,
        "mean_state_distance": 0.0,
        "mean_dtw_distance": 0.0,
        "mean_nominal_anomaly_score": 0.0,
        "rare_word_fraction": 0.0,
        "rare_state_fraction": 0.0,
        "state_entropy": 0.0,
        "state_17_fraction": 0.0,
        "off_nominal_state_fraction": 0.0,
        "word_regime_shift_score": 0.0,
    }
    if "states" not in active_rows.columns:
        return default

    state_words = active_rows["states"].dropna().astype(str)
    if state_words.empty:
        return default

    baseline_profile = _resolve_baseline_profile(active_rows, baseline_profiles)

    counts = state_words.value_counts(dropna=True)
    dominant_word = str(counts.index[0]) if not counts.empty else ""
    total = int(counts.sum())
    # Note: the first word always differs from its NaN shift, hence the -1.
    transitions = int((state_words != state_words.shift(1)).sum() - 1) if len(state_words) > 1 else 0
    transition_rate = float(transitions / max(len(state_words) - 1, 1))
    nominal_match_fraction = _nominal_match_fraction(active_rows)
    rare_word_fraction = _rare_word_fraction(state_words, baseline_profile)
    state_tokens = [state for word in state_words for state in _parse_state_word(word)]
    rare_state_fraction = _rare_state_fraction(state_tokens, baseline_profile)
    state_entropy = _state_entropy(state_tokens)
    state_17_fraction = float(sum(1 for state in state_tokens if state == 17) / max(len(state_tokens), 1))
    off_nominal_state_fraction = _off_nominal_state_fraction(active_rows)
    word_regime_shift_score = _word_regime_shift_score(state_words, baseline_profile)
    payload = default.copy()
    payload.update({
        "dominant_state_word": dominant_word,
        "state_word_diversity": float(state_words.nunique(dropna=True)),
        "dominant_state_word_fraction": float(counts.iloc[0] / total) if total else 0.0,
        "state_word_transition_rate": transition_rate,
        "nominal_state_word_match_fraction": nominal_match_fraction,
        "mean_state_distance": _mean_numeric(active_rows, "state_distance"),
        "mean_dtw_distance": _mean_numeric(active_rows, "dtw_distance"),
        "mean_nominal_anomaly_score": _mean_numeric(active_rows, "anomaly_score"),
        "rare_word_fraction": rare_word_fraction,
        "rare_state_fraction": rare_state_fraction,
        "state_entropy": state_entropy,
        "state_17_fraction": state_17_fraction,
        "off_nominal_state_fraction": off_nominal_state_fraction,
        "word_regime_shift_score": word_regime_shift_score,
    })
    return payload




def _infer_fallback_family(window_summary: dict[str, object]) -> str | None:
    """Infer a family for windows with a stable, localized word-regime shift.

    Used to promote semantically ``NORMAL`` windows whose state-word summary
    matches :func:`_has_localized_stable_shift_signature`.

    Args:
        window_summary: Window payload containing the state-word summary
            keys (missing or falsy values are read as 0.0).

    Returns:
        ``None`` when the signature does not match; otherwise
        ``"process_saturation"`` if state 17 is present, the off-nominal
        state fraction is at least 0.2, the nominal match fraction is at
        most 0.35 or the transition rate is at most 0.35; else
        ``"external_ambiguous_disturbance"``.

    Notes:
        Because the signature itself requires ``transition_rate <= 0.05``,
        the ``transition_rate <= 0.35`` clause always holds when reached,
        so a matching window is in practice always labelled
        ``"process_saturation"``. Both remaining branches return the same
        value.
    """
    shift = float(window_summary.get("word_regime_shift_score", 0.0) or 0.0)
    dominant_fraction = float(window_summary.get("dominant_state_word_fraction", 0.0) or 0.0)
    rare_word = float(window_summary.get("rare_word_fraction", 0.0) or 0.0)
    rare_state = float(window_summary.get("rare_state_fraction", 0.0) or 0.0)
    state17 = float(window_summary.get("state_17_fraction", 0.0) or 0.0)
    off_nominal = float(window_summary.get("off_nominal_state_fraction", 0.0) or 0.0)
    nominal_match = float(window_summary.get("nominal_state_word_match_fraction", 0.0) or 0.0)
    transition_rate = float(window_summary.get("state_word_transition_rate", 0.0) or 0.0)
    diversity = float(window_summary.get("state_word_diversity", 0.0) or 0.0)

    if not _has_localized_stable_shift_signature(
        shift=shift,
        dominant_fraction=dominant_fraction,
        rare_word=rare_word,
        rare_state=rare_state,
        transition_rate=transition_rate,
        diversity=diversity,
    ):
        return None
    if state17 > 0.0 or off_nominal >= 0.2 or nominal_match <= 0.35 or transition_rate <= 0.35:
        return "process_saturation"
    if rare_state >= 0.2 or rare_word >= 0.5:
        return "external_ambiguous_disturbance"
    return "external_ambiguous_disturbance"


def _has_localized_stable_shift_signature(
    *,
    shift: float,
    dominant_fraction: float,
    rare_word: float,
    rare_state: float,
    transition_rate: float,
    diversity: float,
) -> bool:
    """Test the hard-coded localized stable-shift signature.

    Args:
        shift: Word-regime shift score (total variation, ``[0, 1]``).
        dominant_fraction: Fraction of the dominant word.
        rare_word: Fraction of rare words.
        rare_state: Fraction of rare state tokens.
        transition_rate: Word transition rate.
        diversity: Number of distinct words.

    Returns:
        ``True`` iff ``shift >= 0.995``, ``dominant_fraction >= 0.999``,
        ``rare_word <= 0.05``, ``0.10 <= rare_state <= 0.20``,
        ``transition_rate <= 0.05`` and ``diversity <= 1.5`` (i.e. a single
        word, almost absent from the baseline, persisting across the window).
    """
    return bool(
        shift >= 0.995
        and dominant_fraction >= 0.999
        and rare_word <= 0.05
        and 0.10 <= rare_state <= 0.20
        and transition_rate <= 0.05
        and diversity <= 1.5
    )


def _resolve_baseline_profile(active_rows: pd.DataFrame, baseline_profiles: dict[str, object]) -> dict[str, object]:
    """Select the baseline profile matching the window's modal regime.

    Args:
        active_rows: Window rows.
        baseline_profiles: Either the nested output of
            :func:`_fit_behavioral_baseline` or a single flat profile.

    Returns:
        The per-regime profile for the most frequent regime of the rows, or
        the global profile when unavailable.
    """
    global_profile = baseline_profiles.get("global", baseline_profiles)
    by_regime = baseline_profiles.get("by_regime", {})
    if active_rows.empty or not by_regime:
        return global_profile
    regime = _infer_baseline_regime(active_rows).mode(dropna=True)
    if regime.empty:
        return global_profile
    return by_regime.get(str(regime.iloc[0]), global_profile)


def _infer_baseline_regime(frame: pd.DataFrame) -> pd.Series:
    """Label each row with a ``<day_phase>__<activity_band>`` regime.

    Day phases use the hour of ``start_time`` (missing -> hour 12):
    ``night`` for hours < 7 or >= 22, ``morning`` for [7, 10), ``evening``
    for [18, 22) and ``day`` otherwise. The activity band is
    ``high_activity`` when ``run_count`` (else ``sequence_count``, else 1)
    exceeds ``max(median, 1)`` of the frame, ``low_activity`` otherwise.

    Args:
        frame: Rows to label.

    Returns:
        String series aligned with ``frame.index`` (empty for empty input).

    Notes:
        Hours are taken from timestamps as stored (naive UTC after window
        normalization), not local plant time.
    """
    if frame.empty:
        return pd.Series(dtype=str)
    if "start_time" in frame.columns:
        timestamps = pd.to_datetime(frame["start_time"], errors="coerce")
    else:
        timestamps = pd.Series(pd.NaT, index=frame.index)
    hours = timestamps.dt.hour.fillna(12).astype(int)
    day_phase = pd.Series("day", index=frame.index, dtype=str)
    day_phase.loc[(hours < 7) | (hours >= 22)] = "night"
    day_phase.loc[(hours >= 7) & (hours < 10)] = "morning"
    day_phase.loc[(hours >= 18) & (hours < 22)] = "evening"

    if "run_count" in frame.columns:
        activity_values = pd.to_numeric(frame["run_count"], errors="coerce")
    elif "sequence_count" in frame.columns:
        activity_values = pd.to_numeric(frame["sequence_count"], errors="coerce")
    else:
        activity_values = pd.Series(1.0, index=frame.index)
    activity_values = activity_values.fillna(activity_values.median() if activity_values.notna().any() else 1.0)
    median_activity = float(activity_values.median()) if not activity_values.empty else 1.0
    high_activity = activity_values > max(median_activity, 1.0)
    activity_band = pd.Series("low_activity", index=frame.index, dtype=str)
    activity_band.loc[high_activity] = "high_activity"

    return day_phase.str.cat(activity_band, sep="__")

def _parse_state_word(value: object) -> tuple[int, ...]:
    """Parse a state word into a tuple of integer states.

    Args:
        value: Tuple, list, or string literal such as ``"(1, 17, 3)"``.

    Returns:
        Tuple of ints; empty for ``None``, NaN, blank or unparsable input.

    Raises:
        ValueError: If a parsed element cannot be converted to ``int``.
        TypeError: If a parsed element has a type ``int`` cannot convert.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return tuple()
    if isinstance(value, tuple):
        return tuple(int(item) for item in value)
    if isinstance(value, list):
        return tuple(int(item) for item in value)
    text = str(value).strip()
    if not text:
        return tuple()
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return tuple()
    if isinstance(parsed, (list, tuple)):
        return tuple(int(item) for item in parsed)
    return tuple()


def _nominal_match_fraction(active_rows: pd.DataFrame) -> float:
    """Return the fraction of rows whose word equals its nominal match.

    Args:
        active_rows: Rows with ``states`` and ``nominal_match``.

    Returns:
        Exact string-equality rate, or 0.0 if a column is missing or the
        frame is empty.
    """
    if "states" not in active_rows.columns or "nominal_match" not in active_rows.columns or active_rows.empty:
        return 0.0
    return float(active_rows["states"].astype(str).eq(active_rows["nominal_match"].astype(str)).mean())


def _off_nominal_state_fraction(active_rows: pd.DataFrame) -> float:
    """Return the mean position-wise mismatch between words and nominal words.

    For each row with a non-empty observed word, mismatches are counted
    position by position (missing nominal positions count as mismatches,
    and surplus nominal positions are added), divided by the longer length.

    Args:
        active_rows: Rows with ``states`` and ``nominal_match``.

    Returns:
        Mean mismatch fraction in ``[0, 1]``; 0.0 when unavailable.
    """
    if "states" not in active_rows.columns or "nominal_match" not in active_rows.columns or active_rows.empty:
        return 0.0
    fractions: list[float] = []
    for _, row in active_rows.iterrows():
        observed = _parse_state_word(row.get("states"))
        nominal = _parse_state_word(row.get("nominal_match"))
        if not observed:
            continue
        mismatches = sum(1 for idx, state in enumerate(observed) if idx >= len(nominal) or nominal[idx] != state)
        mismatches += max(len(nominal) - len(observed), 0)
        fractions.append(float(mismatches / max(len(observed), len(nominal), 1)))
    return float(sum(fractions) / len(fractions)) if fractions else 0.0


def _rare_word_fraction(state_words: pd.Series, baseline_profiles: dict[str, object]) -> float:
    """Return the fraction of words rarer than the profile threshold.

    Args:
        state_words: Words in the window.
        baseline_profiles: A single profile (``word_frequency``,
            ``rare_word_threshold``).

    Returns:
        Fraction of words whose baseline frequency (0 if unseen) is strictly
        below ``rare_word_threshold``; 0.0 when unavailable.
    """
    freq = baseline_profiles.get("word_frequency", {})
    threshold = float(baseline_profiles.get("rare_word_threshold", 0.0))
    if state_words.empty or not freq:
        return 0.0
    rare = [freq.get(word, 0.0) < threshold for word in state_words.astype(str)]
    return float(sum(rare) / len(rare)) if rare else 0.0


def _rare_state_fraction(state_tokens: list[int], baseline_profiles: dict[str, object]) -> float:
    """Return the fraction of state tokens rarer than the profile threshold.

    Args:
        state_tokens: Flattened states of all window words.
        baseline_profiles: A single profile (``state_frequency``,
            ``rare_state_threshold``).

    Returns:
        Fraction of tokens whose baseline frequency (0 if unseen) is strictly
        below ``rare_state_threshold``; 0.0 when unavailable.
    """
    freq = baseline_profiles.get("state_frequency", {})
    threshold = float(baseline_profiles.get("rare_state_threshold", 0.0))
    if not state_tokens or not freq:
        return 0.0
    rare = [freq.get(state, 0.0) < threshold for state in state_tokens]
    return float(sum(rare) / len(rare)) if rare else 0.0


def _state_entropy(state_tokens: list[int]) -> float:
    r"""Return the Shannon entropy of the state-token distribution in nats.

    .. math:: H = -\sum_s p_s \ln(p_s + 10^{-12})

    Args:
        state_tokens: Flattened states.

    Returns:
        Entropy in nats; 0.0 for an empty list.
    """
    if not state_tokens:
        return 0.0
    series = pd.Series(state_tokens)
    probs = series.value_counts(normalize=True)
    return float(-sum(p * math.log(p + 1e-12) for p in probs))


def _word_regime_shift_score(state_words: pd.Series, baseline_profiles: dict[str, object]) -> float:
    r"""Return the total-variation distance between local and baseline words.

    .. math:: \mathrm{TV} = \tfrac12 \sum_{w} |\hat p_w - p^0_w|

    Args:
        state_words: Words in the window.
        baseline_profiles: A single profile with ``word_frequency``.

    Returns:
        Distance in ``[0, 1]``; 0.0 when unavailable.
    """
    global_freq = baseline_profiles.get("word_frequency", {})
    if state_words.empty or not global_freq:
        return 0.0
    local_freq = state_words.value_counts(normalize=True).to_dict()
    union = set(global_freq) | set(local_freq)
    return float(0.5 * sum(abs(local_freq.get(word, 0.0) - global_freq.get(word, 0.0)) for word in union))


def _mean_numeric(frame: pd.DataFrame, column: str) -> float:
    """Return the mean of a numeric column ignoring missing values.

    Args:
        frame: Rows.
        column: Column name.

    Returns:
        Mean of the coerced values; 0.0 if the column is absent or empty.
    """
    if column not in frame.columns or frame.empty:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else 0.0

def _overlap_seconds(
    left_start: pd.Timestamp,
    left_end: pd.Timestamp,
    right_start: pd.Timestamp,
    right_end: pd.Timestamp,
) -> float:
    """Compute the temporal overlap in seconds between two intervals.

    Args:
        left_start: Start of the first interval.
        left_end: End of the first interval.
        right_start: Start of the second interval (parsed with
            :func:`pandas.to_datetime`).
        right_end: End of the second interval (parsed likewise).

    Returns:
        Non-negative overlap duration in seconds.

    Notes:
        Currently unused within this module; kept as a helper.
    """

    overlap_start = max(left_start, pd.to_datetime(right_start))
    overlap_end = min(left_end, pd.to_datetime(right_end))
    return max((overlap_end - overlap_start).total_seconds(), 0.0)

