"""Episode-level summary metrics grouped by incident family.

Role in the pipeline:
    This module sits at the very end of the Model_D episode route. Once
    :class:`~iabm_incidents.episodes.IncidentEpisodeBuilder` has materialised
    an episode table (semantic route, indicator route or four-level ablation),
    :func:`summarize_episode_metrics` reduces it to one descriptive row per
    ``primary_family``. The CLI writes the result as
    ``evaluation_summary.<suffix>``; :mod:`~iabm_incidents.experimental` and
    :mod:`~iabm_incidents.ablation` reuse it.

Main public objects:
    * :func:`summarize_episode_metrics` - family-level counts, durations,
      time-to-recovery statistics, stable-recovery rate and mean peak score.

Data contract:
    Input episode frames must contain ``primary_family`` and ``episode_id``.
    ``duration_seconds``, ``time_to_recovery_seconds``, ``peak_score`` (or
    ``mean_score`` as fallback) and ``recovery_status`` are optional. A
    missing ``recovery_status`` is treated as ``"unknown"`` and is therefore
    never counted as recovered.

Maintenance notes:
    * The metrics are purely descriptive. They are not detection-quality
      metrics; precision/recall against a registry live in
      :mod:`~iabm_incidents.evaluation`.
    * The column order of the empty frame must stay identical to the
      non-empty one so that CSV/Excel exports keep a stable schema.
"""

from __future__ import annotations

import pandas as pd


def summarize_episode_metrics(episodes: pd.DataFrame) -> pd.DataFrame:
    """Summarize episode metrics grouped by primary family.

    Each output row aggregates every episode that shares one
    ``primary_family`` value. ``NaN`` families form their own group because
    the grouping uses ``dropna=False``.

    Args:
        episodes: Episode table produced by
            :class:`~iabm_incidents.episodes.IncidentEpisodeBuilder`. Required
            columns are ``primary_family`` and ``episode_id``. Optional columns
            are ``duration_seconds`` (seconds), ``time_to_recovery_seconds``
            (seconds), ``peak_score`` (dimensionless deviation score; falls
            back to ``mean_score`` and then to ``0.0``) and ``recovery_status``
            (values of :class:`~iabm_incidents.recovery.RecoveryStatus`). The
            input frame is not modified.

    Returns:
        pandas.DataFrame: One row per family with columns ``primary_family``,
        ``episode_count``, ``mean_duration_seconds``,
        ``median_duration_seconds``, ``mean_time_to_recovery_seconds`` and
        ``median_time_to_recovery_seconds`` (``None`` when no episode of the
        family has a recovery time), ``recovery_rate`` (fraction of episodes
        whose status equals ``"stable"``) and ``mean_peak_score``, sorted by
        ``episode_count`` in descending order. An empty input yields an empty
        frame with the same columns.

    Raises:
        KeyError: If a non-empty ``episodes`` frame lacks ``primary_family``
            or ``episode_id``.

    Notes:
        Only ``"stable"`` counts as recovered; ``"partial"`` and
        ``"reconfigured"`` statuses lower ``recovery_rate``.

    Examples:
        >>> import pandas as pd
        >>> frame = pd.DataFrame({"episode_id": ["e1", "e2"],
        ...                       "primary_family": ["process_saturation"] * 2,
        ...                       "duration_seconds": [600.0, 1200.0],
        ...                       "recovery_status": ["stable", "partial"]})
        >>> summarize_episode_metrics(frame)["recovery_rate"].item()
        0.5

    See Also:
        :meth:`iabm_incidents.occurrence.OccurrenceModeler.summarize`:
        exposure-aware recurrence summaries of the same episode table.
    """

    if episodes.empty:
        return pd.DataFrame(
            columns=[
                "primary_family",
                "episode_count",
                "mean_duration_seconds",
                "median_duration_seconds",
                "mean_time_to_recovery_seconds",
                "median_time_to_recovery_seconds",
                "recovery_rate",
                "mean_peak_score",
            ]
        )

    frame = episodes.copy()
    frame["duration_seconds"] = pd.to_numeric(frame.get("duration_seconds"), errors="coerce")
    frame["time_to_recovery_seconds"] = pd.to_numeric(frame.get("time_to_recovery_seconds"), errors="coerce")
    if "peak_score" not in frame.columns:
        fallback = frame.get("mean_score", pd.Series(0.0, index=frame.index))
        frame["peak_score"] = fallback
    frame["peak_score"] = pd.to_numeric(frame["peak_score"], errors="coerce")
    if "recovery_status" not in frame.columns:
        frame["recovery_status"] = "unknown"

    rows: list[dict[str, object]] = []
    for family, family_frame in frame.groupby("primary_family", dropna=False):
        recovery_series = family_frame["time_to_recovery_seconds"].dropna()
        rows.append(
            {
                "primary_family": family,
                "episode_count": int(family_frame["episode_id"].size),
                "mean_duration_seconds": float(family_frame["duration_seconds"].mean()),
                "median_duration_seconds": float(family_frame["duration_seconds"].median()),
                "mean_time_to_recovery_seconds": float(recovery_series.mean()) if not recovery_series.empty else None,
                "median_time_to_recovery_seconds": float(recovery_series.median()) if not recovery_series.empty else None,
                "recovery_rate": float(family_frame["recovery_status"].eq("stable").mean()),
                "mean_peak_score": float(family_frame["peak_score"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("episode_count", ascending=False)
