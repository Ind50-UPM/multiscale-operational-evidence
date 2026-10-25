"""Family-specific occurrence modelling scaffolding.

Role in the pipeline:
    Downstream of episode construction, :class:`OccurrenceModeler` turns an
    episode table into per-family recurrence descriptors: inter-occurrence
    intervals and, when an exposure table is supplied, an exposure-normalised
    event rate. The CLI writes the result as ``occurrence_summary.<suffix>``.
    It prepares, but does not perform, distribution fitting (e.g. renewal or
    Weibull models of time between events).

Main public objects:
    * :class:`OccurrenceModeler` - family-level occurrence summaries.

Data contracts:
    *Episodes*: ``event_start`` (timestamp), ``primary_family`` and optional
    ``asset_id``.

    *Exposure* (optional; e.g. the ``Observation_exposure`` workbook sheet):
    ``asset_id``, ``observation_start``, ``observation_end`` and optional
    ``excluded_start``/``excluded_end``. Every row contributes its full
    observation span and, when both exclusion bounds are present, its
    excluded span.

Notes:
    For family :math:`f` with :math:`N_f` episodes and effective exposure
    :math:`E_f` (hours) over its participating assets,

    .. math::

        \\hat\\lambda_f = N_f / E_f, \\qquad
        E_f = \\sum_{a} \\max\\left(0, T^{obs}_a - T^{exc}_a\\right).

Maintenance notes:
    * Intervals are computed on the family's episodes pooled across assets
      and sorted by ``event_start``; they are not per-asset intervals.
    * Exclusion spans are not intersected with observation spans; they are
      subtracted as given, then clipped at zero per asset.
    * ``fit_status`` only reports whether ``minimum_events_for_fit`` is
      reached; no model is fitted.
"""

from __future__ import annotations

import pandas as pd

from .config import OccurrenceConfig


class OccurrenceModeler:
    """Derive simple family-specific occurrence indicators.

    Args:
        config: Occurrence configuration; defaults to
            :class:`~iabm_incidents.config.OccurrenceConfig`. Only
            ``minimum_events_for_fit`` is used.
    """

    def __init__(self, config: OccurrenceConfig | None = None) -> None:
        """Resolve the occurrence configuration.

        Args:
            config: Optional configuration; ``None`` uses the defaults.
        """
        self._config = config or OccurrenceConfig()

    def summarize(
        self,
        episodes: pd.DataFrame,
        exposure: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Summarize recurrence intervals and exposure-aware event rates by family.

        Args:
            episodes: Episode table with ``event_start`` and
                ``primary_family`` (optional ``asset_id``).
            exposure: Optional exposure table (see module data contract).

        Returns:
            pandas.DataFrame: One row per family with ``primary_family``,
            ``occurrence_count``, ``mean_/median_/minimum_/maximum_
            occurrence_interval_seconds`` (``None`` with fewer than two
            episodes), ``exposure_hours`` and ``excluded_hours`` (``None``
            without usable exposure), ``event_rate_per_hour`` (``None`` when
            exposure is missing or zero) and ``fit_status``
            (``"insufficient_events"`` or ``"ready_for_distribution_fit"``),
            sorted by ``occurrence_count`` descending. An empty episode frame
            yields an empty frame with these columns.

        Raises:
            KeyError: If required episode or exposure columns are missing.
        """

        if episodes.empty:
            return pd.DataFrame(
                columns=[
                    "primary_family",
                    "occurrence_count",
                    "mean_occurrence_interval_seconds",
                    "median_occurrence_interval_seconds",
                    "minimum_occurrence_interval_seconds",
                    "maximum_occurrence_interval_seconds",
                    "exposure_hours",
                    "excluded_hours",
                    "event_rate_per_hour",
                    "fit_status",
                ]
            )

        exposure_summary = _summarize_exposure(exposure)
        rows = []
        for family, family_frame in episodes.sort_values("event_start").groupby("primary_family"):
            starts = pd.to_datetime(family_frame["event_start"]).tolist()
            # Note: consecutive differences of pooled (multi-asset) starts.
            intervals = [
                (starts[index] - starts[index - 1]).total_seconds()
                for index in range(1, len(starts))
            ]
            asset_ids = family_frame["asset_id"].dropna().unique().tolist() if "asset_id" in family_frame.columns else []
            exposure_hours, excluded_hours = _exposure_for_assets(exposure_summary, asset_ids)
            rows.append(
                {
                    "primary_family": family,
                    "occurrence_count": len(family_frame),
                    "mean_occurrence_interval_seconds": sum(intervals) / len(intervals) if intervals else None,
                    "median_occurrence_interval_seconds": pd.Series(intervals).median() if intervals else None,
                    "minimum_occurrence_interval_seconds": min(intervals) if intervals else None,
                    "maximum_occurrence_interval_seconds": max(intervals) if intervals else None,
                    "exposure_hours": exposure_hours,
                    "excluded_hours": excluded_hours,
                    "event_rate_per_hour": len(family_frame) / exposure_hours if exposure_hours and exposure_hours > 0 else None,
                    "fit_status": "insufficient_events"
                    if len(family_frame) < self._config.minimum_events_for_fit
                    else "ready_for_distribution_fit",
                }
            )
        return pd.DataFrame(rows).sort_values("occurrence_count", ascending=False)


def _summarize_exposure(exposure: pd.DataFrame | None) -> pd.DataFrame:
    """Convert raw exposure intervals into effective observation summaries.

    Args:
        exposure: Exposure table or ``None``.

    Returns:
        pandas.DataFrame: One row per ``asset_id`` with ``exposure_hours``
        (observed minus excluded, clipped at zero) and ``excluded_hours``.
        Empty (with those columns) when ``exposure`` is ``None`` or empty.

    Raises:
        KeyError: If ``observation_start``, ``observation_end`` or
            ``asset_id`` is missing.
    """

    if exposure is None or exposure.empty:
        return pd.DataFrame(columns=["asset_id", "exposure_hours", "excluded_hours"])

    frame = exposure.copy()
    frame["observation_start"] = pd.to_datetime(frame["observation_start"])
    frame["observation_end"] = pd.to_datetime(frame["observation_end"])
    if "excluded_start" in frame.columns:
        frame["excluded_start"] = pd.to_datetime(frame["excluded_start"])
    if "excluded_end" in frame.columns:
        frame["excluded_end"] = pd.to_datetime(frame["excluded_end"])

    rows: list[dict[str, object]] = []
    for asset_id, asset_frame in frame.groupby("asset_id", dropna=False):
        observation_hours = (
            (asset_frame["observation_end"] - asset_frame["observation_start"]).dt.total_seconds().sum() / 3600.0
        )
        excluded_hours = 0.0
        if {"excluded_start", "excluded_end"}.issubset(asset_frame.columns):
            valid_exclusions = asset_frame[asset_frame["excluded_start"].notna() & asset_frame["excluded_end"].notna()]
            excluded_hours = (
                (valid_exclusions["excluded_end"] - valid_exclusions["excluded_start"]).dt.total_seconds().sum() / 3600.0
            )
        rows.append(
            {
                "asset_id": asset_id,
                "exposure_hours": max(observation_hours - excluded_hours, 0.0),
                "excluded_hours": excluded_hours,
            }
        )
    return pd.DataFrame(rows)


def _exposure_for_assets(summary: pd.DataFrame, asset_ids: list[str]) -> tuple[float | None, float | None]:
    """Resolve effective exposure for the assets participating in a family.

    Args:
        summary: Output of :func:`_summarize_exposure`.
        asset_ids: Assets observed in the family's episodes. An empty list
            (no ``asset_id`` information) selects all assets.

    Returns:
        tuple[float | None, float | None]: Summed ``(exposure_hours,
        excluded_hours)``, or ``(None, None)`` when no exposure is available
        or none of the requested assets is present.
    """

    if summary.empty:
        return None, None
    if not asset_ids:
        return float(summary["exposure_hours"].sum()), float(summary["excluded_hours"].sum())
    filtered = summary[summary["asset_id"].isin(asset_ids)]
    if filtered.empty:
        return None, None
    return float(filtered["exposure_hours"].sum()), float(filtered["excluded_hours"].sum())
