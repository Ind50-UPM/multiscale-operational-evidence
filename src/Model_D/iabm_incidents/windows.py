"""Rolling-window generation for indicator-driven Model_D workflows.

Role in the pipeline:
    :class:`~iabm_incidents.pipeline.IndicatorPipeline` first derives
    exposure-aware observation periods
    (:func:`~iabm_incidents.exposure.derive_observation_periods`) and then
    calls :func:`build_rolling_windows` to lay a regular grid of fixed-length
    analysis windows over them. Each window later receives one row of
    deviation indicators and one ``deviation_score``.

Main public objects:
    * :class:`WindowBuildConfig` - window length, step and minimum coverage.
    * :func:`build_rolling_windows` - window table builder.

Data contract:
    Input ``observation_periods`` columns: ``start_time``, ``end_time``
    (timestamps; tz-aware values are converted to naive UTC),
    ``period_type`` (``"observed"`` marks usable intervals),
    ``exclude_from_exposure`` (bool; ``NaN`` means ``False``) and optional
    ``asset_id`` (defaults to ``"asset-unknown"``).

    Output columns: ``window_id`` (``"<asset_id>_<ISO start>"``),
    ``asset_id``, ``start_time``, ``end_time``, ``data_coverage`` (fraction in
    ``[0, 1]``) and ``is_valid`` (``data_coverage >= min_coverage``).

Notes:
    Windows are anchored at the start of each *observed* period and advance
    by ``step`` while the whole window fits inside that period; windows never
    span two observed periods. Coverage is

    .. math::

        c = \\min\\left(1, \\frac{\\max(0, |W \\cap O| - |W \\cap X|)}
        {\\max(|W|, 1\\,\\mathrm{s})}\\right),

    where :math:`O` are observed periods, :math:`X` excluded periods and
    :math:`|\\cdot|` durations in seconds.

Maintenance notes:
    * Invalid windows (low coverage) are *kept* with ``is_valid=False`` so
      that exposure denominators remain auditable; consumers must filter.
    * Overlaps are summed per interval, so overlapping observed periods can
      over-count; the final ``min(..., 1)`` caps the ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WindowBuildConfig:
    """Configuration for rolling observation windows.

    Attributes:
        length: Window duration (default 24 h).
        step: Offset between consecutive window starts (default 1 h). A step
            shorter than ``length`` produces overlapping windows.
        min_coverage: Minimum effective observed fraction in ``[0, 1]`` for a
            window to be flagged valid (default 0.90).
    """

    length: pd.Timedelta = pd.Timedelta(hours=24)
    step: pd.Timedelta = pd.Timedelta(hours=1)
    min_coverage: float = 0.90


def build_rolling_windows(
    observation_periods: pd.DataFrame,
    config: WindowBuildConfig | None = None,
) -> pd.DataFrame:
    """Create rolling windows over observation periods, excluding invalid intervals.

    Excluded intervals (``exclude_from_exposure``) reduce window coverage
    instead of removing windows.

    Args:
        observation_periods: Exposure table (see the module data contract).
            Duplicate column labels are dropped (first occurrence kept) and
            rows with unparsable times are discarded.
        config: Window geometry; defaults to :class:`WindowBuildConfig`.

    Returns:
        pandas.DataFrame: Window table with columns ``window_id``,
        ``asset_id``, ``start_time``, ``end_time``, ``data_coverage`` and
        ``is_valid``. An empty input returns an empty frame with those
        columns; assets without observed periods contribute no rows (if no
        asset has any window the returned frame has no columns).

    Raises:
        KeyError: If a non-empty input lacks ``start_time``, ``end_time``,
            ``period_type`` or ``exclude_from_exposure``.

    Examples:
        >>> import pandas as pd
        >>> periods = pd.DataFrame({
        ...     "asset_id": ["a"], "start_time": ["2022-01-01 00:00"],
        ...     "end_time": ["2022-01-01 03:00"], "period_type": ["observed"],
        ...     "exclude_from_exposure": [False]})
        >>> cfg = WindowBuildConfig(length=pd.Timedelta(hours=2),
        ...                         step=pd.Timedelta(hours=1))
        >>> len(build_rolling_windows(periods, cfg))
        2
    """

    effective_config = config or WindowBuildConfig()
    if observation_periods.empty:
        return pd.DataFrame(columns=['window_id', 'asset_id', 'start_time', 'end_time', 'data_coverage', 'is_valid'])

    frame = observation_periods.copy()
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    # Note: timestamps are normalised to naive UTC for every downstream join.
    frame['start_time'] = pd.to_datetime(frame['start_time'], errors='coerce', utc=True).dt.tz_convert(None)
    frame['end_time'] = pd.to_datetime(frame['end_time'], errors='coerce', utc=True).dt.tz_convert(None)
    frame = frame.dropna(subset=['start_time', 'end_time']).reset_index(drop=True)
    frame['asset_id'] = frame.get('asset_id', pd.Series('asset-unknown', index=frame.index)).fillna('asset-unknown').astype(str)

    rows: list[dict[str, object]] = []
    for asset_id, asset_frame in frame.groupby('asset_id', dropna=False):
        observed = asset_frame[asset_frame['period_type'].eq('observed')].sort_values('start_time').reset_index(drop=True)
        excluded = asset_frame[asset_frame['exclude_from_exposure'].fillna(False)].sort_values('start_time').reset_index(drop=True)
        if observed.empty:
            continue
        for period in observed.itertuples(index=False):
            current_start = period.start_time
            # Invariant: a window never extends beyond its anchoring period.
            while current_start + effective_config.length <= period.end_time:
                current_end = current_start + effective_config.length
                coverage = _window_coverage(current_start, current_end, observed, excluded)
                rows.append(
                    {
                        'window_id': f'{asset_id}_{current_start.isoformat()}',
                        'asset_id': asset_id,
                        'start_time': current_start,
                        'end_time': current_end,
                        'data_coverage': coverage,
                        'is_valid': coverage >= effective_config.min_coverage,
                    }
                )
                current_start = current_start + effective_config.step
    return pd.DataFrame(rows)


def _window_coverage(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    observed: pd.DataFrame,
    excluded: pd.DataFrame,
) -> float:
    """Return the effective observed fraction of one window.

    Args:
        start_time: Window start.
        end_time: Window end.
        observed: Observed periods of the same asset (``start_time``,
            ``end_time``).
        excluded: Excluded periods of the same asset; may be empty.

    Returns:
        float: ``(observed overlap - excluded overlap) / window duration``
        clipped to ``[0, 1]``. The denominator is at least one second.
    """
    total_seconds = max((end_time - start_time).total_seconds(), 1.0)

    observed_slice = observed[(observed['start_time'] < end_time) & (observed['end_time'] > start_time)]
    excluded_slice = excluded[(excluded['start_time'] < end_time) & (excluded['end_time'] > start_time)] if not excluded.empty else excluded

    observed_seconds = 0.0
    for row in observed_slice.itertuples(index=False):
        observed_seconds += _overlap_seconds(start_time, end_time, row.start_time, row.end_time)

    excluded_seconds = 0.0
    for row in excluded_slice.itertuples(index=False):
        excluded_seconds += _overlap_seconds(start_time, end_time, row.start_time, row.end_time)

    effective = max(observed_seconds - excluded_seconds, 0.0)
    return min(effective / total_seconds, 1.0)


def _overlap_seconds(left_start: pd.Timestamp, left_end: pd.Timestamp, right_start: pd.Timestamp, right_end: pd.Timestamp) -> float:
    """Return the overlap of two intervals in seconds (zero if disjoint).

    Args:
        left_start: Start of the first interval.
        left_end: End of the first interval.
        right_start: Start of the second interval (coerced with
            :func:`pandas.to_datetime`).
        right_end: End of the second interval (coerced likewise).

    Returns:
        float: Non-negative overlap duration in seconds.
    """
    overlap_start = max(left_start, pd.to_datetime(right_start))
    overlap_end = min(left_end, pd.to_datetime(right_end))
    return max((overlap_end - overlap_start).total_seconds(), 0.0)
