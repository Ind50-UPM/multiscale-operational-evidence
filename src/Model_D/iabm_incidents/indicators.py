"""Indicator aggregation utilities for the indicator-driven Model_D route.

Role in the pipeline
--------------------
Row-level indicators (one row per Model_B sequence or per electrical scoring
window) are aggregated onto the analysis windows used for segmentation.
Behavioural indicators are centred by a robust nominal median; electrical
scores, which already encode a frozen nominal reference, are kept raw.

Main public objects
-------------------
* :class:`IndicatorBaseline` -- per-column nominal medians.
* :func:`fit_indicator_baseline` -- fits the medians.
* :func:`compute_window_indicators` -- aggregates rows into windows.

Data contracts
--------------
* Source rows: ``start_time``/``end_time`` (or ``window_start``/``window_end``),
  optional ``asset_id`` (missing becomes ``"asset-unknown"``),
  ``semantic_status`` (``"ANOMALOUS"`` or other, missing treated as
  ``"NORMAL"``), ``incident_family`` and any indicator columns.
* Windows: ``start_time``, ``end_time``, optional ``window_id``,
  ``asset_id`` and ``data_coverage``. Windows are half-open; a source row
  contributes when its interval overlaps the window with positive length or
  lies strictly inside it (``row.start < window.end`` and
  ``row.end > window.start``).
* All timestamps are converted to naive UTC; unparsable values are dropped.

Output columns: ``window_id``, ``asset_id``, ``start_time``, ``end_time``,
``data_coverage`` (default 1.0), ``sequence_count``,
``active_sequence_count`` (currently identical to ``sequence_count``),
``semantic_status`` (``"ANOMALOUS"`` if any overlapping row is),
``incident_family``, the centred indicators, and raw electrical columns.
Windows without overlapping rows are omitted.

Maintenance notes
-----------------
* Missing behavioural values are filled with 0 *before* averaging and then
  the baseline median is subtracted.
* ``command_response_deviation`` (mean) and ``transient_concentration``
  (maximum) overwrite any centred value with the raw, NaN-aware aggregate, so
  unsupported electrical windows stay NaN instead of becoming 0.
* :func:`_dominant_family` (frame version) is kept for compatibility; the
  vectorized path uses :func:`_dominant_family_from_arrays`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IndicatorBaseline:
    """Simple nominal baseline used to center window indicators.

    Attributes:
        medians: Mapping from indicator column to its nominal median. Only
            these columns are centred (and, except electrical columns,
            emitted) by :func:`compute_window_indicators`.
    """

    medians: dict[str, float]


def fit_indicator_baseline(frame: pd.DataFrame, columns: list[str] | None = None) -> IndicatorBaseline:
    """Fit a robust median baseline from nominal rows when available.

    Args:
        frame: Row-level indicator table.
        columns: Columns to fit; ``None`` (or an empty list) selects the
            default behavioural and electrical indicator set.

    Returns:
        :class:`IndicatorBaseline` with medians for each requested column that
        exists and has at least one non-null value.

    Notes:
        Rows with ``semantic_status == "ANOMALOUS"`` are excluded; if that
        leaves nothing, all rows are used. This is the leakage guard against
        fitting the centre on documented anomalies.
    """

    candidate_columns = columns or [
        'sequence_divergence', 'duration_drift', 'recurrence_excess', 'persistence_excess',
        'consumption_deviation', 'state_error_rate', 'mode_divergence',
        'command_response_deviation', 'transient_concentration',
    ]
    nominal = frame.copy()
    if 'semantic_status' in nominal.columns:
        nominal = nominal[nominal['semantic_status'].ne('ANOMALOUS')]
    if nominal.empty:
        nominal = frame.copy()
    medians = {
        column: float(pd.to_numeric(nominal[column], errors='coerce').dropna().median())
        for column in candidate_columns
        if column in nominal.columns and nominal[column].notna().any()
    }
    return IndicatorBaseline(medians=medians)


def compute_window_indicators(source_rows: pd.DataFrame, windows: pd.DataFrame, baseline: IndicatorBaseline | None = None) -> pd.DataFrame:
    """Aggregate row-level indicators into the supplied windows.

    Args:
        source_rows: Row-level indicator table (see module data contract).
        windows: Analysis windows.
        baseline: Optional pre-fitted baseline; when ``None`` it is fitted on
            ``source_rows`` with :func:`fit_indicator_baseline`.

    Returns:
        One row per window with at least one overlapping source row of the
        same asset. An empty frame with ``window_id``, ``asset_id``,
        ``start_time`` and ``end_time`` is returned when an input is empty or
        no window has valid timestamps.
    """

    if source_rows.empty or windows.empty:
        return pd.DataFrame(columns=['window_id', 'asset_id', 'start_time', 'end_time'])

    frame = source_rows.copy()
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    if 'window_start' in frame.columns and 'start_time' not in frame.columns:
        frame['start_time'] = frame['window_start']
    if 'window_end' in frame.columns and 'end_time' not in frame.columns:
        frame['end_time'] = frame['window_end']
    frame['start_time'] = _normalize_timestamp_series(frame['start_time'])
    frame['end_time'] = _normalize_timestamp_series(frame['end_time'])
    frame = frame.dropna(subset=['start_time', 'end_time'])
    baseline_values = (baseline or fit_indicator_baseline(frame)).medians

    window_frame = windows.copy()
    window_frame = window_frame.loc[:, ~window_frame.columns.duplicated()].copy()
    window_frame['start_time'] = _normalize_timestamp_series(window_frame['start_time'])
    window_frame['end_time'] = _normalize_timestamp_series(window_frame['end_time'])
    window_frame = window_frame.dropna(subset=['start_time', 'end_time']).reset_index(drop=True)
    if window_frame.empty:
        return pd.DataFrame(columns=['window_id', 'asset_id', 'start_time', 'end_time'])

    if 'asset_id' not in frame.columns:
        frame['asset_id'] = 'asset-unknown'
    frame['asset_id'] = frame['asset_id'].fillna('asset-unknown').astype(str)

    if 'asset_id' not in window_frame.columns:
        window_frame['asset_id'] = 'asset-unknown'
    window_frame['asset_id'] = window_frame['asset_id'].fillna('asset-unknown').astype(str)

    rows: list[dict[str, object]] = []
    for asset_id, asset_windows in window_frame.groupby('asset_id', dropna=False, sort=False):
        asset_rows = frame[frame['asset_id'] == asset_id]
        if asset_rows.empty:
            continue
        rows.extend(_compute_asset_window_indicators(asset_rows, asset_windows, baseline_values))
    return pd.DataFrame(rows)


def _compute_asset_window_indicators(
    asset_rows: pd.DataFrame,
    asset_windows: pd.DataFrame,
    baseline_values: dict[str, float],
) -> list[dict[str, object]]:
    """Aggregate indicators for one asset using NumPy overlap masks.

    Args:
        asset_rows: Source rows of one asset with normalized timestamps.
        asset_windows: Windows of the same asset.
        baseline_values: Column medians to subtract.

    Returns:
        List of output row dictionaries (see module data contract).
    """

    starts = asset_rows['start_time'].to_numpy(dtype='datetime64[ns]').astype('int64')
    ends = asset_rows['end_time'].to_numpy(dtype='datetime64[ns]').astype('int64')
    semantic_values = (
        asset_rows['semantic_status'].fillna('NORMAL').astype(str).to_numpy()
        if 'semantic_status' in asset_rows.columns
        else np.full(len(asset_rows), 'NORMAL', dtype=object)
    )
    family_values = (
        asset_rows['incident_family'].fillna('unclassified_incident').astype(str).to_numpy()
        if 'incident_family' in asset_rows.columns
        else np.full(len(asset_rows), 'unclassified_incident', dtype=object)
    )
    indicator_arrays = {
        column: pd.to_numeric(asset_rows[column], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        for column in baseline_values
        if column in asset_rows.columns
    }

    rows: list[dict[str, object]] = []
    for window in asset_windows.itertuples(index=False):
        window_start = _normalize_scalar_timestamp(getattr(window, 'start_time'))
        window_end = _normalize_scalar_timestamp(getattr(window, 'end_time'))
        window_start_ns = window_start.to_datetime64().astype('datetime64[ns]').astype('int64')
        window_end_ns = window_end.to_datetime64().astype('datetime64[ns]').astype('int64')
        # Invariant: overlap test against the half-open window [start, end).
        mask = (starts < window_end_ns) & (ends > window_start_ns)
        if not bool(mask.any()):
            continue

        active_semantic = semantic_values[mask]
        active_family = family_values[mask]
        row = {
            'window_id': getattr(window, 'window_id', None),
            'asset_id': getattr(window, 'asset_id', None),
            'start_time': window_start,
            'end_time': window_end,
            'data_coverage': getattr(window, 'data_coverage', 1.0),
            'sequence_count': int(mask.sum()),
            'active_sequence_count': int(mask.sum()),
            'semantic_status': 'ANOMALOUS' if bool((active_semantic == 'ANOMALOUS').any()) else 'NORMAL',
            'incident_family': _dominant_family_from_arrays(active_semantic, active_family),
        }
        for column, baseline_value in baseline_values.items():
            if column not in indicator_arrays:
                continue
            row[column] = float(indicator_arrays[column][mask].mean() - baseline_value)
        # Electrical scores already encode a frozen nominal reference.
        # Preserve raw nonnegative evidence and missing support, without recentering.
        for column in ("command_response_deviation", "transient_concentration"):
            if column in asset_rows:
                values = pd.to_numeric(asset_rows.loc[mask, column], errors="coerce")
                row[column] = float(values.max() if column == "transient_concentration" else values.mean())
        for column in asset_rows.columns:
            if column.startswith("command_response_signed_") or column == "command_response_imbalance":
                row[column] = float(pd.to_numeric(asset_rows.loc[mask, column], errors="coerce").mean())
        rows.append(row)
    return rows


def _dominant_family(frame: pd.DataFrame) -> str:
    """Return the modal family of a frame, preferring anomalous rows.

    Args:
        frame: Rows with ``incident_family`` and optionally
            ``semantic_status``.

    Returns:
        The first modal family among anomalous rows (or all rows when none is
        anomalous), or ``"unclassified_incident"``.
    """
    if 'incident_family' not in frame.columns:
        return 'unclassified_incident'
    anomalous = frame[frame.get('semantic_status', pd.Series(index=frame.index)).eq('ANOMALOUS')]
    target = anomalous if not anomalous.empty else frame
    mode = target['incident_family'].mode(dropna=True)
    return str(mode.iloc[0]) if not mode.empty else 'unclassified_incident'


def _dominant_family_from_arrays(semantic_values: np.ndarray, family_values: np.ndarray) -> str:
    """Return the dominant family giving priority to anomalous rows.

    Args:
        semantic_values: Semantic statuses of the overlapping rows.
        family_values: Families of the same rows.

    Returns:
        The first modal family among anomalous rows (or all rows when none is
        anomalous), or ``"unclassified_incident"`` when empty.
    """

    anomalous = family_values[semantic_values == 'ANOMALOUS']
    target = anomalous if anomalous.size else family_values
    if not target.size:
        return 'unclassified_incident'
    mode = pd.Series(target).mode(dropna=True)
    return str(mode.iloc[0]) if not mode.empty else 'unclassified_incident'


def _normalize_timestamp_series(values: pd.Series) -> pd.Series:
    """Parse values as UTC and return naive-UTC timestamps.

    Args:
        values: Series of timestamp-like values.

    Returns:
        ``datetime64`` series; unparsable values become ``NaT``.
    """
    converted = pd.to_datetime(values, errors='coerce', utc=True)
    if getattr(converted.dtype, 'tz', None) is not None:
        converted = converted.dt.tz_convert(None)
    return converted


def _normalize_scalar_timestamp(value: object) -> pd.Timestamp:
    """Parse one value as UTC and return a naive-UTC timestamp.

    Args:
        value: Timestamp-like scalar.

    Returns:
        Naive :class:`pandas.Timestamp`, or ``NaT`` when unparsable.
    """
    timestamp = pd.to_datetime(value, errors='coerce', utc=True)
    if pd.isna(timestamp):
        return timestamp
    if getattr(timestamp, 'tzinfo', None) is not None:
        return timestamp.tz_convert(None)
    return timestamp
