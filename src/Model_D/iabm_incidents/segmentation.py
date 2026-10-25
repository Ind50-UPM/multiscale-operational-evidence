"""Temporal episode segmentation for Model_D.

Role in the pipeline:
    :class:`~iabm_incidents.episodes.IncidentEpisodeBuilder` hands a scored
    window table (one ``deviation_score`` per window) to
    :class:`EpisodeSegmenter`, which returns index ranges of candidate
    episodes. The builder then attaches recovery, family and descriptive
    features to each range.

Main public objects:
    * :class:`EpisodeSegment` - index-based segment record.
    * :class:`EpisodeSegmenter` - onset/gap/duration segmentation.

Data contract:
    Window frame columns: ``start_time``, ``end_time`` (parsed with
    :func:`pandas.to_datetime`), ``deviation_score`` (non-numeric values are
    treated as ``0``) and, optionally, ``semantic_status``. Rows must be in
    chronological order for one asset and the index must be a default
    ``RangeIndex`` (see maintenance notes). Thresholds and durations come from
    :class:`~iabm_incidents.config.EpisodeDetectionConfig`.

.. mermaid::

   flowchart TD
       A[scored windows] --> B{semantic_status column?}
       B -- yes --> C[_segment_semantic: break on first non-candidate window or gap > maximum_gap]
       B -- no --> D[_segment_indicator: tolerate up to recovery_windows-1 non-candidate windows]
       C --> E[_append_if_valid]
       D --> E
       E --> F[_split_long_segment: split at local score minimum if span > maximum_duration]
       F --> G[keep groups with >= onset_windows windows and span >= minimum_duration]
       G --> H[peak = argmax score; _find_recovery_indices]
       H --> I[list of EpisodeSegment]

Notes:
    A window is an onset *candidate* when
    ``deviation_score >= onset_threshold``. The semantic route only uses the
    score mask (``semantic_status`` merely selects the route); the score
    itself already encodes the semantic label upstream.

Maintenance notes:
    * **Invariant:** indices stored in :class:`EpisodeSegment` are index
      labels, while :meth:`EpisodeSegmenter._find_recovery_indices` slices
      positionally with ``iloc[end_index + 1:]``. Both agree only for a
      ``0..n-1`` index.
    * The split heuristic searches a local minimum within +/- 6 windows of
      the nominal split point; this constant is hard-coded.
    * Recovery indices computed here mirror
      :class:`~iabm_incidents.recovery.RecoveryAssessor` but treat missing
      scores as ``0`` (i.e. as recovered), whereas the assessor does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import EpisodeDetectionConfig


@dataclass(frozen=True)
class EpisodeSegment:
    """Rich temporal segment describing one detected episode in the window frame.

    All indices are index labels of the window frame passed to
    :meth:`EpisodeSegmenter.segment`.

    Attributes:
        start_index: First window of the episode.
        peak_index: Window with the maximum ``deviation_score`` (first one on
            ties).
        end_index: Last candidate window of the episode.
        recovery_start_index: First window of the recovery run, or ``None``.
        recovery_end_index: Last window of the recovery run (equal to
            ``recovery_start_index`` for a partial recovery), or ``None``.
    """

    start_index: int
    peak_index: int
    end_index: int
    recovery_start_index: int | None = None
    recovery_end_index: int | None = None


class EpisodeSegmenter:
    """Create temporally robust segments from scored windows.

    The segmenter is stateless; all parameters come from the
    :class:`~iabm_incidents.config.EpisodeDetectionConfig` passed per call.
    """

    def segment(
        self,
        window_frame: pd.DataFrame,
        detection_config: EpisodeDetectionConfig,
    ) -> list[EpisodeSegment]:
        """Segment a scored window frame into candidate episodes.

        Args:
            window_frame: Chronological window-score table (see module data
                contract). Not modified.
            detection_config: Onset/recovery thresholds (score units), onset
                and recovery window counts, ``minimum_duration``,
                ``maximum_gap`` and ``maximum_duration`` (timedeltas; a
                non-positive maximum duration disables splitting).

        Returns:
            list[EpisodeSegment]: Segments in chronological order; empty for
            an empty frame.

        Raises:
            KeyError: If ``start_time``, ``end_time`` or ``deviation_score``
                is missing.
        """
        if window_frame.empty:
            return []

        frame = window_frame.copy()
        frame["start_time"] = pd.to_datetime(frame["start_time"])
        frame["end_time"] = pd.to_datetime(frame["end_time"])
        scores = pd.to_numeric(frame["deviation_score"], errors="coerce").fillna(0.0)
        candidate_mask = scores.ge(detection_config.onset_threshold)

        if "semantic_status" in frame.columns:
            return self._segment_semantic(frame, candidate_mask, detection_config)
        return self._segment_indicator(frame, candidate_mask, detection_config)

    def _segment_semantic(
        self,
        frame: pd.DataFrame,
        candidate_mask: pd.Series,
        detection_config: EpisodeDetectionConfig,
    ) -> list[EpisodeSegment]:
        """Strict run segmentation used when ``semantic_status`` is present.

        A run of consecutive candidate windows is closed by the first
        non-candidate window or by a gap ``start_time - previous end_time``
        larger than ``maximum_gap``.

        Args:
            frame: Window frame with parsed timestamps.
            candidate_mask: Boolean onset-candidate mask aligned with
                ``frame.index`` (missing labels are treated as ``False``).
            detection_config: Detection configuration.

        Returns:
            list[EpisodeSegment]: Validated segments.
        """
        segments: list[EpisodeSegment] = []
        current_indices: list[int] = []
        previous_end_time: pd.Timestamp | None = None

        index_values = frame.index.to_list()
        start_times = frame['start_time'].to_list()
        end_times = frame['end_time'].to_list()
        candidate_values = candidate_mask.reindex(frame.index, fill_value=False).to_numpy(dtype=bool)

        for position, index in enumerate(index_values):
            if not candidate_values[position]:
                if current_indices:
                    self._append_if_valid(frame, current_indices, segments, detection_config)
                    current_indices = []
                    previous_end_time = None
                continue

            if current_indices and previous_end_time is not None:
                gap = start_times[position] - previous_end_time
                if gap > detection_config.maximum_gap:
                    self._append_if_valid(frame, current_indices, segments, detection_config)
                    current_indices = []

            current_indices.append(index)
            previous_end_time = end_times[position]

        if current_indices:
            self._append_if_valid(frame, current_indices, segments, detection_config)
        return segments

    def _segment_indicator(
        self,
        frame: pd.DataFrame,
        candidate_mask: pd.Series,
        detection_config: EpisodeDetectionConfig,
    ) -> list[EpisodeSegment]:
        """Tolerant run segmentation for indicator-only window frames.

        Up to ``recovery_windows - 1`` non-candidate windows may interrupt a
        run without closing it; the run closes at the ``recovery_windows``-th
        consecutive non-candidate window or at a gap larger than
        ``maximum_gap``. Interrupting windows are *not* added to the segment,
        so a segment spans its first to its last candidate window.

        Args:
            frame: Window frame with parsed timestamps.
            candidate_mask: Boolean onset-candidate mask indexed like
                ``frame``.
            detection_config: Detection configuration.

        Returns:
            list[EpisodeSegment]: Validated segments.
        """
        segments: list[EpisodeSegment] = []
        current_candidate_indices: list[int] = []
        pending_gap_indices: list[int] = []
        previous_end_time: pd.Timestamp | None = None

        for index, row in frame.iterrows():
            if current_candidate_indices and previous_end_time is not None:
                gap_duration = row["start_time"] - previous_end_time
                if gap_duration > detection_config.maximum_gap:
                    self._append_if_valid(frame, current_candidate_indices, segments, detection_config)
                    current_candidate_indices = []
                    pending_gap_indices = []

            if bool(candidate_mask.loc[index]):
                current_candidate_indices.append(index)
                pending_gap_indices = []
                previous_end_time = row["end_time"]
                continue

            if current_candidate_indices:
                pending_gap_indices.append(index)
                previous_end_time = row["end_time"]
                if len(pending_gap_indices) >= detection_config.recovery_windows:
                    self._append_if_valid(frame, current_candidate_indices, segments, detection_config)
                    current_candidate_indices = []
                    pending_gap_indices = []
                    previous_end_time = None

        if current_candidate_indices:
            self._append_if_valid(frame, current_candidate_indices, segments, detection_config)
        return segments

    @staticmethod
    def _append_if_valid(
        window_frame: pd.DataFrame,
        indices: list[int],
        segments: list[EpisodeSegment],
        detection_config: EpisodeDetectionConfig,
    ) -> None:
        """Validate a raw run, split it if too long and append its segments.

        Each (sub)group must contain at least ``onset_windows`` windows and
        span at least ``minimum_duration`` (first ``start_time`` to last
        ``end_time``). The peak is the first maximum of ``deviation_score``
        and recovery indices are searched after the last window.

        Args:
            window_frame: Full window frame (needed for recovery lookup).
            indices: Candidate window labels of the run, in order.
            segments: Output list, extended in place.
            detection_config: Detection configuration.

        Returns:
            None
        """
        if len(indices) < detection_config.onset_windows:
            return
        split_groups = EpisodeSegmenter._split_long_segment(window_frame, indices, detection_config)
        for group in split_groups:
            if len(group) < detection_config.onset_windows:
                continue
            start_row = window_frame.loc[group[0]]
            end_row = window_frame.loc[group[-1]]
            if end_row["end_time"] - start_row["start_time"] < detection_config.minimum_duration:
                continue

            scores = pd.to_numeric(window_frame.loc[group, "deviation_score"], errors="coerce").fillna(0.0)
            peak_index = int(scores.idxmax())
            recovery_start_index, recovery_end_index = EpisodeSegmenter._find_recovery_indices(
                window_frame,
                end_index=group[-1],
                recovery_threshold=detection_config.recovery_threshold,
                recovery_windows=detection_config.recovery_windows,
            )
            segments.append(
                EpisodeSegment(
                    start_index=group[0],
                    peak_index=peak_index,
                    end_index=group[-1],
                    recovery_start_index=recovery_start_index,
                    recovery_end_index=recovery_end_index,
                )
            )

    @staticmethod
    def _split_long_segment(
        window_frame: pd.DataFrame,
        indices: list[int],
        detection_config: EpisodeDetectionConfig,
    ) -> list[list[int]]:
        """Split a run whose span exceeds ``maximum_duration``.

        Starting from the beginning of the run, the remainder is cut at the
        position chosen by :meth:`_choose_split_index` until every piece
        fits. If no admissible split point exists the remainder is halved,
        which guarantees termination (each piece has at least one window).

        Args:
            window_frame: Window frame providing timestamps and scores.
            indices: Window labels of the run.
            detection_config: Provides ``maximum_duration``; values
                ``<= 0`` disable splitting.

        Returns:
            list[list[int]]: Consecutive, non-empty groups covering
            ``indices``; ``[]`` for an empty input.
        """
        if not indices:
            return []
        maximum_duration = detection_config.maximum_duration
        if maximum_duration <= pd.Timedelta(0):
            return [indices]
        groups: list[list[int]] = []
        pending = list(indices)
        while pending:
            start_index = pending[0]
            start_time = pd.to_datetime(window_frame.loc[start_index, "start_time"])
            end_time = pd.to_datetime(window_frame.loc[pending[-1], "end_time"])
            if end_time - start_time <= maximum_duration:
                groups.append(pending)
                break
            split_at = EpisodeSegmenter._choose_split_index(window_frame, pending, start_time + maximum_duration)
            if split_at <= 0 or split_at >= len(pending):
                split_at = max(1, min(len(pending) - 1, len(pending) // 2))
            groups.append(pending[:split_at])
            pending = pending[split_at:]
        return groups

    @staticmethod
    def _choose_split_index(
        window_frame: pd.DataFrame,
        indices: list[int],
        target_time: pd.Timestamp,
    ) -> int:
        """Choose a split position near ``target_time`` at a local score minimum.

        The anchor is the first window starting at or after ``target_time``;
        the lowest-scoring window within six positions of it (excluding
        position 0) becomes the first window of the next piece.

        Args:
            window_frame: Window frame providing ``start_time`` and
                ``deviation_score``.
            indices: Window labels of the remaining run.
            target_time: Nominal cut time (``start + maximum_duration``).

        Returns:
            int: Split position in ``[1, len(indices) - 1]``, or
            ``len(indices)`` when no window starts after ``target_time``
            (the caller then falls back to halving).
        """
        candidate_positions = [
            position
            for position, index in enumerate(indices)
            if pd.to_datetime(window_frame.loc[index, "start_time"]) >= target_time
        ]
        if not candidate_positions:
            return len(indices)
        anchor = candidate_positions[0]
        left = max(1, anchor - 6)
        right = min(len(indices) - 1, anchor + 6)
        candidate_slice = indices[left:right + 1]
        scores = pd.to_numeric(window_frame.loc[candidate_slice, "deviation_score"], errors="coerce").fillna(0.0)
        local_min_index = int(scores.idxmin())
        chosen = indices.index(local_min_index)
        return max(1, min(len(indices) - 1, chosen))

    @staticmethod
    def _find_recovery_indices(
        window_frame: pd.DataFrame,
        *,
        end_index: int,
        recovery_threshold: float,
        recovery_windows: int,
    ) -> tuple[int | None, int | None]:
        """Locate the recovery run after an episode.

        Args:
            window_frame: Window frame with a default ``RangeIndex``.
            end_index: Last window of the episode (label == position).
            recovery_threshold: Score below which a window is recovered.
            recovery_windows: Consecutive recovered windows required.

        Returns:
            tuple[int | None, int | None]: ``(start, end)`` labels of the first
            run of ``recovery_windows`` recovered windows; otherwise the first
            single recovered window twice; otherwise ``(None, None)``.
        """
        # Note: positional slice; assumes labels equal positions (RangeIndex).
        following = window_frame.iloc[end_index + 1 :].copy()
        if following.empty:
            return None, None

        below = pd.to_numeric(following["deviation_score"], errors="coerce").fillna(0.0).lt(recovery_threshold)
        if len(following) >= recovery_windows:
            sustained = below.rolling(recovery_windows, min_periods=recovery_windows).sum().eq(recovery_windows)
            if bool(sustained.any()):
                recovery_end_index = int(sustained[sustained].index[0])
                recovery_position = following.index.get_loc(recovery_end_index)
                recovery_start_index = int(following.index[recovery_position - recovery_windows + 1])
                return recovery_start_index, recovery_end_index
        if bool(below.any()):
            recovery_index = int(below[below].index[0])
            return recovery_index, recovery_index
        return None, None
