"""Longitudinal episode construction logic.

Role in the pipeline
--------------------
This module turns window-level deviation evidence into incident episodes. It is
responsible for chaining segmentation, recovery assessment, and family
assignment into a stable tabular contract that downstream evaluation and
occurrence modules can consume.

Two routes converge in :meth:`IncidentEpisodeBuilder.build_episodes_from_segments`:

* the *semantic* route, where Model_B sequences and Model_C assignments are
  merged into window scores by :func:`~iabm_incidents.detection.build_window_scores`;
* the *indicator* route, where a window-score table (for example built by
  :mod:`iabm_incidents.indicators`) is already available.

.. mermaid::

    flowchart LR
        S["Model_B sequences"] --> W["build_window_scores"]
        A["Model_C assignments"] --> W
        W --> G["EpisodeSegmenter.segment"]
        I["precomputed window scores"] --> G
        G --> C["CandidateSegment list"]
        C --> F["_finalize_episode<br/>RecoveryAssessor + RuleBasedFamilyClassifier"]
        F --> E["episode DataFrame"]

Main public objects
-------------------
* :class:`CandidateSegment` -- serializable segment contract.
* :class:`IncidentEpisode` -- typed episode record (one output row).
* :class:`IncidentEpisodeBuilder` -- orchestrates both routes.

Data contracts
--------------
* Window-score frames must contain ``start_time``, ``end_time`` (timestamps)
  and ``deviation_score`` (float); ``asset_id`` and ``semantic_status`` are
  optional. Rows are addressed by *label* with ``.loc`` using the integer
  positions produced by the segmenter, so the frame is expected to carry a
  default ``RangeIndex`` (labels equal positions).
* Episode frames carry exactly the columns listed in
  :meth:`IncidentEpisodeBuilder._empty_episode_frame`, in that order, which
  match the fields of :class:`IncidentEpisode`.
* ``event_start`` is the ``start_time`` of the first window and ``event_end``
  the ``end_time`` of the last window of the segment.

Maintenance notes
-----------------
* Keep the field order of :class:`IncidentEpisode` and the column list of
  ``_empty_episode_frame`` synchronized; empty and non-empty outputs must share
  one schema.
* ``registry_incident_id`` and ``label_strength`` are always ``None`` here;
  they are populated downstream when episodes are matched to the registry.
* ``episode_id`` is derived from window positions and is only unique within
  one window-score table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from .classification import RuleBasedFamilyClassifier
from .config import ModelDConfig
from .detection import build_window_scores
from .recovery import RecoveryAssessor
from .segmentation import EpisodeSegmenter


@dataclass(frozen=True)
class CandidateSegment:
    """Serializable segment contract for the indicator-driven route.

    Attributes:
        candidate_id: Stable identifier for the candidate segment.
        start_index: Inclusive start index in the source score table.
        end_index: Inclusive end index in the source score table.
        event_start: Timestamp of the first anomalous window.
        peak_time: Timestamp of the segment peak.
        event_end: Timestamp of the last anomalous window.
        recovery_start: Optional timestamp when recovery begins.
        recovery_end: Optional timestamp when recovery ends.
        source_window_ids: Window identifiers participating in the segment.
    """

    candidate_id: str
    start_index: int
    end_index: int
    event_start: pd.Timestamp
    peak_time: pd.Timestamp
    event_end: pd.Timestamp
    recovery_start: pd.Timestamp | None = None
    recovery_end: pd.Timestamp | None = None
    source_window_ids: tuple[str, ...] = tuple()


@dataclass(frozen=True)
class IncidentEpisode:
    """Typed representation of one detected incident episode.

    Instances are converted to rows with ``episode.__dict__``; the field order
    defines the column order of the episode frame.

    Attributes:
        episode_id: ``"episode_<start>_<end>"`` from window positions.
        event_start: ``start_time`` of the first window of the episode.
        event_end: ``end_time`` of the last window of the episode.
        pre_event_start: ``event_start`` minus the pre-event look-back window.
        peak_time: ``start_time`` of the peak window.
        recovery_start: Start of confirmed or segmenter-provided recovery, or
            ``None``.
        recovery_end: End of recovery, or ``None``.
        primary_family: Assigned incident family (or
            ``"unclassified_incident"``), possibly ``None``.
        secondary_families: Alternative families with positive scores.
        family_confidence: Confidence of the primary assignment.
        assignment_method: Identifier of the assignment method.
        evidence: Human-readable evidence strings from the classifier.
        onset_score: ``deviation_score`` of the first window.
        peak_score: Maximum ``deviation_score`` over the episode windows.
        mean_score: Mean ``deviation_score`` over the episode windows.
        duration_seconds: ``event_end - event_start`` in seconds.
        time_to_recovery_seconds: Seconds from ``event_end`` to recovery end,
            or ``None``.
        recovery_status: String value of
            :class:`~iabm_incidents.recovery.RecoveryStatus`.
        source_sequence_indices: Window positions spanned by the episode
            (inclusive range).
        source_window_ids: Window identifiers of the episode.
        asset_id: Asset of the first window when the column exists.
        registry_incident_id: Matched registry incident (filled downstream).
        label_strength: Label strength of the matched incident (filled
            downstream).
    """

    episode_id: str
    event_start: pd.Timestamp
    event_end: pd.Timestamp
    pre_event_start: pd.Timestamp
    peak_time: pd.Timestamp
    recovery_start: pd.Timestamp | None
    recovery_end: pd.Timestamp | None
    primary_family: str | None
    secondary_families: tuple[str, ...]
    family_confidence: float | None
    assignment_method: str | None
    evidence: tuple[str, ...]
    onset_score: float
    peak_score: float
    mean_score: float
    duration_seconds: float
    time_to_recovery_seconds: float | None
    recovery_status: str
    source_sequence_indices: tuple[int, ...]
    source_window_ids: tuple[str, ...]
    asset_id: str | None
    registry_incident_id: str | None = None
    label_strength: str | None = None


class IncidentEpisodeBuilder:
    """Build incident episodes from sequence-level and semantic outputs.

    Args:
        config: Optional typed configuration governing window and detection
            behavior.

    Notes:
        The builder supports two routes: a backward-compatible semantic route
        where sequence tables and assignment tables are merged into window
        scores, and an indicator-driven route where window scores are already
        available from upstream processing.
    """

    def __init__(self, config: ModelDConfig | None = None) -> None:
        """Create a builder with default segmenter, assessor and classifier.

        Args:
            config: Model D configuration; ``None`` uses ``ModelDConfig()``.
        """
        self._config = config or ModelDConfig()
        self._segmenter = EpisodeSegmenter()
        self._recovery = RecoveryAssessor()
        self._classifier = RuleBasedFamilyClassifier()

    def build(
        self,
        sequences: pd.DataFrame,
        assignments: pd.DataFrame,
        *,
        pre_event_window: pd.Timedelta | None = None,
    ) -> pd.DataFrame:
        """Compatibility wrapper for the semantic assignment route.

        Args:
            sequences: Model_B sequence table.
            assignments: Model_C assignment table.
            pre_event_window: Optional look-back horizon; defaults to the
                configured window step.

        Returns:
            Episode dataframe, see
            :meth:`build_episodes_from_semantic_assignments`.
        """
        return self.build_episodes_from_semantic_assignments(
            sequences,
            assignments,
            pre_event_window=pre_event_window,
        )

    def build_episodes_from_semantic_assignments(
        self,
        sequences: pd.DataFrame,
        assignments: pd.DataFrame,
        *,
        pre_event_window: pd.Timedelta | None = None,
        window_scores: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Construct episodes from sequences plus semantic assignments.

        Args:
            sequences: Model_B sequence table.
            assignments: Model_C assignment table.
            pre_event_window: Optional look-back horizon attached to each
                episode for contextual analysis.
            window_scores: Optional precomputed semantic window-score table.
                When provided, the method reuses it instead of rebuilding the
                same windows internally.

        Returns:
            Episode dataframe derived from merged semantic window scores; an
            empty frame with the canonical schema when either input is empty.

        Notes:
            ``pre_event_window`` falls back to ``config.window.step`` when it is
            ``None`` (or otherwise falsy, e.g. a zero ``Timedelta``).
        """
        if sequences.empty or assignments.empty:
            return self._empty_episode_frame()

        merged = window_scores.copy() if window_scores is not None else self.build_window_scores(sequences, assignments)
        effective_pre_event_window = pre_event_window or self._config.window.step
        segments = self._segmenter.segment(merged, self._config.detection)
        candidate_segments = self._candidate_segments_from_frame(segments, merged)
        return self.build_episodes_from_segments(
            candidate_segments,
            merged,
            pre_event_window=effective_pre_event_window,
        )

    def build_episodes_from_window_scores(
        self,
        window_scores: pd.DataFrame,
        *,
        pre_event_window: pd.Timedelta | None = None,
    ) -> pd.DataFrame:
        """Construct episodes directly from a precomputed window-score table.

        Args:
            window_scores: Window-score table with ``start_time``,
                ``end_time`` and ``deviation_score`` and a ``RangeIndex``.
            pre_event_window: Optional look-back horizon; falls back to the
                configured window step when falsy.

        Returns:
            Episode dataframe; an empty frame with the canonical schema when
            ``window_scores`` is empty or no segment is found.
        """
        if window_scores.empty:
            return self._empty_episode_frame()

        effective_pre_event_window = pre_event_window or self._config.window.step
        segments = self._segmenter.segment(window_scores, self._config.detection)
        candidate_segments = self._candidate_segments_from_frame(segments, window_scores)
        return self.build_episodes_from_segments(
            candidate_segments,
            window_scores,
            pre_event_window=effective_pre_event_window,
        )

    def build_episodes_from_segments(
        self,
        segments: Sequence[CandidateSegment],
        indicators: pd.DataFrame,
        *,
        pre_event_window: pd.Timedelta | None = None,
    ) -> pd.DataFrame:
        """Finalize typed episodes from candidate segments and indicator rows.

        Args:
            segments: Candidate segments emitted by the segmenter.
            indicators: Window-score or indicator dataframe used as source.
            pre_event_window: Optional context window attached before each onset.

        Returns:
            Final episode dataframe ready for evaluation and occurrence
            analysis; an empty canonical frame when ``segments`` or
            ``indicators`` is empty.

        Notes:
            This method is the convergence point for both semantic and
            indicator-driven processing routes.
        """
        if not segments or indicators.empty:
            return self._empty_episode_frame()

        effective_pre_event_window = pre_event_window or self._config.window.step
        episodes: list[dict[str, object]] = []
        for segment in segments:
            episode = self._finalize_episode(
                merged=indicators,
                start_index=segment.start_index,
                end_index=segment.end_index,
                pre_event_window=effective_pre_event_window,
                source_window_ids=segment.source_window_ids,
                candidate_segment=segment,
            )
            episodes.append(episode.__dict__)
        return pd.DataFrame(episodes)

    def build_window_scores(
        self,
        sequences: pd.DataFrame,
        assignments: pd.DataFrame,
    ) -> pd.DataFrame:
        """Derive window scores from sequence and assignment tables.

        Args:
            sequences: Model_B sequence table.
            assignments: Model_C assignment table.

        Returns:
            Window-score dataframe from
            :func:`~iabm_incidents.detection.build_window_scores` using the
            configured weights and window settings.
        """
        return build_window_scores(sequences, assignments, self._config.weights, self._config.window)

    def _finalize_episode(
        self,
        merged: pd.DataFrame,
        start_index: int,
        end_index: int,
        pre_event_window: pd.Timedelta,
        source_window_ids: tuple[str, ...] = tuple(),
        candidate_segment: CandidateSegment | None = None,
    ) -> IncidentEpisode:
        """Finalize one incident episode from a segmented interval.

        Args:
            merged: Window-score frame with a ``RangeIndex``.
            start_index: Inclusive first window position.
            end_index: Inclusive last window position.
            pre_event_window: Look-back subtracted from ``event_start``.
            source_window_ids: Window identifiers; when empty the positions
                are used as strings.
            candidate_segment: Optional segment providing peak time and
                recovery bounds that override the defaults.

        Returns:
            The typed :class:`IncidentEpisode`.

        Notes:
            Recovery assessment and family assignment are intentionally computed
            after segmentation so they can observe the whole interval rather than
            only the onset window.

            When the candidate segment carries a ``recovery_start``, its
            recovery bounds replace those of the
            :class:`~iabm_incidents.recovery.RecoveryAssessor`, while the
            assessor's ``recovery_status`` is kept. The peak is the window
            whose ``start_time`` equals ``candidate_segment.peak_time`` (falling
            back to the segment start); without a candidate segment it is the
            arg-max of ``deviation_score``.
        """
        indices = list(range(start_index, end_index + 1))
        episode_frame = merged.loc[indices]
        start_row = merged.loc[start_index]
        end_row = merged.loc[end_index]
        recovery = self._recovery.assess(
            merged,
            last_event_index=end_index,
            recovery_threshold=self._config.detection.recovery_threshold,
            recovery_windows=self._config.detection.recovery_windows,
        )
        if candidate_segment is not None and candidate_segment.recovery_start is not None:
            recovery_end = candidate_segment.recovery_end
            recovery = type(recovery)(
                recovery_start=candidate_segment.recovery_start,
                recovery_end=recovery_end,
                recovery_status=recovery.recovery_status,
                time_to_recovery_seconds=(recovery_end - end_row["end_time"]).total_seconds() if recovery_end is not None else recovery.time_to_recovery_seconds,
            )
        family_assignment = self._classifier.assign(episode_frame, self._config.family_assignment)
        peak_index = candidate_segment.start_index if candidate_segment is not None else start_index
        if candidate_segment is not None:
            # Note: the conditional below always adds 0; the effective peak is
            # resolved by matching ``peak_time`` against ``start_time``.
            peak_index = candidate_segment.start_index + (0 if candidate_segment.peak_time is None else 0)
            peak_matches = merged.index[merged["start_time"].eq(candidate_segment.peak_time)]
            if len(peak_matches) > 0:
                peak_index = int(peak_matches[0])
        else:
            peak_index = int(pd.to_numeric(episode_frame["deviation_score"], errors="coerce").fillna(0.0).idxmax())

        return IncidentEpisode(
            episode_id=f"episode_{start_index}_{end_index}",
            event_start=start_row["start_time"],
            event_end=end_row["end_time"],
            pre_event_start=start_row["start_time"] - pre_event_window,
            peak_time=merged.loc[peak_index, "start_time"],
            recovery_start=recovery.recovery_start,
            recovery_end=recovery.recovery_end,
            primary_family=family_assignment.primary_family,
            secondary_families=family_assignment.secondary_families,
            family_confidence=family_assignment.family_confidence,
            assignment_method=family_assignment.assignment_method,
            evidence=family_assignment.evidence,
            onset_score=float(start_row["deviation_score"]),
            peak_score=float(episode_frame["deviation_score"].max()),
            mean_score=float(episode_frame["deviation_score"].mean()),
            duration_seconds=(end_row["end_time"] - start_row["start_time"]).total_seconds(),
            time_to_recovery_seconds=recovery.time_to_recovery_seconds,
            recovery_status=recovery.recovery_status.value,
            source_sequence_indices=tuple(indices),
            source_window_ids=source_window_ids or tuple(str(index) for index in indices),
            asset_id=start_row.get("asset_id") if "asset_id" in merged.columns else None,
        )

    @staticmethod
    def _candidate_segments_from_frame(segments, frame: pd.DataFrame) -> list[CandidateSegment]:
        """Convert segmenter output into a stable candidate-segment contract.

        Args:
            segments: Iterable of
                :class:`~iabm_incidents.segmentation.EpisodeSegment` objects
                with positional indices into ``frame``.
            frame: Window-score frame addressed with ``.loc``.

        Returns:
            One :class:`CandidateSegment` per segment, with timestamps resolved
            from ``start_time``/``end_time`` and positional window ids.
        """
        return [
            CandidateSegment(
                candidate_id=f"segment_{segment.start_index}_{segment.end_index}",
                start_index=segment.start_index,
                end_index=segment.end_index,
                event_start=pd.to_datetime(frame.loc[segment.start_index, "start_time"]),
                peak_time=pd.to_datetime(frame.loc[segment.peak_index, "start_time"]),
                event_end=pd.to_datetime(frame.loc[segment.end_index, "end_time"]),
                recovery_start=pd.to_datetime(frame.loc[segment.recovery_start_index, "start_time"]) if segment.recovery_start_index is not None else None,
                recovery_end=pd.to_datetime(frame.loc[segment.recovery_end_index, "end_time"]) if segment.recovery_end_index is not None else None,
                source_window_ids=tuple(str(index) for index in range(segment.start_index, segment.end_index + 1)),
            )
            for segment in segments
        ]

    @staticmethod
    def _empty_episode_frame() -> pd.DataFrame:
        """Return an empty episode dataframe with the canonical output schema.

        Returns:
            Empty dataframe whose columns mirror the fields of
            :class:`IncidentEpisode`.
        """
        return pd.DataFrame(
            columns=[
                "episode_id",
                "event_start",
                "event_end",
                "pre_event_start",
                "peak_time",
                "recovery_start",
                "recovery_end",
                "primary_family",
                "secondary_families",
                "family_confidence",
                "assignment_method",
                "evidence",
                "onset_score",
                "peak_score",
                "mean_score",
                "duration_seconds",
                "time_to_recovery_seconds",
                "recovery_status",
                "source_sequence_indices",
                "source_window_ids",
                "asset_id",
                "registry_incident_id",
                "label_strength",
            ]
        )
