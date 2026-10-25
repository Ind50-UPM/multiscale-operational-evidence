"""Recovery assessment primitives for Model_D episodes.

Role in the pipeline:
    After :class:`~iabm_incidents.segmentation.EpisodeSegmenter` has delimited
    an episode, :class:`~iabm_incidents.episodes.IncidentEpisodeBuilder` asks
    :class:`RecoveryAssessor` whether, and when, the deviation score returned
    to nominal levels after the last anomalous window. The outcome is stored as
    ``recovery_start``, ``recovery_end``, ``recovery_status`` and
    ``time_to_recovery_seconds`` in the episode table and in the CLI's
    ``recovery_assessment`` report.

Main public objects:
    * :class:`RecoveryStatus` - string enumeration of recovery outcomes.
    * :class:`RecoveryAssessment` - immutable result record.
    * :class:`RecoveryAssessor` - hysteresis rule with a consecutive-window
      confirmation requirement.

Data contract:
    The window frame must contain ``deviation_score``, ``start_time`` and
    ``end_time`` (timestamps) and must be indexed by a default
    ``RangeIndex`` (labels equal positions), see the maintenance notes.

Maintenance notes:
    * **Invariant:** ``assess`` mixes positional slicing
      (``iloc[last_event_index + 1:]``) with label lookups
      (``loc[last_event_index, ...]``). Both coincide only for a
      ``0..n-1`` index; callers must ``reset_index(drop=True)`` beforehand.
    * Only :attr:`RecoveryStatus.STABLE`, :attr:`RecoveryStatus.PARTIAL` and
      :attr:`RecoveryStatus.NOT_RECOVERED` are produced here.
      ``RECONFIGURED`` and ``UNKNOWN`` exist for registries and downstream
      consumers.
    * The recovery threshold should be lower than the onset threshold
      (hysteresis); this is not enforced here.
    * No temporal-gap or ``asset_id`` check is applied to the following
      windows: "consecutive" means consecutive rows. Pass a single-asset,
      chronologically sorted frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class RecoveryStatus(str, Enum):
    """Recovery outcome categories for a detected episode.

    The enumeration subclasses :class:`str`, so members compare equal to
    their values (``RecoveryStatus.STABLE == "stable"``) and serialise as
    plain strings.

    Attributes:
        NOT_RECOVERED: No post-episode window falls below the recovery
            threshold, or the episode ends the frame.
        PARTIAL: At least one window falls below the threshold, but never for
            the required number of consecutive windows.
        STABLE: The score stays below the threshold for the required number
            of consecutive windows.
        RECONFIGURED: The asset returned to operation in a different regime
            (reserved for annotated registries).
        UNKNOWN: Recovery information is unavailable.
    """

    NOT_RECOVERED = "not_recovered"
    PARTIAL = "partial"
    STABLE = "stable"
    RECONFIGURED = "reconfigured"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RecoveryAssessment:
    """Outcome of recovery confirmation for one episode.

    Attributes:
        recovery_start: ``start_time`` of the first window of the confirming
            run (stable) or of the first below-threshold window (partial);
            ``None`` when not recovered.
        recovery_end: ``end_time`` of the last window of the confirming run
            (stable) or of the same single window (partial); ``None`` when not
            recovered.
        recovery_status: Outcome category.
        time_to_recovery_seconds: Seconds between the ``end_time`` of the
            last event window and ``recovery_end``; ``None`` when not
            recovered.
    """

    recovery_start: pd.Timestamp | None
    recovery_end: pd.Timestamp | None
    recovery_status: RecoveryStatus
    time_to_recovery_seconds: float | None


class RecoveryAssessor:
    """Confirm recovery using threshold hysteresis and consecutive windows.

    The assessor is stateless; one instance can be shared across episodes.
    """

    def assess(
        self,
        window_frame: pd.DataFrame,
        *,
        last_event_index: int,
        recovery_threshold: float,
        recovery_windows: int,
    ) -> RecoveryAssessment:
        """Assess whether a stable recovery is observed after the event.

        Only windows strictly after ``last_event_index`` are inspected. A
        window is *below* when ``deviation_score < recovery_threshold``.

        Args:
            window_frame: Chronologically ordered window-score frame with a
                default ``RangeIndex`` and columns ``deviation_score``,
                ``start_time`` and ``end_time``.
            last_event_index: Index (label and position) of the last window
                belonging to the episode.
            recovery_threshold: Score level below which a window counts as
                recovered (same units as ``deviation_score``).
            recovery_windows: Number of consecutive below-threshold windows
                required for :attr:`RecoveryStatus.STABLE`; positive integer.

        Returns:
            RecoveryAssessment: ``STABLE`` for the first run of
            ``recovery_windows`` consecutive below-threshold windows,
            otherwise ``PARTIAL`` for the first isolated below-threshold
            window, otherwise ``NOT_RECOVERED`` with ``None`` fields.

        Notes:
            Time to recovery is measured to the *end* of the confirming run,
            so it includes the confirmation delay:

            .. math::

                T_{rec} = t^{end}_{k+R-1} - t^{end}_{e},

            where :math:`e` is the last event window, :math:`k` the first
            window of the confirming run and :math:`R` = ``recovery_windows``.
            Missing scores are compared as ``NaN`` and therefore never count
            as below the threshold.
        """

        following = window_frame.iloc[last_event_index + 1 :].copy()
        if following.empty:
            return RecoveryAssessment(None, None, RecoveryStatus.NOT_RECOVERED, None)

        below = following["deviation_score"].lt(recovery_threshold)
        # Note: the rolling sum equals R exactly at the last window of each
        # run of R consecutive below-threshold windows.
        sustained = (
            below.rolling(recovery_windows, min_periods=recovery_windows).sum().eq(recovery_windows)
            if len(following) >= recovery_windows
            else pd.Series(False, index=following.index)
        )
        if bool(sustained.any()):
            stable_index = sustained[sustained].index[0]
            stable_position = following.index.get_loc(stable_index)
            start_index = following.index[stable_position - recovery_windows + 1]
            stable_start = window_frame.loc[start_index]
            stable_end = window_frame.loc[stable_index]
            return RecoveryAssessment(
                recovery_start=stable_start["start_time"],
                recovery_end=stable_end["end_time"],
                recovery_status=RecoveryStatus.STABLE,
                time_to_recovery_seconds=(
                    stable_end["end_time"] - window_frame.loc[last_event_index, "end_time"]
                ).total_seconds(),
            )

        if bool(below.any()):
            first_index = below[below].index[0]
            first_row = window_frame.loc[first_index]
            return RecoveryAssessment(
                recovery_start=first_row["start_time"],
                recovery_end=first_row["end_time"],
                recovery_status=RecoveryStatus.PARTIAL,
                time_to_recovery_seconds=(
                    first_row["end_time"] - window_frame.loc[last_event_index, "end_time"]
                ).total_seconds(),
            )

        return RecoveryAssessment(None, None, RecoveryStatus.NOT_RECOVERED, None)
