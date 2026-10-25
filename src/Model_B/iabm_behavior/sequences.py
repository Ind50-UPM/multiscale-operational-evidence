"""Sequence-analysis utilities for Model_B behavioral modeling.

This module contains the typed data structures and core logic that translate a
state timeline into progressively richer behavioral abstractions: contiguous
runs, active sequences, nominal references, anomaly comparisons, longitudinal
metrics, recovery metrics, and family-signature summaries.

Role in the pipeline:
    Model_A (or digital labelling) produces a per-sample state timeline whose
    states are non-negative integers (bitmasks of active components, ``0``
    meaning "nothing running"). Model_B condenses that timeline into
    *state words*: ordered tuples of consecutive non-zero run states delimited
    by idle (``0``) samples. Model_C later decodes those words into semantic
    modes, and Model_D aggregates the evidence into incident episodes.

Abstraction levels:

1. **Sample timeline** -- :class:`pandas.DataFrame` with a
   :class:`pandas.DatetimeIndex` and an integer state column.
2. :class:`StateRun` -- maximal block of consecutive samples sharing a state.
3. :class:`ActiveSequence` -- maximal block of consecutive runs with
   ``state > 0``; its ``states`` tuple is the *word*.
4. :class:`NominalSequenceReference` -- word frequency/duration statistics
   derived from a nominal (healthy) timeline.
5. :class:`SequenceComparison`, :class:`LongitudinalSequenceMetrics`,
   :class:`RecoveryMetrics` -- per-sequence diagnostics against the nominal
   reference.

Data contracts:

* Input timeline: either a ``Time`` column (parsed with
  :func:`pandas.to_datetime`) or a ``DatetimeIndex``; a state column named
  ``state_column`` (default ``Predicted_State``) or the legacy name
  ``estado``. State values must be castable to ``int`` (no ``NaN``).
  Timestamps may be naive or timezone-aware but must be consistent.
* Time convention: a run/sequence ``start_time`` and ``end_time`` are the
  timestamps of its first and *last* sample (inclusive). Durations are
  therefore ``end_time - start_time`` in seconds, which is ``0.0`` for a
  single-sample run; the sampling interval after the last sample is not
  counted.
* All durations are expressed in seconds (``float``).

.. mermaid::

   classDiagram
       class BehavioralSequenceAnalyzer {
           +str state_column
           +load_state_timeline(file_path) DataFrame
           +smooth_short_runs(timeline, min_duration_seconds, min_samples) DataFrame
           +extract_runs(timeline) list~StateRun~
           +extract_active_sequences(timeline) list~ActiveSequence~
           +summarize_sequence_words(sequences) DataFrame
           +build_nominal_reference(sequences) list~NominalSequenceReference~
           +compare_to_nominal(observed, nominal, anomaly_threshold) DataFrame
           +compute_longitudinal_metrics(observed, nominal, anomaly_threshold, rolling_window) DataFrame
           +compute_recovery_metrics(observed, nominal, anomaly_threshold) DataFrame
           +summarize_family_signatures(longitudinal_metrics) DataFrame
       }
       class StateRun {
           +int state
           +Timestamp start_time
           +Timestamp end_time
           +int sample_count
           +float duration_seconds
       }
       class ActiveSequence {
           +Timestamp start_time
           +Timestamp end_time
           +tuple states
           +float total_duration_seconds
           +int run_count
       }
       class NominalSequenceReference {
           +tuple states
           +int count
           +float avg_duration_seconds
       }
       class SequenceComparison
       class LongitudinalSequenceMetrics
       class RecoveryMetrics
       BehavioralSequenceAnalyzer ..> StateRun : creates
       BehavioralSequenceAnalyzer ..> ActiveSequence : creates
       BehavioralSequenceAnalyzer ..> NominalSequenceReference : creates
       BehavioralSequenceAnalyzer ..> SequenceComparison : creates
       BehavioralSequenceAnalyzer ..> LongitudinalSequenceMetrics : creates
       BehavioralSequenceAnalyzer ..> RecoveryMetrics : creates
       ActiveSequence o-- StateRun : aggregates

Maintenance notes:
    All dataclasses are frozen and the analyzer holds no state besides
    ``state_column``, so every method is a pure function of its inputs (apart
    from file I/O in :meth:`BehavioralSequenceAnalyzer.load_state_timeline`).
    The dynamic-programming distances are implemented in pure Python with
    :math:`O(n \\cdot m)` cost per word pair; words are short in practice, but
    :meth:`~BehavioralSequenceAnalyzer.compute_longitudinal_metrics`
    re-evaluates comparisons inside a rolling window, so its cost grows with
    ``len(sequences) * rolling_window * len(references)``. Column names and
    order of the returned dataframes follow the dataclass field order and are
    consumed by downstream layers; do not rename fields casually. The family
    labels produced here (``nominal_cycle``, ``recurrent_spurious_cycle``,
    ``process_saturation``, ``pump_abrupt_failure``) are independent from the
    Model_C incident-family heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class StateRun:
    """Contiguous temporal segment where the decoded state remains constant.

    Attributes:
        state: Integer state identifier observed throughout the run (``0``
            denotes the idle state).
        start_time: Timestamp of the first sample in the run.
        end_time: Timestamp of the last sample in the run (inclusive).
        sample_count: Number of samples belonging to the run (``>= 1``).
        duration_seconds: Duration covered by the run in seconds, computed as
            ``end_time - start_time`` and clipped at ``0.0``; a single-sample
            run therefore has duration ``0.0``.
    """

    state: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    sample_count: int
    duration_seconds: float


@dataclass(frozen=True)
class ActiveSequence:
    """Higher-level sequence formed by consecutive non-zero runs.

    An active sequence is delimited by idle (``state == 0``) runs or by the
    timeline boundaries. Because consecutive runs always have different
    states, adjacent entries of :attr:`states` are never equal.

    Attributes:
        start_time: Timestamp of the first sample of the first run in the
            sequence.
        end_time: Timestamp of the last sample of the last run in the sequence.
        states: Ordered tuple of non-zero states composing the sequence (the
            sequence *word*), one entry per run.
        total_duration_seconds: Sum of the member runs' ``duration_seconds``.
            This is not ``end_time - start_time``: the gaps between the last
            sample of a run and the first sample of the next run are excluded.
        run_count: Number of runs participating in the sequence
            (``== len(states)``).
    """

    start_time: pd.Timestamp
    end_time: pd.Timestamp
    states: tuple[int, ...]
    total_duration_seconds: float
    run_count: int


@dataclass(frozen=True)
class NominalSequenceReference:
    """Condensed representation of a recurrent nominal sequence word.

    Attributes:
        states: Canonical state word used as nominal reference.
        count: Number of nominal active sequences exhibiting this word.
        avg_duration_seconds: Mean ``total_duration_seconds`` of those
            sequences, in seconds.
    """

    states: tuple[int, ...]
    count: int
    avg_duration_seconds: float


@dataclass(frozen=True)
class SequenceComparison:
    """Anomaly-oriented comparison between one observed and one nominal word.

    Attributes:
        observed_states: Observed state word.
        nominal_states: Best-matching nominal word, or ``()`` when no nominal
            reference was available.
        exact_match: Whether both words are identical.
        state_distance: Levenshtein edit distance between the two words
            (unit-cost insertion, deletion and substitution).
        dtw_distance: Dynamic-time-warping mismatch score with a 0/1 local
            cost; this is the structural term of :attr:`anomaly_score`.
        duration_ratio_delta: Signed relative duration deviation from the
            nominal average, :math:`(d_{obs} - d_{nom}) / d_{nom}`.
        anomaly_score: Combined score
            :math:`\\mathrm{dtw} + |\\Delta_{dur}|` used for thresholding.
        is_anomalous: ``True`` when ``anomaly_score >= anomaly_threshold``.
    """

    observed_states: tuple[int, ...]
    nominal_states: tuple[int, ...]
    exact_match: bool
    state_distance: int
    dtw_distance: float
    duration_ratio_delta: float
    anomaly_score: float
    is_anomalous: bool


@dataclass(frozen=True)
class LongitudinalSequenceMetrics:
    """Rolling metrics describing the temporal behavior of one sequence event.

    Attributes:
        sequence_index: Zero-based position of the sequence in the observed
            (chronologically ordered) sequence list.
        states: State word of the sequence.
        recurrence_interval_seconds: Seconds between this sequence's
            ``start_time`` and that of the most recent earlier sequence with
            the same word; ``None`` for the first occurrence.
        sequence_persistence_ratio: Mean run duration within the sequence,
            ``total_duration_seconds / run_count`` (seconds per run, despite
            the "ratio" name).
        over_activation_rate: Number of active (``> 0``) entries in the word
            divided by the number of distinct states; ``1.0`` means no state
            is revisited, larger values indicate oscillation between states.
        transition_instability: Edit distance to the immediately previous
            sequence's word normalised by the longer word length, in
            :math:`[0, 1]`; ``0.0`` for the first sequence.
        duration_drift: Signed relative duration deviation from the
            best-matching nominal word (same as
            :attr:`SequenceComparison.duration_ratio_delta`).
        rolling_divergence: Mean anomaly score over the trailing window of
            sequences ending at (and including) this one.
        nominal_match: Best-matching nominal word (``()`` if none).
        anomaly_flag: Whether the sequence itself is anomalous.
    """

    sequence_index: int
    states: tuple[int, ...]
    recurrence_interval_seconds: float | None
    sequence_persistence_ratio: float
    over_activation_rate: float
    transition_instability: float
    duration_drift: float
    rolling_divergence: float
    nominal_match: tuple[int, ...]
    anomaly_flag: bool


@dataclass(frozen=True)
class RecoveryMetrics:
    """Recovery-oriented summary anchored on one anomalous sequence.

    Recovery is searched among *later* sequences whose best-matching nominal
    word equals that of the anomalous sequence.

    Attributes:
        anomaly_sequence_index: Index of the anomalous sequence.
        recovered_sequence_index: Index of the first later, non-anomalous
            sequence sharing the same nominal match; ``None`` if none exists.
        time_to_recovery_seconds: Seconds between the anomalous sequence's
            ``end_time`` and the recovering sequence's ``end_time``; ``None``
            when no recovery was found.
        partial_recovery_score: :math:`\\max(0, 1 - \\min(s_f, s_a)/s_a)` where
            :math:`s_a` is the anomalous score and :math:`s_f` the score of the
            last examined same-nominal future sequence (the recovering one if
            found); ``0.0`` when no such future sequence exists.
        post_intervention_regime_shift: ``True`` when the recovering sequence's
            observed word differs from the anomalous one's (recovery through a
            different behaviour); ``False`` otherwise or without recovery.
    """

    anomaly_sequence_index: int
    recovered_sequence_index: int | None
    time_to_recovery_seconds: float | None
    partial_recovery_score: float
    post_intervention_regime_shift: bool


class BehavioralSequenceAnalyzer:
    """Load state timelines and derive run-, sequence-, and longitudinal-level features.

    Args:
        state_column: Name of the state column used when reading input tables
            and when extracting runs.

    Attributes:
        state_column: Name of the integer state column.

    Examples:
        >>> import pandas as pd
        >>> analyzer = BehavioralSequenceAnalyzer()
        >>> tl = pd.DataFrame(
        ...     {"Predicted_State": [0, 1, 1, 2, 0, 4]},
        ...     index=pd.date_range("2022-01-01", periods=6, freq="1s"),
        ... )
        >>> [s.states for s in analyzer.extract_active_sequences(tl)]
        [(1, 2), (4,)]

    Notes:
        The analyzer is intentionally deterministic and side-effect free so it
        can be used both by small CLIs and by the larger month-wise batch
        utilities introduced for historical campaign processing.
    """

    def __init__(self, state_column: str = "Predicted_State") -> None:
        """Initialise the analyzer.

        Args:
            state_column: Name of the state column (default
                ``"Predicted_State"``).
        """
        self.state_column = state_column

    def load_state_timeline(self, file_path: str | Path) -> pd.DataFrame:
        """Load a state timeline from Parquet, Excel, or CSV.

        The reader is selected from the (case-insensitive) file suffix:
        ``.parquet``, ``.xlsx``/``.xls`` or ``.csv``. If the configured state
        column is absent but a column named ``estado`` exists, it is renamed.
        A ``Time`` column, when present, is parsed and promoted to the index;
        otherwise the existing index must already be a ``DatetimeIndex``.

        Args:
            file_path: Path to the timeline source.

        Returns:
            Time-indexed dataframe sorted chronologically, containing all
            original columns (except ``Time``, which becomes the index).

        Raises:
            ValueError: If the file format is unsupported or no usable temporal
                axis and state column can be resolved.

        Notes:
            The helper supports the repository's real datasets where the state
            column may be named either ``Predicted_State`` or ``estado``. Note
            that CSV/Excel round-trips do not preserve a ``DatetimeIndex``, so
            such files need an explicit ``Time`` column.
        """
        path = Path(file_path)
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

        # Note: legacy Spanish column name from the original labelling tools.
        if self.state_column not in df.columns and "estado" in df.columns:
            df = df.rename(columns={"estado": self.state_column})

        if self.state_column not in df.columns:
            raise ValueError(f"State column '{self.state_column}' not found in {path}")

        if "Time" in df.columns:
            df["Time"] = pd.to_datetime(df["Time"])
            df = df.set_index("Time")
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("A datetime index or a 'Time' column is required.")

        return df.sort_index()

    def smooth_short_runs(
        self,
        timeline: pd.DataFrame,
        *,
        min_duration_seconds: float = 1.0,
        min_samples: int = 1,
    ) -> pd.DataFrame:
        """Merge short transient runs into the following active state.

        A run is relabelled with the state of the next run when all of the
        following hold: the run and the next run are both active
        (``state > 0``), their states differ, ``duration_seconds <
        min_duration_seconds`` (strict) and ``sample_count <= min_samples``.
        Idle runs and the final run are never modified.

        Args:
            timeline: Time-indexed state timeline (as returned by
                :meth:`load_state_timeline`).
            min_duration_seconds: Exclusive upper bound, in seconds, on the
                duration of a run considered transient.
            min_samples: Inclusive upper bound on the number of samples of a
                run considered transient.

        Returns:
            Smoothed copy of the input timeline; the input is not mutated.
            Timelines with fewer than two runs are returned as an unchanged
            copy.

        Notes:
            The heuristic is intentionally asymmetric: transient active runs are
            merged into the following active run, not into the previous one, so
            later sequence extraction remains compatible with the operational
            progression of the signal.

            The pass is single-shot: eligibility is evaluated on the runs of
            the *original* timeline, so a chain of consecutive transients is
            each relabelled to its own original successor rather than being
            collapsed iteratively.
        """
        smoothed = timeline.copy()
        runs = self.extract_runs(smoothed)
        if len(runs) < 2:
            return smoothed

        values = smoothed[self.state_column].to_numpy(copy=True)
        # Invariant: boundaries are computed on the same sample order as
        # ``runs`` so ``run_boundaries[i]`` is the [start, stop) slice of runs[i].
        run_boundaries = self._compute_run_boundaries(values)

        for index, run in enumerate(runs[:-1]):
            if (
                run.state > 0
                and runs[index + 1].state > 0
                and run.state != runs[index + 1].state
                and run.duration_seconds < min_duration_seconds
                and run.sample_count <= min_samples
            ):
                start_idx, stop_idx = run_boundaries[index]
                values[start_idx:stop_idx] = runs[index + 1].state

        smoothed[self.state_column] = values
        return smoothed

    def extract_runs(self, timeline: pd.DataFrame) -> list[StateRun]:
        """Extract contiguous runs of the same state from a timeline.

        Args:
            timeline: Time-indexed dataframe containing :attr:`state_column`.
                The index must be datetime-like and already sorted.

        Returns:
            Runs in chronological order, covering every sample exactly once.
            Empty list for an empty timeline.

        Raises:
            KeyError: If :attr:`state_column` is not a column of ``timeline``.
            ValueError: If the state column cannot be cast to ``int`` (e.g. it
                contains ``NaN``).
        """
        states = timeline[self.state_column].astype(int).to_numpy()
        boundaries = self._compute_run_boundaries(states)
        runs: list[StateRun] = []
        for start_idx, stop_idx in boundaries:
            start_time = timeline.index[start_idx]
            # Note: ``stop_idx`` is exclusive, so the last sample is stop_idx - 1.
            end_time = timeline.index[stop_idx - 1]
            duration_seconds = max((end_time - start_time).total_seconds(), 0.0)
            runs.append(
                StateRun(
                    state=int(states[start_idx]),
                    start_time=start_time,
                    end_time=end_time,
                    sample_count=stop_idx - start_idx,
                    duration_seconds=duration_seconds,
                )
            )
        return runs

    def extract_active_sequences(self, timeline: pd.DataFrame) -> list[ActiveSequence]:
        """Group consecutive non-zero runs into higher-level active sequences.

        Every idle run (``state <= 0``) closes the sequence being accumulated;
        a trailing sequence that reaches the end of the timeline is also
        emitted.

        Args:
            timeline: Time-indexed dataframe containing :attr:`state_column`.

        Returns:
            Active sequences in chronological order.

        Raises:
            KeyError: If :attr:`state_column` is missing.
            ValueError: If the state column cannot be cast to ``int``.

        See Also:
            :meth:`extract_runs`
        """
        runs = self.extract_runs(timeline)
        sequences: list[ActiveSequence] = []
        current_runs: list[StateRun] = []

        for run in runs:
            if run.state > 0:
                current_runs.append(run)
                continue
            if current_runs:
                sequences.append(self._build_active_sequence(current_runs))
                current_runs = []

        # Note: flush a sequence that is still open at the end of the timeline.
        if current_runs:
            sequences.append(self._build_active_sequence(current_runs))

        return sequences

    def summarize_sequence_words(self, sequences: Iterable[ActiveSequence]) -> pd.DataFrame:
        """Aggregate repeated sequence words and their average duration.

        Args:
            sequences: Active sequences to summarise (consumed once).

        Returns:
            Dataframe with columns ``word`` (tuple of ``int``), ``count``
            (``int``, number of occurrences) and ``avg_duration_seconds``
            (``float``, mean ``total_duration_seconds``), sorted by ``count``
            then ``avg_duration_seconds``, both descending. The index is not
            reset after sorting. An empty dataframe with those three columns
            is returned when ``sequences`` is empty.
        """
        rows = [
            {
                "word": sequence.states,
                "count": 1,
                "duration_seconds": sequence.total_duration_seconds,
            }
            for sequence in sequences
        ]
        if not rows:
            return pd.DataFrame(columns=["word", "count", "avg_duration_seconds"])

        frame = pd.DataFrame(rows)
        # Note: tuples are hashable, so they can be used directly as group keys.
        return (
            frame.groupby("word", dropna=False)
            .agg(count=("count", "sum"), avg_duration_seconds=("duration_seconds", "mean"))
            .reset_index()
            .sort_values(["count", "avg_duration_seconds"], ascending=[False, False])
        )

    def build_nominal_reference(self, sequences: Iterable[ActiveSequence]) -> list[NominalSequenceReference]:
        """Convert observed sequences into a condensed nominal reference list.

        Args:
            sequences: Active sequences extracted from a nominal (healthy)
                timeline.

        Returns:
            One :class:`NominalSequenceReference` per distinct word, in the
            order produced by :meth:`summarize_sequence_words` (most frequent
            first). Empty list when ``sequences`` is empty.
        """
        summary = self.summarize_sequence_words(sequences)
        # Invariant: column order of ``summary`` is (word, count, avg_duration_seconds).
        return [
            NominalSequenceReference(
                states=tuple(word),
                count=int(count),
                avg_duration_seconds=float(avg_duration_seconds),
            )
            for word, count, avg_duration_seconds in summary.itertuples(index=False)
        ]

    def compare_to_nominal(
        self,
        observed_sequences: Iterable[ActiveSequence],
        nominal_references: Iterable[NominalSequenceReference],
        *,
        anomaly_threshold: float = 1.0,
    ) -> pd.DataFrame:
        """Compare observed sequence words against a nominal reference set.

        Args:
            observed_sequences: Observed active sequences.
            nominal_references: Nominal references (materialised into a list,
                so a generator is acceptable).
            anomaly_threshold: Score at or above which a sequence is flagged
                anomalous.

        Returns:
            One row per observed sequence, in input order, with the fields of
            :class:`SequenceComparison` as columns. A dataframe with no rows
            and no columns is returned when there are no observed sequences.

        See Also:
            :meth:`_compare_single_sequence`
        """
        references = list(nominal_references)
        return pd.DataFrame(
            [
                self._compare_single_sequence(
                    sequence,
                    references,
                    anomaly_threshold=anomaly_threshold,
                ).__dict__
                for sequence in observed_sequences
            ]
        )

    def compute_longitudinal_metrics(
        self,
        observed_sequences: Iterable[ActiveSequence],
        nominal_references: Iterable[NominalSequenceReference],
        *,
        anomaly_threshold: float = 1.0,
        rolling_window: int = 3,
    ) -> pd.DataFrame:
        """Compute temporal recurrence and divergence metrics for each sequence.

        Args:
            observed_sequences: Observed active sequences in chronological
                order.
            nominal_references: Nominal references used for matching.
            anomaly_threshold: Score at or above which a sequence is flagged
                anomalous.
            rolling_window: Number of trailing sequences (including the
                current one) averaged by ``rolling_divergence``; should be
                ``>= 1``.

        Returns:
            One row per sequence with the fields of
            :class:`LongitudinalSequenceMetrics` as columns. A dataframe with
            no rows and no columns is returned for empty input.

        Notes:
            Only past information is used for each row (previous sequences
            and the trailing window), so the metrics are causal with respect
            to the observed sequence order; the nominal reference, however, is
            taken as given.
        """
        sequences = list(observed_sequences)
        references = list(nominal_references)
        rows: list[dict[str, object]] = []

        for index, sequence in enumerate(sequences):
            comparison = self._compare_single_sequence(
                sequence,
                references,
                anomaly_threshold=anomaly_threshold,
            )
            previous_same = self._find_previous_same_sequence(sequences, index)
            previous_sequence = sequences[index - 1] if index > 0 else None
            rows.append(
                LongitudinalSequenceMetrics(
                    sequence_index=index,
                    states=sequence.states,
                    recurrence_interval_seconds=(
                        None
                        if previous_same is None
                        else (sequence.start_time - previous_same.start_time).total_seconds()
                    ),
                    sequence_persistence_ratio=self._sequence_persistence_ratio(sequence),
                    over_activation_rate=self._over_activation_rate(sequence),
                    transition_instability=self._transition_instability(sequence, previous_sequence),
                    duration_drift=comparison.duration_ratio_delta,
                    # Note: recomputes comparisons for the window (no caching).
                    rolling_divergence=self._rolling_divergence(
                        sequences,
                        references,
                        index=index,
                        anomaly_threshold=anomaly_threshold,
                        window=rolling_window,
                    ),
                    nominal_match=comparison.nominal_states,
                    anomaly_flag=comparison.is_anomalous,
                ).__dict__
            )
        return pd.DataFrame(rows)

    def compute_recovery_metrics(
        self,
        observed_sequences: Iterable[ActiveSequence],
        nominal_references: Iterable[NominalSequenceReference],
        *,
        anomaly_threshold: float = 1.0,
    ) -> pd.DataFrame:
        """Estimate recovery behavior after anomalous sequences.

        For each anomalous sequence ``i``, later sequences ``j > i`` whose best
        nominal match equals that of ``i`` are scanned in order. Each such
        ``j`` updates the partial recovery score; the scan stops at the first
        non-anomalous ``j``, which is recorded as the recovery point.

        Args:
            observed_sequences: Observed active sequences in chronological
                order.
            nominal_references: Nominal references used for matching.
            anomaly_threshold: Score at or above which a sequence is flagged
                anomalous.

        Returns:
            One row per anomalous sequence with the fields of
            :class:`RecoveryMetrics` as columns. A dataframe with no rows and
            no columns is returned when no sequence is anomalous.
        """
        sequences = list(observed_sequences)
        references = list(nominal_references)
        comparisons = [
            self._compare_single_sequence(sequence, references, anomaly_threshold=anomaly_threshold)
            for sequence in sequences
        ]
        rows: list[dict[str, object]] = []

        for index, comparison in enumerate(comparisons):
            if not comparison.is_anomalous:
                continue

            recovered_index = None
            recovery_time = None
            partial_recovery_score = 0.0
            regime_shift = False

            for future_index in range(index + 1, len(comparisons)):
                future = comparisons[future_index]
                if future.nominal_states != comparison.nominal_states:
                    continue
                # Note: the score is overwritten (not maximised) on every
                # same-nominal future; 1e-9 guards division by a zero score
                # when anomaly_threshold <= 0.
                partial_recovery_score = max(
                    0.0,
                    1.0 - min(future.anomaly_score, comparison.anomaly_score) / max(comparison.anomaly_score, 1e-9),
                )
                if not future.is_anomalous:
                    recovered_index = future_index
                    # Note: measured end-to-end (anomaly end -> recovery end).
                    recovery_time = (
                        sequences[future_index].end_time - sequences[index].end_time
                    ).total_seconds()
                    regime_shift = future.observed_states != comparison.observed_states
                    break

            rows.append(
                RecoveryMetrics(
                    anomaly_sequence_index=index,
                    recovered_sequence_index=recovered_index,
                    time_to_recovery_seconds=recovery_time,
                    partial_recovery_score=partial_recovery_score,
                    post_intervention_regime_shift=regime_shift,
                ).__dict__
            )
        return pd.DataFrame(rows)

    def summarize_family_signatures(self, longitudinal_metrics: pd.DataFrame) -> pd.DataFrame:
        """Summarize sequence signatures into coarse incident-family candidates.

        Each longitudinal row is labelled with
        :meth:`_infer_family_candidate` and rows are then grouped by
        ``(incident_family_candidate, states)``.

        Args:
            longitudinal_metrics: Output of
                :meth:`compute_longitudinal_metrics`; must contain
                ``states``, ``transition_instability``, ``duration_drift`` and
                ``anomaly_flag``.

        Returns:
            Dataframe with columns ``incident_family_candidate``,
            ``signature_word``, ``count``, ``mean_transition_instability`` and
            ``mean_duration_drift``, sorted by ``count`` then
            ``mean_transition_instability`` (both descending). An empty
            dataframe with those columns is returned for empty input.

        Raises:
            KeyError: If a required column is missing from a non-empty input.
        """
        if longitudinal_metrics.empty:
            return pd.DataFrame(
                columns=[
                    "incident_family_candidate",
                    "signature_word",
                    "count",
                    "mean_transition_instability",
                    "mean_duration_drift",
                ]
            )

        metrics = longitudinal_metrics.copy()
        metrics["incident_family_candidate"] = metrics.apply(
            lambda row: self._infer_family_candidate(
                row["transition_instability"],
                row["duration_drift"],
                bool(row["anomaly_flag"]),
            ),
            axis=1,
        )
        return (
            metrics.groupby(["incident_family_candidate", "states"], dropna=False)
            .agg(
                count=("states", "size"),
                mean_transition_instability=("transition_instability", "mean"),
                mean_duration_drift=("duration_drift", "mean"),
            )
            .reset_index()
            .rename(columns={"states": "signature_word"})
            .sort_values(["count", "mean_transition_instability"], ascending=[False, False])
        )

    @staticmethod
    def _compute_run_boundaries(states: pd.Series | list[int] | pd.Index | pd.array) -> list[tuple[int, int]]:
        """Compute contiguous-state boundaries from a one-dimensional state series.

        Args:
            states: Any one-dimensional iterable of comparable state values
                (a NumPy array is also accepted in practice). Values are
                compared with ``!=`` without casting.

        Returns:
            List of half-open positional intervals ``(start, stop)`` such that
            ``states[start:stop]`` is a maximal block of equal values. The
            intervals are contiguous and cover ``range(len(states))``. Empty
            list for empty input.

        Examples:
            >>> BehavioralSequenceAnalyzer._compute_run_boundaries([0, 1, 1, 2])
            [(0, 1), (1, 3), (3, 4)]
        """
        values = list(states)
        if not values:
            return []

        boundaries: list[tuple[int, int]] = []
        start = 0
        for index in range(1, len(values)):
            if values[index] != values[index - 1]:
                boundaries.append((start, index))
                start = index
        # Invariant: the final run always extends to the end of the series.
        boundaries.append((start, len(values)))
        return boundaries

    @staticmethod
    def _build_active_sequence(runs: list[StateRun]) -> ActiveSequence:
        """Assemble one active sequence from a list of non-zero runs.

        Args:
            runs: Non-empty, chronologically ordered list of consecutive
                active runs.

        Returns:
            The corresponding :class:`ActiveSequence`.

        Raises:
            IndexError: If ``runs`` is empty.
        """
        return ActiveSequence(
            start_time=runs[0].start_time,
            end_time=runs[-1].end_time,
            states=tuple(run.state for run in runs),
            total_duration_seconds=sum(run.duration_seconds for run in runs),
            run_count=len(runs),
        )

    def _compare_single_sequence(
        self,
        sequence: ActiveSequence,
        references: list[NominalSequenceReference],
        *,
        anomaly_threshold: float,
    ) -> SequenceComparison:
        """Compare one observed sequence against the best nominal reference.

        The best reference minimises, lexicographically: the edit distance
        between words, then the absolute relative duration deviation, then
        ``-count`` (i.e. prefers the most frequent nominal word). Ties beyond
        that keep the first reference in list order.

        Args:
            sequence: Observed active sequence.
            references: Nominal references (may be empty).
            anomaly_threshold: Score at or above which the sequence is flagged
                anomalous.

        Returns:
            The comparison record. With an empty reference list a fallback
            record is returned: ``nominal_states=()``,
            ``state_distance = dtw_distance = len(word)``,
            ``duration_ratio_delta = 1.0`` and
            ``anomaly_score = len(word) + 1``.

        Notes:
            Matching uses the edit distance, whereas the score uses the DTW
            distance: :math:`\\text{score} = \\mathrm{DTW}(w_{obs}, w_{nom}) +
            |\\Delta_{dur}|`.
        """
        if not references:
            anomaly_score = float(len(sequence.states) + 1.0)
            return SequenceComparison(
                observed_states=sequence.states,
                nominal_states=(),
                exact_match=False,
                state_distance=len(sequence.states),
                dtw_distance=float(len(sequence.states)),
                duration_ratio_delta=1.0,
                anomaly_score=anomaly_score,
                is_anomalous=anomaly_score >= anomaly_threshold,
            )

        best_reference = min(
            references,
            key=lambda reference: (
                self._sequence_edit_distance(sequence.states, reference.states),
                abs(self._duration_ratio_delta(sequence.total_duration_seconds, reference.avg_duration_seconds)),
                -reference.count,
            ),
        )
        state_distance = self._sequence_edit_distance(sequence.states, best_reference.states)
        dtw_distance = self._sequence_dtw_distance(sequence.states, best_reference.states)
        duration_ratio_delta = self._duration_ratio_delta(
            sequence.total_duration_seconds,
            best_reference.avg_duration_seconds,
        )
        anomaly_score = float(dtw_distance + abs(duration_ratio_delta))

        return SequenceComparison(
            observed_states=sequence.states,
            nominal_states=best_reference.states,
            exact_match=sequence.states == best_reference.states,
            state_distance=state_distance,
            dtw_distance=dtw_distance,
            duration_ratio_delta=duration_ratio_delta,
            anomaly_score=anomaly_score,
            is_anomalous=anomaly_score >= anomaly_threshold,
        )

    @staticmethod
    def _sequence_edit_distance(observed: tuple[int, ...], nominal: tuple[int, ...]) -> int:
        """Compute a Levenshtein-like edit distance between two state words.

        Standard Wagner-Fischer dynamic programme with unit costs for
        insertion, deletion and substitution.

        Args:
            observed: First state word.
            nominal: Second state word.

        Returns:
            Minimum number of edits turning ``observed`` into ``nominal``
            (``>= 0``; ``0`` iff the words are equal).

        Examples:
            >>> BehavioralSequenceAnalyzer._sequence_edit_distance((1, 2), (1, 2, 4))
            1
        """
        rows = len(observed) + 1
        cols = len(nominal) + 1
        distance = [[0] * cols for _ in range(rows)]

        # Note: first row/column = cost of building from / reducing to the empty word.
        for row in range(rows):
            distance[row][0] = row
        for col in range(cols):
            distance[0][col] = col

        for row in range(1, rows):
            for col in range(1, cols):
                substitution_cost = 0 if observed[row - 1] == nominal[col - 1] else 1
                # Recurrence: deletion, insertion, substitution/match.
                distance[row][col] = min(
                    distance[row - 1][col] + 1,
                    distance[row][col - 1] + 1,
                    distance[row - 1][col - 1] + substitution_cost,
                )
        return distance[-1][-1]

    @staticmethod
    def _duration_ratio_delta(observed_seconds: float, nominal_seconds: float) -> float:
        """Measure relative duration drift between observed and nominal sequence durations.

        Args:
            observed_seconds: Observed duration in seconds.
            nominal_seconds: Nominal (reference) duration in seconds.

        Returns:
            Signed relative deviation
            :math:`(d_{obs} - d_{nom}) / d_{nom}` (``-1.0`` means zero
            observed duration, positive means longer than nominal). When
            ``nominal_seconds <= 0`` the ratio is undefined and the function
            returns ``0.0`` if ``observed_seconds <= 0`` and ``1.0`` otherwise.
        """
        # Why: avoid division by zero for degenerate (single-sample) nominal words.
        if nominal_seconds <= 0:
            return 0.0 if observed_seconds <= 0 else 1.0
        return float((observed_seconds - nominal_seconds) / nominal_seconds)

    @staticmethod
    def _sequence_dtw_distance(observed: tuple[int, ...], nominal: tuple[int, ...]) -> float:
        """Compute a simple DTW-inspired mismatch score between state words.

        Classic dynamic time warping with local cost ``0`` for equal states
        and ``1`` otherwise, and steps (1, 0), (0, 1) and (1, 1). Unlike the
        edit distance, repeated alignment of one element to several others is
        free when the states match, so words differing only in the number of
        repetitions of a state score ``0``.

        Args:
            observed: First state word.
            nominal: Second state word.

        Returns:
            Accumulated DTW cost as ``float``. Both words empty yields
            ``0.0``; exactly one empty yields the length of the other.
        """
        if not observed and not nominal:
            return 0.0
        if not observed or not nominal:
            return float(max(len(observed), len(nominal)))

        rows = len(observed)
        cols = len(nominal)
        # Note: (rows+1) x (cols+1) table padded with +inf so the first
        # row/column cannot be entered except through dtw[0][0].
        dtw = [[float("inf")] * (cols + 1) for _ in range(rows + 1)]
        dtw[0][0] = 0.0

        for row in range(1, rows + 1):
            for col in range(1, cols + 1):
                cost = 0.0 if observed[row - 1] == nominal[col - 1] else 1.0
                dtw[row][col] = cost + min(
                    dtw[row - 1][col],
                    dtw[row][col - 1],
                    dtw[row - 1][col - 1],
                )
        return float(dtw[rows][cols])

    @staticmethod
    def _sequence_persistence_ratio(sequence: ActiveSequence) -> float:
        """Estimate persistence as duration per contributing run.

        Args:
            sequence: Active sequence.

        Returns:
            ``total_duration_seconds / run_count`` in seconds per run, or
            ``0.0`` when ``run_count <= 0``.
        """
        if sequence.run_count <= 0:
            return 0.0
        return float(sequence.total_duration_seconds / max(sequence.run_count, 1))

    @staticmethod
    def _over_activation_rate(sequence: ActiveSequence) -> float:
        """Estimate how densely a sequence reuses active states.

        Args:
            sequence: Active sequence.

        Returns:
            Number of entries ``> 0`` in the word divided by the number of
            distinct states, or ``0.0`` for an empty word. For words produced
            by :meth:`extract_active_sequences` every entry is active, so the
            value equals ``run_count / n_distinct_states`` (``>= 1.0``).
        """
        if not sequence.states:
            return 0.0
        active_runs = sum(1 for state in sequence.states if state > 0)
        unique_states = len(set(sequence.states))
        return float(active_runs / max(unique_states, 1))

    def _transition_instability(
        self,
        sequence: ActiveSequence,
        previous_sequence: ActiveSequence | None,
    ) -> float:
        """Quantify state-word change against the immediately previous sequence.

        Args:
            sequence: Current active sequence.
            previous_sequence: The sequence immediately before it, or ``None``
                for the first sequence.

        Returns:
            Edit distance between the two words divided by the longer word
            length (at least ``1``), a value in :math:`[0, 1]`; ``0.0`` when
            ``previous_sequence`` is ``None``.
        """
        if previous_sequence is None:
            return 0.0
        distance = self._sequence_edit_distance(sequence.states, previous_sequence.states)
        scale = max(len(sequence.states), len(previous_sequence.states), 1)
        return float(distance / scale)

    def _rolling_divergence(
        self,
        sequences: list[ActiveSequence],
        references: list[NominalSequenceReference],
        *,
        index: int,
        anomaly_threshold: float,
        window: int,
    ) -> float:
        """Average recent anomaly scores over a rolling sequence window.

        Args:
            sequences: Full chronologically ordered sequence list.
            references: Nominal references.
            index: Position of the current sequence.
            anomaly_threshold: Passed through to the comparison (does not
                affect the score itself).
            window: Window length; the window is
                ``sequences[max(0, index - window + 1) : index + 1]``.

        Returns:
            Mean anomaly score over the window, or ``0.0`` if the window is
            empty (which happens only when ``window <= 0``).
        """
        # Note: trailing (causal) window, truncated at the start of the series.
        start = max(0, index - window + 1)
        scores = [
            self._compare_single_sequence(sequence, references, anomaly_threshold=anomaly_threshold).anomaly_score
            for sequence in sequences[start : index + 1]
        ]
        return float(sum(scores) / len(scores)) if scores else 0.0

    @staticmethod
    def _find_previous_same_sequence(sequences: list[ActiveSequence], index: int) -> ActiveSequence | None:
        """Return the most recent prior sequence with the same state word.

        Args:
            sequences: Full chronologically ordered sequence list.
            index: Position of the current sequence.

        Returns:
            The closest earlier sequence whose ``states`` equals that of
            ``sequences[index]``, or ``None`` if there is none.
        """
        current = sequences[index]
        # Note: backward linear scan; O(index) per call.
        for previous_index in range(index - 1, -1, -1):
            if sequences[previous_index].states == current.states:
                return sequences[previous_index]
        return None

    @staticmethod
    def _infer_family_candidate(
        transition_instability: float,
        duration_drift: float,
        anomaly_flag: bool,
    ) -> str:
        """Map sequence-level metrics into a coarse family candidate.

        Rules are evaluated in order:

        1. not anomalous -> ``"nominal_cycle"``;
        2. ``transition_instability >= 0.7`` -> ``"recurrent_spurious_cycle"``;
        3. ``duration_drift >= 0.5`` -> ``"process_saturation"``;
        4. otherwise -> ``"pump_abrupt_failure"``.

        Args:
            transition_instability: Normalised word change vs. the previous
                sequence, in :math:`[0, 1]`.
            duration_drift: Signed relative duration deviation from nominal.
            anomaly_flag: Whether the sequence is anomalous.

        Returns:
            The family candidate label.

        Notes:
            The thresholds are fixed heuristics, not learned parameters.
        """
        if not anomaly_flag:
            return "nominal_cycle"
        if transition_instability >= 0.7:
            return "recurrent_spurious_cycle"
        if duration_drift >= 0.5:
            return "process_saturation"
        return "pump_abrupt_failure"
