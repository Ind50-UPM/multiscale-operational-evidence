r"""Heuristic generation of candidate incident registries from operational history.

Role in the pipeline:
    Model_D needs outcome references for evaluation and risk targets, but the
    documented maintenance inventory is sparse. This module derives a
    *weakly labelled* candidate registry from three upstream products:
    Model_A state/power samples (``states``), Model_B run-length sequences
    (``sequences``) and symbolic state-word tables (``words``). It is run
    month by month by ``src/generate_candidate_incident_registry_batch.py``
    and its output follows the :mod:`~iabm_incidents.registry` schema. Every
    row is ``label_strength="weak"``: these are derived signal events, never
    independent truth.

Main public objects:
    * :class:`RegistryGenerationConfig` - all heuristic thresholds.
    * :class:`CandidateIncidentRegistryBuilder` - builder with file and frame
      entry points.

.. mermaid::

   flowchart TD
       ST[states: date, RP1..RP4, estado, pred_estado, source_month] --> NS[_normalize_states]
       SQ[sequences: date, span, Values, Runs, source_month] --> NQ[_normalize_sequences]
       WD[words: date, span, word, source_month] --> NW[_normalize_words]
       NS --> DT[_build_downtime_candidates<br/>external_ambiguous_disturbance]
       NQ --> DT
       NQ --> SA[_build_process_saturation_candidates<br/>process_saturation]
       NW --> RC[_build_recurrent_disturbance_candidates<br/>float_recurrent_disturbance]
       NS --> AB[_build_abrupt_failure_candidates<br/>pump_abrupt_failure + context redirects]
       NQ --> AB
       NW --> AB
       SA --> FS[_filter_saturation_overlaps_with_redirects]
       AB --> FS
       DT --> CAT[concat]
       FS --> CAT
       RC --> CAT
       AB --> CAT
       CAT --> EN[_enrich_with_word_evidence]
       EN --> MG[_merge_nearby_candidates]
       MG --> OUT[AUTO-nnnnnn registry, 20 columns]

Abrupt-failure cascade:
    1. *Raw drops*: relative total-power change
       :math:`\Delta P/P \le -` ``abrupt_drop_fraction``, power at or below
       the monthly ``low_power_quantile`` and a change of ``pred_estado``.
    2. *Support*: a saturation or recurrence candidate window of the same
       month within +/- ``supporting_word_window_hours``.
    3. *Context redirect*: a persistent-regime penalty (stable/recurrent
       words, high nominal match, long sequences, state 17) above
       ``abrupt_persistence_penalty_max_score`` with a competing family
       re-labels the drop as ``process_saturation``
       (``secondary_family="abrupt_context_redirect"``).
    4. *Rebound*: maximum power within ``abrupt_recovery_window_seconds``
       must reach ``abrupt_recovery_fraction`` of the pre-drop power.
    5. *Specificity*: evidence score >= ``abrupt_specific_evidence_min_score``
       and strictly greater than the persistence penalty.
    6. *Clustering*: drops closer than ``abrupt_cluster_gap_seconds`` in the
       same month collapse to the deepest one.

Data contracts:
    Input columns are listed in the diagram. The final registry has, in
    order: ``incident_id``, ``source_window_start``, ``source_window_end``,
    ``documented_start``, ``documented_end``, ``event_time``,
    ``event_time_precision`` (``"estimated"``), ``incident_family``,
    ``family``, ``secondary_family``, ``recovery_time``, ``recovery_status``,
    ``label_strength`` (``"weak"``), ``source_type``, ``downtime_start``,
    ``downtime_end``, ``maintenance_time``, ``affected_subsystem``
    (``"industrial_asset"``), ``notes`` and ``asset_id``
    (``"asset-unknown"``).

Maintenance notes:
    * All timestamps are normalised to naive UTC by
      :func:`_normalize_timestamp_series`.
    * Quantile thresholds are computed per ``source_month``; downtime gaps
      are computed on the globally sorted state series.
    * Intermediate candidate frames are projected onto
      :meth:`CandidateIncidentRegistryBuilder._candidate_columns`, so
      diagnostic columns such as ``abrupt_specificity_score`` do not reach
      the final registry (they survive only in ``notes``).
    * :meth:`CandidateIncidentRegistryBuilder._merge_nearby_candidates`
      compares ``secondary_family`` with ``==``; rows whose value is ``NaN``
      (e.g. recurrence candidates after concatenation) therefore never merge.
    * Recurrence detection groups rare words across all months and emits only
      the first qualifying burst per word.
    * State value ``17`` is treated as a known rare/persistent controller
      state of the studied asset; this is asset-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

import pandas as pd


@dataclass(frozen=True)
class RegistryGenerationConfig:
    """Thresholds controlling heuristic incident candidate generation.

    Durations are in seconds unless the name ends in ``_hours``; fractions and
    quantiles are in ``[0, 1]``; ``*_score`` and ``*_count`` fields are
    integer counts of satisfied criteria or occurrences.

    Attributes:
        downtime_gap_seconds: Minimum gap between consecutive state samples
            that creates a downtime candidate.
        low_power_quantile: Monthly total-power quantile below which a drop
            counts as low power.
        abrupt_drop_fraction: Minimum relative power drop of a raw abrupt
            candidate (0.5 = 50 %).
        saturation_span_quantile: Monthly sequence-duration quantile above
            which a sequence is a saturation candidate.
        rare_word_frequency_quantile: Monthly word-frequency quantile below
            which a word is rare.
        recurrence_window_hours: Sliding window for rare-word bursts.
        minimum_recurrence_count: Occurrences of a rare word within the
            window required for a recurrence candidate.
        candidate_merge_gap_seconds: Maximum gap for merging same-family
            candidates.
        supporting_word_window_hours: Half-width of the context window used
            for support, context and word enrichment.
        downtime_operational_stop_seconds: Gap length classified directly as
            ``operational_stop_candidate``.
        abrupt_cluster_gap_seconds: Clustering gap of abrupt candidates.
        abrupt_recovery_window_seconds: Look-ahead window for the power
            rebound.
        abrupt_recovery_fraction: Required rebound as a fraction of
            pre-drop power.
        abrupt_local_dominant_word_fraction: Dominant-word share defining a
            stable word regime.
        abrupt_local_transition_rate: Maximum word transition rate of a
            stable regime.
        abrupt_nominal_match_floor: Declared nominal-match floor (not read by
            the current heuristics).
        abrupt_persistent_nominal_match_floor: Nominal match at or above
            which a regime is persistent.
        abrupt_degraded_nominal_match_ceiling: Nominal match at or below
            which a regime is degraded.
        abrupt_recurrence_count_floor: Dominant-word count defining a
            recurrent regime.
        abrupt_specific_drop_fraction: Drop that earns a specificity point.
        abrupt_specific_rebound_fraction: Rebound/pre-drop ratio that earns a
            specificity point.
        abrupt_specific_transition_rate: Word transition rate that earns a
            specificity point.
        abrupt_specific_diversity_floor: Distinct-word count that earns a
            specificity point.
        abrupt_specific_evidence_min_score: Minimum specificity score.
        abrupt_persistence_penalty_max_score: Penalty above which a drop may
            be redirected.
        competing_family_penalty_threshold: Minimum penalty before a competing
            family is considered.
    """

    downtime_gap_seconds: float = 600.0
    low_power_quantile: float = 0.01
    abrupt_drop_fraction: float = 0.50
    saturation_span_quantile: float = 0.995
    rare_word_frequency_quantile: float = 0.10
    recurrence_window_hours: float = 24.0
    minimum_recurrence_count: int = 3
    candidate_merge_gap_seconds: float = 1800.0
    supporting_word_window_hours: float = 6.0
    downtime_operational_stop_seconds: float = 3600.0
    abrupt_cluster_gap_seconds: float = 1800.0
    abrupt_recovery_window_seconds: float = 1800.0
    abrupt_recovery_fraction: float = 0.75
    abrupt_local_dominant_word_fraction: float = 0.60
    abrupt_local_transition_rate: float = 0.40
    abrupt_nominal_match_floor: float = 0.50
    abrupt_persistent_nominal_match_floor: float = 0.75
    abrupt_degraded_nominal_match_ceiling: float = 0.35
    abrupt_recurrence_count_floor: int = 3
    abrupt_specific_drop_fraction: float = 0.85
    abrupt_specific_rebound_fraction: float = 0.85
    abrupt_specific_transition_rate: float = 0.55
    abrupt_specific_diversity_floor: int = 3
    abrupt_specific_evidence_min_score: int = 2
    abrupt_persistence_penalty_max_score: int = 0
    competing_family_penalty_threshold: int = 2


class CandidateIncidentRegistryBuilder:
    """Build a weakly labelled incident registry from states, sequences, and words.

    Args:
        config: Heuristic thresholds; defaults to
            :class:`RegistryGenerationConfig`.

    Examples:
        From frames already loaded in memory::

            builder = CandidateIncidentRegistryBuilder()
            registry = builder.build_from_frames(states=states, sequences=sequences, words=words)
            registry.to_csv("candidate_incident_registry.csv", index=False)
    """

    def __init__(self, config: RegistryGenerationConfig | None = None) -> None:
        """Resolve the configuration.

        Args:
            config: Optional thresholds; ``None`` uses the defaults.
        """
        self._config = config or RegistryGenerationConfig()

    def build_from_files(
        self,
        *,
        states_path: str | Path,
        sequences_path: str | Path,
        words_path: str | Path,
    ) -> pd.DataFrame:
        """Load the three upstream Parquet products and build the registry.

        Args:
            states_path: Model_A state/power Parquet file; only
                ``source_month``, ``date``, ``RP1``-``RP4``, ``estado`` and
                ``pred_estado`` are read (all must exist).
            sequences_path: Run-length sequence Parquet file (``date``,
                ``span``, ``source_month``, optional ``Values``/``Runs``).
            words_path: Symbolic word Parquet file (``date``, ``span``,
                ``word``, ``source_month``).

        Returns:
            pandas.DataFrame: Candidate registry (see
            :meth:`build_from_frames`).
        """
        states = pd.read_parquet(states_path, columns=["source_month", "date", "RP1", "RP2", "RP3", "RP4", "estado", "pred_estado"])
        sequences = pd.read_parquet(sequences_path)
        words = pd.read_parquet(words_path)
        return self.build_from_frames(states=states, sequences=sequences, words=words)

    def build_from_frames(
        self,
        *,
        states: pd.DataFrame,
        sequences: pd.DataFrame,
        words: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build the weakly labelled candidate registry from in-memory frames.

        Args:
            states: State/power samples with ``date`` and ``source_month``;
                ``RP1``-``RP4`` and ``pred_estado`` enable abrupt-failure
                candidates.
            sequences: Run-length sequences with ``date``, ``span`` (seconds)
                and ``source_month``; ``Values`` is renamed ``state_value``.
            words: Word table with ``date``, ``span``, ``word`` and
                ``source_month``.

        Returns:
            pandas.DataFrame: Chronologically ordered registry with identifiers
            ``AUTO-000001``, ``AUTO-000002``, ... and the 20 columns of the
            module data contract; an empty registry with those columns when
            no heuristic fires.

        Raises:
            KeyError: If ``date`` is missing from any input, or if a column
                required by an active heuristic (e.g. ``source_month`` or
                ``word``) is missing.
        """
        normalized_states = self._normalize_states(states)
        normalized_sequences = self._normalize_sequences(sequences)
        normalized_words = self._normalize_words(words)

        downtime = self._build_downtime_candidates(normalized_states, normalized_sequences)
        saturation = self._build_process_saturation_candidates(normalized_sequences)
        recurrence = self._build_recurrent_disturbance_candidates(normalized_words)
        abrupt = self._build_abrupt_failure_candidates(normalized_states, normalized_sequences, normalized_words)
        saturation = self._filter_saturation_overlaps_with_redirects(saturation, abrupt)

        candidate_parts = [frame for frame in [downtime, saturation, recurrence, abrupt] if not frame.empty]
        candidates = pd.concat(candidate_parts, ignore_index=True, sort=False) if candidate_parts else self._empty_candidate_frame()
        if candidates.empty:
            return self._empty_final_registry()

        candidates = self._enrich_with_word_evidence(candidates, normalized_words)
        candidates = self._merge_nearby_candidates(candidates)
        candidates = candidates.sort_values("event_time").reset_index(drop=True)
        # Note: identifiers are assigned after merging and sorting, so they are
        # stable only for identical inputs and configuration.
        candidates["incident_id"] = [f"AUTO-{index + 1:06d}" for index in range(len(candidates))]
        candidates["event_time_precision"] = candidates.get("event_time_precision", "estimated")
        candidates["label_strength"] = candidates.get("label_strength", "weak")
        candidates["source_type"] = candidates.get("source_type", "derived_candidate")
        candidates["secondary_family"] = candidates.get("secondary_family", None)
        candidates["recovery_status"] = candidates.get("recovery_status", None)
        candidates["affected_subsystem"] = candidates.get("affected_subsystem", "industrial_asset")
        candidates["asset_id"] = candidates.get("asset_id", "asset-unknown")
        candidates["family"] = candidates.get("incident_family")
        for column in ["recovery_time", "maintenance_time", "downtime_start", "downtime_end"]:
            if column not in candidates.columns:
                candidates[column] = None
        return candidates[
            [
                "incident_id",
                "source_window_start",
                "source_window_end",
                "documented_start",
                "documented_end",
                "event_time",
                "event_time_precision",
                "incident_family",
                "family",
                "secondary_family",
                "recovery_time",
                "recovery_status",
                "label_strength",
                "source_type",
                "downtime_start",
                "downtime_end",
                "maintenance_time",
                "affected_subsystem",
                "notes",
                "asset_id",
            ]
        ]

    def _build_downtime_candidates(self, states: pd.DataFrame, sequences: pd.DataFrame) -> pd.DataFrame:
        """Create ``external_ambiguous_disturbance`` candidates from sample gaps.

        A gap longer than ``downtime_gap_seconds`` between consecutive state
        samples (globally sorted) yields a candidate spanning the gap, with
        ``event_time`` at the last sample before it and ``secondary_family``
        from :meth:`_classify_gap`.

        Args:
            states: Normalised states (``event_time``).
            sequences: Normalised sequences used for gap classification.

        Returns:
            pandas.DataFrame: Candidates in the intermediate candidate schema.
        """
        if states.empty:
            return self._empty_candidate_frame()
        frame = states.sort_values("event_time").copy()
        frame["gap_seconds"] = frame["event_time"].diff().dt.total_seconds()
        gaps = frame[frame["gap_seconds"].gt(self._config.downtime_gap_seconds)].copy()
        if gaps.empty:
            return self._empty_candidate_frame()
        previous_times = frame["event_time"].shift(1)
        gaps["downtime_start"] = previous_times.loc[gaps.index].values
        gaps["downtime_end"] = gaps["event_time"]
        gaps["event_time"] = gaps["downtime_start"]
        gaps["source_window_start"] = gaps["downtime_start"]
        gaps["source_window_end"] = gaps["downtime_end"]
        gaps["documented_start"] = gaps["downtime_start"]
        gaps["documented_end"] = gaps["downtime_end"]
        gaps["incident_family"] = "external_ambiguous_disturbance"
        gaps["source_type"] = "derived_gap"
        gaps["label_strength"] = "weak"
        gap_classes: list[str] = []
        gap_notes: list[str] = []
        for _, row in gaps.iterrows():
            gap_class = self._classify_gap(
                pd.to_datetime(row["downtime_start"]),
                pd.to_datetime(row["downtime_end"]),
                float(row["gap_seconds"]),
                sequences,
            )
            gap_classes.append(gap_class)
            gap_notes.append(f"Detected {gap_class} gap of {row['gap_seconds']:.1f} seconds")
        gaps["secondary_family"] = gap_classes
        gaps["notes"] = gap_notes
        return self._with_candidate_schema(gaps)

    def _build_abrupt_failure_candidates(
        self,
        states: pd.DataFrame,
        sequences: pd.DataFrame,
        words: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create ``pump_abrupt_failure`` candidates and context redirects.

        Implements the six-step cascade described in the module docstring.

        Args:
            states: Normalised states with ``total_real_power`` and
                ``source_month`` (``pred_estado`` optional; without it no
                state change is detected and no candidate is produced).
            sequences: Normalised sequences.
            words: Normalised words.

        Returns:
            pandas.DataFrame: Abrupt candidates
            (``source_type="derived_state_power_supported"``, point windows at
            ``event_time``) followed by redirected ``process_saturation``
            candidates; empty when no drop is supported.
        """
        if states.empty or "total_real_power" not in states.columns:
            return self._empty_candidate_frame()
        frame = states.sort_values("event_time").copy()
        # Note: pct_change is computed on the global series, across month
        # boundaries; state changes are computed within each month.
        frame["relative_power_drop"] = frame["total_real_power"].pct_change().fillna(0.0)
        frame["state_changed"] = (
            frame.groupby("source_month")["pred_estado"].diff().fillna(0).ne(0)
            if "pred_estado" in frame.columns
            else False
        )
        monthly_threshold = frame.groupby("source_month")["total_real_power"].transform(
            lambda series: series.quantile(self._config.low_power_quantile)
        )
        raw = frame[
            frame["relative_power_drop"].le(-self._config.abrupt_drop_fraction)
            & frame["total_real_power"].le(monthly_threshold)
            & frame["state_changed"]
        ].copy()
        if raw.empty:
            return self._empty_candidate_frame()

        support_parts = []
        saturation_support = self._build_process_saturation_candidates(sequences)
        if not saturation_support.empty:
            support_parts.append(saturation_support[["source_window_start", "source_window_end", "source_month"]])
        recurrence_support = self._build_recurrent_disturbance_candidates(words)
        if not recurrence_support.empty:
            support_parts.append(recurrence_support[["source_window_start", "source_window_end", "source_month"]])
        support_windows = pd.concat(support_parts, ignore_index=True) if support_parts else pd.DataFrame(columns=["source_window_start", "source_window_end", "source_month"])
        if support_windows.empty:
            return self._empty_candidate_frame()

        support_delta = pd.Timedelta(hours=self._config.supporting_word_window_hours)
        monthly_sequence_duration_threshold = sequences.groupby("source_month")["duration_seconds"].quantile(
            self._config.saturation_span_quantile
        ).to_dict() if not sequences.empty else {}
        monthly_word_frequency = words.groupby(["source_month", "word"]).size().rename("word_count").reset_index() if not words.empty else pd.DataFrame(columns=["source_month", "word", "word_count"])
        words_with_frequency = words.merge(monthly_word_frequency, on=["source_month", "word"], how="left") if not words.empty else words.copy()
        words_by_month = {
            month: month_frame.sort_values("start_time").reset_index(drop=True)
            for month, month_frame in words_with_frequency.groupby("source_month", sort=False)
        }
        sequences_by_month = {
            month: month_frame.sort_values("start_time").reset_index(drop=True)
            for month, month_frame in sequences.groupby("source_month", sort=False)
        }
        supported_parts: list[pd.DataFrame] = []
        redirected_parts: list[pd.DataFrame] = []
        grouped_support = {month: month_frame.sort_values("source_window_start") for month, month_frame in support_windows.groupby("source_month", sort=False)}
        for month, month_raw in raw.groupby("source_month", sort=False):
            month_support = grouped_support.get(month)
            if month_support is None or month_support.empty:
                continue
            support_start = month_support["source_window_start"].to_numpy(dtype="datetime64[ns]").astype("int64")
            support_end = month_support["source_window_end"].to_numpy(dtype="datetime64[ns]").astype("int64")
            event_ns = month_raw["event_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
            lower = event_ns - int(support_delta.value)
            upper = event_ns + int(support_delta.value)
            keep_mask = np.zeros(len(month_raw), dtype=bool)
            for idx, (lo, hi) in enumerate(zip(lower, upper)):
                left = np.searchsorted(support_start, hi, side="right")
                if left == 0:
                    continue
                overlap_idx = np.flatnonzero(support_end[:left] >= lo)
                if overlap_idx.size:
                    keep_mask[idx] = True
            if keep_mask.any():
                supported = month_raw.loc[keep_mask].copy()
                kept, redirected = self._separate_abrupt_candidates_by_context(
                    supported,
                    sequences_by_month.get(month),
                    words_by_month.get(month),
                    monthly_sequence_duration_threshold.get(month),
                    support_delta,
                )
                if not kept.empty:
                    supported_parts.append(kept)
                if not redirected.empty:
                    redirected_parts.append(redirected)
        redirected_candidates = pd.concat(redirected_parts, ignore_index=True) if redirected_parts else self._empty_candidate_frame()
        # Note: steps 4-6 apply only to kept (non-redirected) candidates.
        if not supported_parts:
            return redirected_candidates
        candidates = pd.concat(supported_parts, ignore_index=True)
        candidates = self._filter_abrupt_candidates_with_recovery(candidates, frame)
        if candidates.empty:
            return self._empty_candidate_frame()
        candidates = self._filter_abrupt_candidates_by_specificity(
            candidates,
            sequences_by_month,
            words_by_month,
            support_delta,
        )
        if candidates.empty:
            return redirected_candidates
        candidates = self._cluster_abrupt_candidates(candidates)
        if candidates.empty:
            return redirected_candidates
        candidates["source_window_start"] = candidates["event_time"]
        candidates["source_window_end"] = candidates["event_time"]
        candidates["documented_start"] = candidates["event_time"]
        candidates["documented_end"] = candidates["event_time"]
        candidates["incident_family"] = "pump_abrupt_failure"
        candidates["notes"] = candidates.apply(
            lambda row: (
                f"Abrupt power drop {row['relative_power_drop']:.2%} with specific abrupt evidence "
                f"(score={int(row.get('abrupt_specificity_score', 0))}, penalty={int(row.get('abrupt_persistence_penalty', 0))}), "
                f"low power {row['total_real_power']:.2f} and rebound {row['recovery_power']:.2f}"
            ),
            axis=1,
        )
        candidates["source_type"] = "derived_state_power_supported"
        candidates["label_strength"] = "weak"
        abrupt_candidates = self._with_candidate_schema(candidates)
        if redirected_candidates.empty:
            return abrupt_candidates
        return pd.concat([abrupt_candidates, redirected_candidates], ignore_index=True, sort=False)

    def _separate_abrupt_candidates_by_context(
        self,
        candidates: pd.DataFrame,
        month_sequences: pd.DataFrame | None,
        month_words: pd.DataFrame | None,
        sequence_duration_threshold: float | None,
        support_delta: pd.Timedelta,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split supported drops into kept abrupt candidates and redirects.

        A drop is redirected when its persistent-regime penalty exceeds
        ``abrupt_persistence_penalty_max_score`` *and*
        :meth:`_preferred_competing_family` proposes a family.

        Args:
            candidates: Supported raw drops of one month.
            month_sequences: Sequences of that month (sorted) or ``None``.
            month_words: Words of that month with ``word_count`` or ``None``.
            sequence_duration_threshold: Monthly long-sequence threshold
                (seconds) or ``None``.
            support_delta: Half-width of the context window.

        Returns:
            tuple[pandas.DataFrame, pandas.DataFrame]: ``(kept, redirected)``;
            ``redirected`` rows are built by :meth:`_build_competing_candidate`.
        """
        if candidates.empty:
            return candidates, self._empty_candidate_frame()
        keep_rows: list[pd.Series] = []
        redirected_rows: list[dict[str, object]] = []
        for _, row in candidates.iterrows():
            event_time = pd.to_datetime(row["event_time"])
            sequence_window = self._nearby_sequence_window(month_sequences, event_time, support_delta)
            word_window = self._nearby_word_window(month_words, event_time, support_delta)
            penalty = self._persistent_regime_penalty(sequence_window, word_window, sequence_duration_threshold)
            preferred_family = self._preferred_competing_family(sequence_window, word_window, sequence_duration_threshold, penalty)
            if penalty > self._config.abrupt_persistence_penalty_max_score and preferred_family is not None:
                redirected_rows.append(
                    self._build_competing_candidate(row, sequence_window, preferred_family, penalty)
                )
            else:
                keep_rows.append(row)
        kept = pd.DataFrame(keep_rows) if keep_rows else candidates.iloc[0:0].copy()
        redirected = pd.DataFrame(redirected_rows) if redirected_rows else self._empty_candidate_frame()
        return kept, redirected

    def _filter_abrupt_candidates_by_specificity(
        self,
        candidates: pd.DataFrame,
        sequences_by_month: dict[object, pd.DataFrame],
        words_by_month: dict[object, pd.DataFrame],
        support_delta: pd.Timedelta,
    ) -> pd.DataFrame:
        """Keep abrupt candidates whose specific evidence beats persistence.

        A candidate is kept when its :meth:`_abrupt_specificity_score` is at
        least ``abrupt_specific_evidence_min_score`` and strictly greater than
        its :meth:`_persistent_regime_penalty` (computed without the
        long-sequence criterion).

        Args:
            candidates: Rebound-filtered candidates (with ``recovery_power``).
            sequences_by_month: Month to sorted sequence frame.
            words_by_month: Month to sorted word frame with ``word_count``.
            support_delta: Half-width of the context window.

        Returns:
            pandas.DataFrame: Kept rows with added
            ``abrupt_specificity_score`` and ``abrupt_persistence_penalty``.
        """
        if candidates.empty:
            return candidates
        keep_mask = np.zeros(len(candidates), dtype=bool)
        evidence_scores = np.zeros(len(candidates), dtype=int)
        penalty_scores = np.zeros(len(candidates), dtype=int)
        for position, (_, row) in enumerate(candidates.iterrows()):
            event_time = pd.to_datetime(row["event_time"])
            month = row.get("source_month")
            sequence_window = self._nearby_sequence_window(sequences_by_month.get(month), event_time, support_delta)
            word_window = self._nearby_word_window(words_by_month.get(month), event_time, support_delta)
            evidence_score = self._abrupt_specificity_score(row, sequence_window, word_window)
            penalty_score = self._persistent_regime_penalty(sequence_window, word_window, None)
            evidence_scores[position] = evidence_score
            penalty_scores[position] = penalty_score
            keep_mask[position] = evidence_score >= self._config.abrupt_specific_evidence_min_score and evidence_score > penalty_score
        kept = candidates.loc[keep_mask].copy()
        if kept.empty:
            return kept
        kept["abrupt_specificity_score"] = evidence_scores[keep_mask]
        kept["abrupt_persistence_penalty"] = penalty_scores[keep_mask]
        return kept

    def _abrupt_specificity_score(
        self,
        candidate_row: pd.Series,
        nearby_sequences: pd.DataFrame,
        nearby_words: pd.DataFrame,
    ) -> int:
        """Count abrupt-specific evidence criteria (0-5).

        One point each for: drop at least ``abrupt_specific_drop_fraction``;
        rebound/pre-drop power at least
        ``abrupt_specific_rebound_fraction``; local word transition rate at
        least ``abrupt_specific_transition_rate``; at least
        ``abrupt_specific_diversity_floor`` distinct words; at least two
        distinct nearby ``state_value`` values.

        Args:
            candidate_row: Candidate with ``relative_power_drop``,
                ``total_real_power`` and ``recovery_power``.
            nearby_sequences: Sequences overlapping the context window.
            nearby_words: Words overlapping the context window.

        Returns:
            int: Specificity score.
        """
        score = 0
        relative_drop = abs(float(candidate_row.get("relative_power_drop", 0.0)))
        if relative_drop >= self._config.abrupt_specific_drop_fraction:
            score += 1

        current_power = float(candidate_row.get("total_real_power", 0.0))
        # Why: pct_change = P_t/P_{t-1} - 1, hence P_{t-1} = P_t/(1 + change).
        previous_power = current_power / max(1.0 + float(candidate_row.get("relative_power_drop", 0.0)), 1e-6)
        rebound_power = float(candidate_row.get("recovery_power", 0.0)) if pd.notna(candidate_row.get("recovery_power")) else 0.0
        if previous_power > 0.0 and rebound_power / previous_power >= self._config.abrupt_specific_rebound_fraction:
            score += 1

        word_metrics = self._word_regime_metrics(nearby_words)
        if word_metrics["transition_rate"] >= self._config.abrupt_specific_transition_rate:
            score += 1
        if word_metrics["diversity"] >= self._config.abrupt_specific_diversity_floor:
            score += 1

        if not nearby_sequences.empty and "state_value" in nearby_sequences.columns:
            state_values = pd.to_numeric(nearby_sequences["state_value"], errors="coerce").dropna()
            if state_values.nunique() >= 2:
                score += 1
        return score

    def _preferred_competing_family(
        self,
        nearby_sequences: pd.DataFrame,
        nearby_words: pd.DataFrame,
        sequence_duration_threshold: float | None,
        penalty_score: int,
    ) -> str | None:
        """Propose a competing family for a persistent-regime drop.

        Args:
            nearby_sequences: Context sequences.
            nearby_words: Context words.
            sequence_duration_threshold: Long-sequence threshold or ``None``.
            penalty_score: Persistent-regime penalty of the drop.

        Returns:
            str | None: ``"process_saturation"`` when the penalty reaches
            ``competing_family_penalty_threshold`` and the context shows a
            long sequence, state 17, a degraded-but-recurrent regime or a
            persistent nominal regime; otherwise ``None``.
        """
        if penalty_score < self._config.competing_family_penalty_threshold:
            return None
        if self._has_long_sequence(nearby_sequences, sequence_duration_threshold) or self._has_state17(nearby_sequences):
            return "process_saturation"
        nominal_match_mean = self._mean_nominal_match(nearby_sequences)
        word_metrics = self._word_regime_metrics(nearby_words)
        if nominal_match_mean <= self._config.abrupt_degraded_nominal_match_ceiling and word_metrics["dominant_count"] >= self._config.abrupt_recurrence_count_floor:
            return "process_saturation"
        if nominal_match_mean >= self._config.abrupt_persistent_nominal_match_floor and word_metrics["dominant_fraction"] >= self._config.abrupt_local_dominant_word_fraction:
            return "process_saturation"
        return None

    def _build_competing_candidate(
        self,
        candidate_row: pd.Series,
        nearby_sequences: pd.DataFrame,
        preferred_family: str,
        penalty_score: int,
    ) -> dict[str, object]:
        """Build a redirected candidate record for a competing family.

        Args:
            candidate_row: Original abrupt drop.
            nearby_sequences: Context sequences; their overall span becomes
                the source window (point window at ``event_time`` if empty).
            preferred_family: Family proposed by
                :meth:`_preferred_competing_family`.
            penalty_score: Persistent-regime penalty, recorded in ``notes``.

        Returns:
            dict[str, object]: Candidate record with
            ``secondary_family="abrupt_context_redirect"`` and
            ``source_type="derived_abrupt_context_redirect"``.
        """
        if not nearby_sequences.empty:
            source_window_start = nearby_sequences["start_time"].min()
            source_window_end = nearby_sequences["end_time"].max()
            duration_seconds = float(
                pd.to_numeric(nearby_sequences.get("duration_seconds"), errors="coerce").fillna(0.0).max()
            )
            state_value = pd.to_numeric(nearby_sequences.get("state_value"), errors="coerce").dropna()
            dominant_state = int(state_value.mode().iloc[0]) if not state_value.empty else None
        else:
            source_window_start = pd.to_datetime(candidate_row["event_time"])
            source_window_end = pd.to_datetime(candidate_row["event_time"])
            duration_seconds = 0.0
            dominant_state = None
        return {
            "source_window_start": source_window_start,
            "source_window_end": source_window_end,
            "documented_start": source_window_start,
            "documented_end": source_window_end,
            "event_time": pd.to_datetime(candidate_row["event_time"]),
            "incident_family": preferred_family,
            "source_month": candidate_row.get("source_month"),
            "secondary_family": "abrupt_context_redirect",
            "notes": f"Context redirected from abrupt candidate to {preferred_family} (penalty={penalty_score}, state={dominant_state}, span={duration_seconds:.1f}s)",
            "source_type": "derived_abrupt_context_redirect",
            "label_strength": "weak",
        }

    def _persistent_regime_penalty(
        self,
        nearby_sequences: pd.DataFrame,
        nearby_words: pd.DataFrame,
        sequence_duration_threshold: float | None,
    ) -> int:
        """Score how strongly the context indicates a persistent regime (0-5).

        A *stable* word regime has dominant-word share at least
        ``abrupt_local_dominant_word_fraction`` and transition rate at most
        ``abrupt_local_transition_rate``; a *recurrent* one has a dominant-word
        count at least ``abrupt_recurrence_count_floor``. One point each for:
        stable and recurrent; stable with nominal match at least
        ``abrupt_persistent_nominal_match_floor``; stable with a long
        sequence; stable with state 17; recurrent with nominal match at most
        ``abrupt_degraded_nominal_match_ceiling``.

        Args:
            nearby_sequences: Context sequences.
            nearby_words: Context words.
            sequence_duration_threshold: Long-sequence threshold; ``None``
                disables that criterion.

        Returns:
            int: Penalty score; ``0`` without any context.
        """
        if nearby_sequences.empty and nearby_words.empty:
            return 0

        penalty = 0
        word_metrics = self._word_regime_metrics(nearby_words)
        stable_word_regime = (
            word_metrics["dominant_fraction"] >= self._config.abrupt_local_dominant_word_fraction
            and word_metrics["transition_rate"] <= self._config.abrupt_local_transition_rate
        )
        recurrent_word_regime = word_metrics["dominant_count"] >= self._config.abrupt_recurrence_count_floor
        nominal_match_mean = self._mean_nominal_match(nearby_sequences)
        long_sequence = self._has_long_sequence(nearby_sequences, sequence_duration_threshold)
        state17_present = self._has_state17(nearby_sequences)

        if stable_word_regime and recurrent_word_regime:
            penalty += 1
        if stable_word_regime and nominal_match_mean >= self._config.abrupt_persistent_nominal_match_floor:
            penalty += 1
        if stable_word_regime and long_sequence:
            penalty += 1
        if state17_present and stable_word_regime:
            penalty += 1
        if recurrent_word_regime and nominal_match_mean <= self._config.abrupt_degraded_nominal_match_ceiling:
            penalty += 1
        return penalty

    @staticmethod
    def _has_long_sequence(nearby_sequences: pd.DataFrame, sequence_duration_threshold: float | None) -> bool:
        """Return whether any context sequence reaches the duration threshold.

        Args:
            nearby_sequences: Context sequences (``duration_seconds``).
            sequence_duration_threshold: Threshold in seconds or ``None``.

        Returns:
            bool: ``False`` for empty context or a missing threshold.
        """
        if nearby_sequences.empty or sequence_duration_threshold is None or pd.isna(sequence_duration_threshold):
            return False
        return bool(
            pd.to_numeric(nearby_sequences.get("duration_seconds"), errors="coerce")
            .fillna(0.0)
            .ge(float(sequence_duration_threshold))
            .any()
        )

    @staticmethod
    def _has_state17(nearby_sequences: pd.DataFrame) -> bool:
        """Return whether controller state value 17 occurs in the context.

        Args:
            nearby_sequences: Context sequences (``state_value``).

        Returns:
            bool: ``True`` if any ``state_value`` equals 17.
        """
        if nearby_sequences.empty or "state_value" not in nearby_sequences.columns:
            return False
        state_values = pd.to_numeric(nearby_sequences["state_value"], errors="coerce")
        return bool(state_values.eq(17).any())

    @staticmethod
    def _mean_nominal_match(nearby_sequences: pd.DataFrame) -> float:
        """Return the mean ``nominal_match`` of the context sequences.

        Args:
            nearby_sequences: Context sequences.

        Returns:
            float: Mean nominal match, or the neutral value ``0.5`` when the
            column or valid values are absent.
        """
        if nearby_sequences.empty or "nominal_match" not in nearby_sequences.columns:
            return 0.5
        nominal_match = pd.to_numeric(nearby_sequences["nominal_match"], errors="coerce").dropna()
        if nominal_match.empty:
            return 0.5
        return float(nominal_match.mean())

    @staticmethod
    def _word_regime_metrics(nearby_words: pd.DataFrame) -> dict[str, float]:
        """Summarise the local word regime in chronological order.

        Args:
            nearby_words: Context words (``word``, ``start_time``).

        Returns:
            dict[str, float]: ``dominant_fraction`` (share of the most
            frequent word), ``transition_rate`` (changes between consecutive
            words divided by ``n - 1``), ``diversity`` (distinct words) and
            ``dominant_count``. Without words the neutral values are
            ``0, 1, 0, 0`` (maximally unstable).
        """
        if nearby_words.empty or "word" not in nearby_words.columns:
            return {
                "dominant_fraction": 0.0,
                "transition_rate": 1.0,
                "diversity": 0.0,
                "dominant_count": 0.0,
            }
        ordered_words = nearby_words.sort_values("start_time")["word"].dropna().astype(str)
        if ordered_words.empty:
            return {
                "dominant_fraction": 0.0,
                "transition_rate": 1.0,
                "diversity": 0.0,
                "dominant_count": 0.0,
            }
        counts = ordered_words.value_counts()
        # Note: the first element always differs from its NaN shift, hence -1.
        transitions = int((ordered_words != ordered_words.shift(1)).sum() - 1) if len(ordered_words) > 1 else 0
        return {
            "dominant_fraction": float(counts.iloc[0] / max(int(counts.sum()), 1)),
            "transition_rate": float(transitions / max(len(ordered_words) - 1, 1)),
            "diversity": float(ordered_words.nunique()),
            "dominant_count": float(counts.iloc[0]),
        }

    @staticmethod
    def _nearby_sequence_window(
        month_sequences: pd.DataFrame | None,
        event_time: pd.Timestamp,
        support_delta: pd.Timedelta,
    ) -> pd.DataFrame:
        """Return sequences overlapping ``[event_time - delta, event_time + delta]``.

        Args:
            month_sequences: Sequences of the month or ``None``.
            event_time: Candidate time.
            support_delta: Half-width of the window.

        Returns:
            pandas.DataFrame: Overlapping rows (inclusive bounds); an empty
            frame without columns when no sequences are available.
        """
        if month_sequences is None or month_sequences.empty:
            return pd.DataFrame()
        mask = (
            month_sequences["start_time"].le(event_time + support_delta)
            & month_sequences["end_time"].ge(event_time - support_delta)
        )
        return month_sequences.loc[mask].copy()

    @staticmethod
    def _nearby_word_window(
        month_words: pd.DataFrame | None,
        event_time: pd.Timestamp,
        support_delta: pd.Timedelta,
    ) -> pd.DataFrame:
        """Return words overlapping ``[event_time - delta, event_time + delta]``.

        Args:
            month_words: Words of the month or ``None``.
            event_time: Candidate time.
            support_delta: Half-width of the window.

        Returns:
            pandas.DataFrame: Overlapping rows (inclusive bounds); an empty
            frame without columns when no words are available.
        """
        if month_words is None or month_words.empty:
            return pd.DataFrame()
        mask = (
            month_words["start_time"].le(event_time + support_delta)
            & month_words["end_time"].ge(event_time - support_delta)
        )
        return month_words.loc[mask].copy()

    def _build_process_saturation_candidates(self, sequences: pd.DataFrame) -> pd.DataFrame:
        """Create ``process_saturation`` candidates from unusually long sequences.

        A sequence whose ``duration_seconds`` is at or above the monthly
        ``saturation_span_quantile`` becomes a candidate spanning the
        sequence, with ``event_time`` at its start.

        Args:
            sequences: Normalised sequences with ``source_month``.

        Returns:
            pandas.DataFrame: Candidates in the intermediate schema
            (``source_type="derived_sequence_duration"``).
        """
        if sequences.empty:
            return self._empty_candidate_frame()
        frame = sequences.copy()
        monthly_threshold = frame.groupby("source_month")["duration_seconds"].transform(
            lambda series: series.quantile(self._config.saturation_span_quantile)
        )
        candidates = frame[frame["duration_seconds"].ge(monthly_threshold)].copy()
        if candidates.empty:
            return self._empty_candidate_frame()
        candidates["event_time"] = candidates["start_time"]
        candidates["source_window_start"] = candidates["start_time"]
        candidates["source_window_end"] = candidates["end_time"]
        candidates["documented_start"] = candidates["start_time"]
        candidates["documented_end"] = candidates["end_time"]
        candidates["incident_family"] = "process_saturation"
        candidates["source_month"] = candidates.get("source_month")
        candidates["notes"] = candidates.apply(
            lambda row: f"Long sequence span {row['duration_seconds']:.1f}s for value {row.get('state_value')}",
            axis=1,
        )
        candidates["source_type"] = "derived_sequence_duration"
        candidates["label_strength"] = "weak"
        return self._with_candidate_schema(candidates)

    def _filter_saturation_overlaps_with_redirects(
        self,
        saturation: pd.DataFrame,
        abrupt: pd.DataFrame,
    ) -> pd.DataFrame:
        """Drop saturation candidates already covered by an abrupt redirect.

        Args:
            saturation: Saturation candidates.
            abrupt: Abrupt output, possibly containing
                ``abrupt_context_redirect`` rows.

        Returns:
            pandas.DataFrame: Saturation candidates whose source window does
            not overlap (inclusive) a redirect window of the same month.
        """
        if saturation.empty or abrupt.empty or "secondary_family" not in abrupt.columns:
            return saturation
        redirects = abrupt[abrupt["secondary_family"] == "abrupt_context_redirect"].copy()
        if redirects.empty:
            return saturation
        keep_mask = np.ones(len(saturation), dtype=bool)
        for position, (_, sat_row) in enumerate(saturation.iterrows()):
            sat_month = sat_row.get("source_month")
            sat_start = pd.to_datetime(sat_row.get("source_window_start"))
            sat_end = pd.to_datetime(sat_row.get("source_window_end"))
            month_redirects = redirects[redirects["source_month"] == sat_month]
            if month_redirects.empty:
                continue
            overlap = month_redirects[
                (pd.to_datetime(month_redirects["source_window_start"]) <= sat_end)
                & (pd.to_datetime(month_redirects["source_window_end"]) >= sat_start)
            ]
            if not overlap.empty:
                keep_mask[position] = False
        return saturation.loc[keep_mask].copy()

    def _build_recurrent_disturbance_candidates(self, words: pd.DataFrame) -> pd.DataFrame:
        """Create ``float_recurrent_disturbance`` candidates from rare-word bursts.

        Words whose monthly frequency is at or below the monthly
        ``rare_word_frequency_quantile`` are rare. For each rare word, the
        first start time from which at least ``minimum_recurrence_count``
        occurrences start within ``recurrence_window_hours`` defines one
        candidate spanning that burst.

        Args:
            words: Normalised words with ``word`` and ``source_month``.

        Returns:
            pandas.DataFrame: At most one candidate per rare word
            (``source_type="derived_word_recurrence"``); not projected onto
            the intermediate schema (no ``secondary_family`` column).
        """
        if words.empty:
            return self._empty_candidate_frame()
        frame = words.copy()
        frequency = frame.groupby(["source_month", "word"]).size().rename("word_count").reset_index()
        frame = frame.merge(frequency, on=["source_month", "word"], how="left")
        monthly_cutoff = frame.groupby("source_month")["word_count"].transform(
            lambda series: series.quantile(self._config.rare_word_frequency_quantile)
        )
        rare = frame[frame["word_count"].le(monthly_cutoff)].copy()
        if rare.empty:
            return self._empty_candidate_frame()
        rare = rare.sort_values(["word", "start_time"])
        window = pd.Timedelta(hours=self._config.recurrence_window_hours)
        window_ns = int(window.value)
        rows: list[dict[str, object]] = []
        for word, word_frame in rare.groupby("word", sort=False):
            starts = word_frame["start_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
            ends = word_frame["end_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
            if len(starts) < self._config.minimum_recurrence_count:
                continue
            right_idx = np.searchsorted(starts, starts + window_ns, side="right")
            counts = right_idx - np.arange(len(starts))
            valid = np.flatnonzero(counts >= self._config.minimum_recurrence_count)
            if valid.size == 0:
                continue
            idx = int(valid[0])
            end_idx = int(right_idx[idx] - 1)
            rows.append(
                {
                    "source_window_start": word_frame["start_time"].iloc[idx],
                    "source_window_end": word_frame["end_time"].iloc[end_idx],
                    "documented_start": word_frame["start_time"].iloc[idx],
                    "documented_end": word_frame["end_time"].iloc[end_idx],
                    "event_time": word_frame["start_time"].iloc[idx],
                    "incident_family": "float_recurrent_disturbance",
                    "source_month": word_frame["source_month"].iloc[idx],
                    "notes": f"Rare word burst '{word}' repeated {int(counts[idx])} times in {self._config.recurrence_window_hours:.0f}h",
                    "source_type": "derived_word_recurrence",
                    "label_strength": "weak",
                }
            )
        return pd.DataFrame(rows) if rows else self._empty_candidate_frame()

    def _enrich_with_word_evidence(self, candidates: pd.DataFrame, words: pd.DataFrame) -> pd.DataFrame:
        """Append the three most frequent nearby words to each candidate's notes.

        Args:
            candidates: Concatenated candidates.
            words: Normalised words.

        Returns:
            pandas.DataFrame: Copy with ``notes`` extended by
            ``"; supporting words: w (count=n), ..."`` when words of the same
            month overlap the source window widened by
            ``supporting_word_window_hours``.
        """
        if candidates.empty or words.empty:
            return candidates
        frame = candidates.copy()
        word_support_window = pd.Timedelta(hours=self._config.supporting_word_window_hours)
        word_frequency = words.groupby(["source_month", "word"]).size().rename("word_count").reset_index()
        words_with_frequency = words.merge(word_frequency, on=["source_month", "word"], how="left")
        words_by_month = {month: month_frame.sort_values(["start_time", "word_count", "duration_seconds"]) for month, month_frame in words_with_frequency.groupby("source_month")}
        enriched_notes = []
        for _, row in frame.iterrows():
            start = pd.to_datetime(row["source_window_start"])
            end = pd.to_datetime(row["source_window_end"])
            month_words = words_by_month.get(row.get("source_month"))
            if month_words is None or month_words.empty:
                enriched_notes.append(row.get("notes"))
                continue
            mask = (
                month_words["start_time"].le(end + word_support_window)
                & month_words["end_time"].ge(start - word_support_window)
            )
            nearby_words = month_words.loc[mask, ["word", "word_count", "duration_seconds"]]
            if nearby_words.empty:
                enriched_notes.append(row.get("notes"))
                continue
            top_words = nearby_words.nlargest(3, ["word_count", "duration_seconds"])
            word_note = "; supporting words: " + ", ".join(
                f"{word} (count={count})" for word, count in zip(top_words["word"], top_words["word_count"])
            )
            enriched_notes.append(f"{row.get('notes', '')}{word_note}")
        frame["notes"] = enriched_notes
        return frame

    def _merge_nearby_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Merge consecutive candidates of the same family and secondary family.

        Candidates are sorted by family and ``event_time``; a candidate whose
        ``source_window_start`` is at most ``candidate_merge_gap_seconds``
        after the running ``source_window_end`` is absorbed (window and
        documented end extended, ``merge_count`` accumulated, notes merged).

        Args:
            candidates: Enriched candidates.

        Returns:
            pandas.DataFrame: Merged candidates; the first row's
            ``event_time`` and start are kept.
        """
        if candidates.empty:
            return candidates
        frame = candidates.sort_values(["incident_family", "event_time"]).reset_index(drop=True)
        merge_gap = pd.Timedelta(seconds=self._config.candidate_merge_gap_seconds)
        rows: list[dict[str, object]] = []
        current: dict[str, object] | None = None
        for _, row in frame.iterrows():
            row_dict = row.to_dict()
            row_start = pd.to_datetime(row_dict["source_window_start"])
            row_end = pd.to_datetime(row_dict["source_window_end"])
            if current is None:
                current = row_dict
                continue
            current_end = pd.to_datetime(current["source_window_end"])
            same_family = current["incident_family"] == row_dict["incident_family"]
            same_secondary_family = current.get("secondary_family") == row_dict.get("secondary_family")
            close_in_time = row_start - current_end <= merge_gap
            if same_family and same_secondary_family and close_in_time:
                current["source_window_end"] = max(current_end, row_end)
                current["documented_end"] = max(pd.to_datetime(current["documented_end"]), pd.to_datetime(row_dict["documented_end"]))
                current["merge_count"] = int(current.get("merge_count", 1)) + int(row_dict.get("merge_count", 1))
                current["notes"] = self._merge_notes(current.get("notes"), row_dict.get("notes"), current["merge_count"])
                continue
            rows.append(current)
            current = row_dict
        if current is not None:
            rows.append(current)
        return pd.DataFrame(rows)


    @staticmethod
    def _merge_notes(current_note: object, new_note: object, merge_count: int) -> str:
        """Merge free-text notes with bounded length.

        Args:
            current_note: Accumulated note.
            new_note: Note of the absorbed candidate.
            merge_count: Total number of merged observations.

        Returns:
            str: For up to two observations, both notes joined by
            ``" | merged: "``; afterwards a base note plus
            ``" | merged summary: <n> nearby observations"``. Always at most
            1200 characters.
        """
        current_text = str(current_note or "").strip()
        new_text = str(new_note or "").strip()
        if merge_count <= 2:
            merged = " | merged: ".join(part for part in [current_text, new_text] if part)
            return merged[:1200]
        if not current_text:
            return new_text[:1200]
        if " | merged summary:" in current_text:
            summary_base = current_text.split(" | merged summary:", 1)[0]
        else:
            summary_base = current_text
        if new_text and new_text != summary_base:
            return f"{summary_base} | merged summary: {merge_count} nearby observations"[:1200]
        return f"{summary_base} | merged summary: {merge_count} nearby observations"[:1200]

    @staticmethod
    def _normalize_states(states: pd.DataFrame) -> pd.DataFrame:
        """Normalise state samples.

        Args:
            states: Raw states with ``date`` and optional ``RP1``-``RP4``,
                ``estado`` and ``pred_estado``.

        Returns:
            pandas.DataFrame: Copy with naive-UTC ``event_time``,
            ``total_real_power`` (sum of available ``RP*`` columns, missing as
            0) and numeric ``estado``/``pred_estado``.

        Raises:
            KeyError: If ``date`` is missing.
        """
        frame = states.copy()
        frame["event_time"] = _normalize_timestamp_series(frame["date"])
        power_columns = [column for column in ["RP1", "RP2", "RP3", "RP4"] if column in frame.columns]
        if power_columns:
            frame["total_real_power"] = frame[power_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
        for column in ["estado", "pred_estado"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    def _classify_gap(
        self,
        gap_start: pd.Timestamp,
        gap_end: pd.Timestamp,
        gap_seconds: float,
        sequences: pd.DataFrame,
    ) -> str:
        """Classify a sample gap as telemetry loss or operational stop.

        Args:
            gap_start: Last sample before the gap.
            gap_end: First sample after the gap.
            gap_seconds: Gap length.
            sequences: Normalised sequences.

        Returns:
            str: ``"operational_stop_candidate"`` for gaps of at least
            ``downtime_operational_stop_seconds`` or when a sequence with
            ``state_value == 0`` overlaps the gap +/- 5 minutes; otherwise
            ``"telemetry"``.
        """
        if gap_seconds >= self._config.downtime_operational_stop_seconds:
            return "operational_stop_candidate"
        if sequences.empty or "start_time" not in sequences.columns:
            return "telemetry"
        nearby = sequences[
            (sequences["start_time"] <= gap_end + pd.Timedelta(minutes=5))
            & (sequences["end_time"] >= gap_start - pd.Timedelta(minutes=5))
        ]
        if nearby.empty:
            return "telemetry"
        state_values = pd.to_numeric(nearby.get("state_value"), errors="coerce")
        if state_values.eq(0).any():
            return "operational_stop_candidate"
        return "telemetry"

    def _filter_abrupt_candidates_with_recovery(self, candidates: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
        """Keep abrupt drops followed by a power rebound.

        For each drop at time :math:`t`, the maximum total power over samples
        in :math:`(t, t + W]` (``W`` = ``abrupt_recovery_window_seconds``)
        must reach ``abrupt_recovery_fraction`` times the pre-drop power.

        Args:
            candidates: Kept supported drops.
            states: Normalised states (all months).

        Returns:
            pandas.DataFrame: Kept rows with added ``recovery_power``.
        """
        if candidates.empty:
            return candidates
        rebound_fraction = self._config.abrupt_recovery_fraction
        recovery_window_ns = int(pd.Timedelta(seconds=self._config.abrupt_recovery_window_seconds).value)
        rows: list[pd.DataFrame] = []
        grouped_states = {month: month_frame.sort_values("event_time") for month, month_frame in states.groupby("source_month", sort=False)}
        for month, month_candidates in candidates.groupby("source_month", sort=False):
            month_states = grouped_states.get(month)
            if month_states is None or month_states.empty:
                continue
            state_times = month_states["event_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
            state_power = month_states["total_real_power"].to_numpy(dtype=float)
            event_ns = month_candidates["event_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
            start_idx = np.searchsorted(state_times, event_ns, side="right")
            end_idx = np.searchsorted(state_times, event_ns + recovery_window_ns, side="right")
            recovery_power = np.full(len(month_candidates), np.nan, dtype=float)
            keep_mask = np.zeros(len(month_candidates), dtype=bool)
            previous_power = month_candidates["total_real_power"].to_numpy(dtype=float) / np.maximum(
                1.0 + month_candidates["relative_power_drop"].to_numpy(dtype=float),
                1e-6,
            )
            for idx, (left, right) in enumerate(zip(start_idx, end_idx)):
                if right <= left:
                    continue
                max_power = float(state_power[left:right].max())
                recovery_power[idx] = max_power
                if max_power >= previous_power[idx] * rebound_fraction:
                    keep_mask[idx] = True
            if keep_mask.any():
                kept = month_candidates.loc[keep_mask].copy()
                kept["recovery_power"] = recovery_power[keep_mask]
                rows.append(kept)
        return pd.concat(rows, ignore_index=True) if rows else candidates.iloc[0:0].copy()

    def _cluster_abrupt_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Collapse nearby abrupt drops of the same month to the deepest one.

        Rows are scanned chronologically; a row within
        ``abrupt_cluster_gap_seconds`` of the current representative replaces
        it if its drop is deeper (more negative). Because the representative
        can move forward, clusters may chain beyond one gap length.

        Args:
            candidates: Specificity-filtered candidates.

        Returns:
            pandas.DataFrame: One row per cluster.
        """
        if candidates.empty:
            return candidates
        frame = candidates.sort_values(["source_month", "event_time"]).reset_index(drop=True)
        cluster_gap = pd.Timedelta(seconds=self._config.abrupt_cluster_gap_seconds)
        rows: list[dict[str, object]] = []
        current: dict[str, object] | None = None
        for _, row in frame.iterrows():
            row_dict = row.to_dict()
            row_time = pd.to_datetime(row_dict["event_time"])
            if current is None:
                current = row_dict
                continue
            current_time = pd.to_datetime(current["event_time"])
            same_month = current["source_month"] == row_dict["source_month"]
            if same_month and row_time - current_time <= cluster_gap:
                if row_dict["relative_power_drop"] < current["relative_power_drop"]:
                    current = row_dict
                continue
            rows.append(current)
            current = row_dict
        if current is not None:
            rows.append(current)
        return pd.DataFrame(rows)

    @staticmethod
    def _normalize_sequences(sequences: pd.DataFrame) -> pd.DataFrame:
        """Normalise run-length sequences.

        Args:
            sequences: Raw sequences with ``date`` and ``span`` (seconds),
                optional ``Values``/``Runs``.

        Returns:
            pandas.DataFrame: Copy with ``start_time``, ``duration_seconds``
            (missing as 0), ``end_time`` and, when ``Values`` exists,
            ``state_value``/``run_count`` renamed columns.

        Raises:
            KeyError: If ``date`` is missing.
        """
        frame = sequences.copy()
        frame["start_time"] = _normalize_timestamp_series(frame["date"])
        frame["duration_seconds"] = pd.to_numeric(frame.get("span"), errors="coerce").fillna(0.0)
        frame["end_time"] = frame["start_time"] + pd.to_timedelta(frame["duration_seconds"], unit="s")
        if "Values" in frame.columns:
            frame = frame.rename(columns={"Values": "state_value", "Runs": "run_count"})
        return frame

    @staticmethod
    def _normalize_words(words: pd.DataFrame) -> pd.DataFrame:
        """Normalise word occurrences.

        Args:
            words: Raw words with ``date`` and ``span`` (seconds).

        Returns:
            pandas.DataFrame: Copy with ``start_time``, ``duration_seconds``
            and ``end_time``.

        Raises:
            KeyError: If ``date`` is missing.
        """
        frame = words.copy()
        frame["start_time"] = _normalize_timestamp_series(frame["date"])
        frame["duration_seconds"] = pd.to_numeric(frame.get("span"), errors="coerce").fillna(0.0)
        frame["end_time"] = frame["start_time"] + pd.to_timedelta(frame["duration_seconds"], unit="s")
        return frame

    @classmethod
    def _with_candidate_schema(cls, frame: pd.DataFrame) -> pd.DataFrame:
        """Project a frame onto the intermediate candidate schema.

        Args:
            frame: Candidate rows.

        Returns:
            pandas.DataFrame: Columns of :meth:`_candidate_columns` in order;
            missing columns are created as ``None`` and extra columns dropped.
        """
        normalized = frame.copy()
        for column in cls._candidate_columns():
            if column not in normalized.columns:
                normalized[column] = None
        return normalized.loc[:, cls._candidate_columns()]

    @staticmethod
    def _candidate_columns() -> list[str]:
        """Return the ordered intermediate candidate columns.

        Returns:
            list[str]: Thirteen column names shared by all heuristics.
        """
        return [
            "source_window_start", "source_window_end", "documented_start", "documented_end", "event_time",
            "incident_family", "source_month", "downtime_start", "downtime_end", "secondary_family", "notes", "source_type", "label_strength",
        ]

    @classmethod
    def _empty_candidate_frame(cls) -> pd.DataFrame:
        """Return an empty frame with the intermediate candidate schema.

        Returns:
            pandas.DataFrame: Empty candidate frame.
        """
        return pd.DataFrame(columns=cls._candidate_columns())

    @staticmethod
    def _empty_final_registry() -> pd.DataFrame:
        """Return an empty frame with the final 20-column registry schema.

        Returns:
            pandas.DataFrame: Empty registry.
        """
        return pd.DataFrame(
            columns=[
                "incident_id", "source_window_start", "source_window_end", "documented_start", "documented_end",
                "event_time", "event_time_precision", "incident_family", "family", "secondary_family",
                "recovery_time", "recovery_status", "label_strength", "source_type", "downtime_start",
                "downtime_end", "maintenance_time", "affected_subsystem", "notes", "asset_id",
            ]
        )


def _normalize_timestamp_series(series: pd.Series) -> pd.Series:
    """Parse timestamps as UTC and drop the timezone.

    Args:
        series: Raw timestamp values (strings, datetimes, tz-aware or naive;
            naive values are interpreted as UTC).

    Returns:
        pandas.Series: Naive UTC timestamps; unparsable values become ``NaT``.
    """
    timestamps = pd.to_datetime(series, utc=True, errors="coerce")
    return timestamps.dt.tz_convert(None)
