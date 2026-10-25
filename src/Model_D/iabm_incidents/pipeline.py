"""Indicator-driven orchestration for Model_D.

This module contains the first fully materialized implementation of the
indicator-based route proposed for ``Model_D``. The pipeline starts from
state sequences and semantic assignments, derives observation exposure,
constructs rolling windows, computes deviation indicators, scores each window,
segments anomalous periods into candidate episodes, and enriches those episodes
with aggregate descriptive features.

The implementation intentionally remains lightweight and dataframe-oriented so
that it can be executed in batch mode over historical campaigns without
introducing external workflow dependencies. Every stage returns explicit
tables, which makes the module convenient for exploratory notebooks, CLI batch
processing, and Sphinx-generated API documentation.

Role in the pipeline:
    This is the ``--detection-mode indicators`` route of the CLI when raw
    (not pre-windowed) inputs are supplied. It consumes Model_B sequences and
    Model_C semantic assignments, optionally raw analogue/digital telemetry
    and frozen electrical models, and emits the Model_D window and episode
    tables.

Main public objects:
    * :class:`IndicatorPipeline` - stage orchestrator.
    * :class:`IndicatorPipelineResult` - immutable bundle of stage outputs.

.. mermaid::

   flowchart TD
       S[sequences / analogue / digital] --> E[derive_observation_periods]
       R[registry] --> E
       E --> W[build_rolling_windows]
       S --> M[_merge_sources]
       A[assignments] --> M
       M --> B[fit_nominal_baseline]
       M --> I[compute_window_indicators]
       W --> I
       B --> I
       I --> X{frozen electrical models?}
       X -- yes --> C[CommandResponseObserver.residuals + TransientConcentration.score_windows]
       X -- no --> SC
       C --> SC[WeightedDeviationScorer.score]
       SC --> EP[IncidentEpisodeBuilder.build_episodes_from_window_scores]
       EP --> F[build_episode_features]
       F --> O[IndicatorPipelineResult]
       F -. opt-in .-> P[EventFrequencyResponseInterpreter.explain]
       P -.-> O

Data contracts:
    * ``window_scores`` carries ``window_id``, ``asset_id``, window bounds,
      the seven behavioural indicators (``sequence_divergence``,
      ``duration_drift``, ``recurrence_excess``, ``persistence_excess``,
      ``consumption_deviation``, ``state_error_rate``, ``mode_divergence``),
      optionally ``command_response_deviation``, ``transient_concentration``
      (Tconc) and electrical support columns, and the final
      ``deviation_score``.
    * Electrical columns are merged on ``['window_id', 'asset_id']`` with a
      one-to-one validation; electrical columns replace same-named indicator
      columns.

Maintenance notes:
    * **Leakage guard:** the command observer and transient reference are
      fitted by the caller on declared nominal partitions and are never
      refitted inside :meth:`IndicatorPipeline.run`.
    * The nominal baseline is passed to
      :func:`~iabm_incidents.indicators.compute_window_indicators` through an
      ad-hoc object exposing only ``medians``; keep that duck-typed contract
      in sync with the indicators module.
    * :func:`~iabm_incidents.indicators.compute_window_indicators` always
      emits ``semantic_status``. The ``.get(..., pd.Series(dtype=str))``
      fallback in :meth:`IndicatorPipeline.run` would yield an empty,
      unalignable mask (pandas ``IndexingError``) if a future indicator
      source dropped that column.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import ModelDConfig
from .electrical import CommandResponseObserver, TransientConcentration
from .episodes import IncidentEpisodeBuilder
from .interpretation import (EventFrequencyResponseInterpreter, EventFrequencyInterpretationResult,
                             FrequencyInterpretationConfig)
from .exposure import ExposureConfig, derive_observation_periods
from .baseline import fit_nominal_baseline
from .features import build_episode_features
from .indicators import compute_window_indicators
from .scoring import WeightedDeviationScorer
from .windows import WindowBuildConfig, build_rolling_windows


@dataclass(frozen=True)
class IndicatorPipelineResult:
    """Materialized outputs of the indicator-driven pipeline.

    Attributes:
        observation_periods: Exposure-aware availability table describing the
            intervals where the asset is considered observed and therefore
            eligible for baseline estimation and episode detection.
        windows: Rolling analysis windows built from the observation periods.
        window_scores: Window-level indicator table with raw features and the
            final aggregate deviation score.
        episodes: Incident episodes segmented from scored windows and enriched
            with aggregate episode-level descriptors when available.
        event_interpretation: Optional separate post hoc FIR/Bode artifacts.
            These never enter window scores or change episode detection.
            ``None`` unless an ``interpretation_config`` was supplied.

    Notes:
        When no window can be scored, ``window_scores`` is the empty indicator
        table and ``episodes`` is an empty :class:`pandas.DataFrame` without
        columns.
    """

    observation_periods: pd.DataFrame
    windows: pd.DataFrame
    window_scores: pd.DataFrame
    episodes: pd.DataFrame
    event_interpretation: EventFrequencyInterpretationResult | None = None


class IndicatorPipeline:
    """Build indicator-driven episode outputs from source tables.

    Args:
        config: Full ``Model_D`` configuration. When omitted, the default
            in-package configuration is used.
        command_observer: Optional fitted nominal command-response observer.
        transient_model: Optional fitted fourth-order reference, provided with
            the observer. Both remain frozen during ``run``.
        interpretation_config: Opt-in event-conditioned frequency-response
            interpretation, executed after detection. Omit to avoid post hoc work.

    Notes:
        The class is intentionally state-light. It only keeps the resolved
        configuration plus the reusable scorer and episode builder so that
        callers can instantiate it once and execute it for many monthly or
        asset-specific batches.

    Raises:
        ValueError: If exactly one of ``command_observer`` and
            ``transient_model`` is given, if their ``config`` objects differ,
            or if ``interpretation_config`` is given without electrical
            models.

    Examples:
        Behavioural-only route (no electrical models)::

            pipeline = IndicatorPipeline(load_model_d_config("configs/model_d.example.json"))
            result = pipeline.run(sequences=sequences, assignments=assignments)
            result.episodes.head()
    """

    def __init__(
        self, config: ModelDConfig | None = None, *,
        command_observer: CommandResponseObserver | None = None,
        transient_model: TransientConcentration | None = None,
        interpretation_config: FrequencyInterpretationConfig | None = None,
    ) -> None:
        """Validate the electrical-model pairing and build reusable stages.

        Args:
            config: Model_D configuration; ``None`` uses
                :class:`~iabm_incidents.config.ModelDConfig` defaults.
            command_observer: Frozen nominal command-response observer.
            transient_model: Frozen fourth-order transient reference.
            interpretation_config: Opt-in post hoc interpretation settings.

        Raises:
            ValueError: See the class docstring.
        """
        if (command_observer is None) != (transient_model is None):
            raise ValueError("Provide both frozen electrical models or neither")
        if command_observer is not None and command_observer.config != transient_model.config:
            raise ValueError("Electrical models must share the telemetry contract")
        if interpretation_config is not None and transient_model is None:
            raise ValueError("Event interpretation requires frozen electrical models")
        self._interpretation_config = interpretation_config
        self._command_observer = command_observer
        self._transient_model = transient_model
        self._config = config or ModelDConfig()
        self._builder = IncidentEpisodeBuilder(self._config)
        self._scorer = WeightedDeviationScorer(self._config.weights)

    def run(
        self,
        *,
        sequences: pd.DataFrame,
        assignments: pd.DataFrame,
        registry: pd.DataFrame | None = None,
        analogue: pd.DataFrame | None = None,
        digital: pd.DataFrame | None = None,
    ) -> IndicatorPipelineResult:
        """Execute the end-to-end indicator pipeline.

        Args:
            sequences: Sequence-level table, typically produced by ``Model_B``,
                containing temporal state information.
            assignments: Semantic assignment table, typically produced by
                ``Model_C``, aligned with the sequence rows or the derived
                analysis windows.
            registry: Optional canonical incident registry used to mark exposure
                periods that should be excluded from nominal baseline fitting.
            analogue: When electrical models are supplied, aligned timestamped
                telemetry containing three phases and the actual controller state
                according to ``ElectricalConfig``. Otherwise optional analog
                telemetry. When omitted, the method
                falls back to the ``sequences`` dataframe as the observable time
                base.
            digital: Optional digital telemetry table. When omitted, the method
                falls back to the ``sequences`` dataframe as the observable time
                base.

        Returns:
            A fully materialized :class:`IndicatorPipelineResult`.

        Raises:
            ValueError: If electrical models are configured but ``analogue``
                is ``None`` (raised only when at least one window was scored).

        Notes:
            Stage order: exposure -> rolling windows -> merged sources ->
            nominal baseline -> window indicators -> (optional) electrical
            evidence -> weighted ``deviation_score`` -> episode segmentation
            -> episode features -> (optional) frequency interpretation. The
            exposure gap tolerance equals ``config.detection.maximum_gap``.
        """

        analogue_frame = analogue if analogue is not None else sequences
        digital_frame = digital if digital is not None else sequences
        observation_periods = derive_observation_periods(
            analogue_frame,
            digital_frame,
            states=sequences,
            registry=registry,
            config=ExposureConfig(gap_tolerance=self._config.detection.maximum_gap),
        )
        windows = build_rolling_windows(
            observation_periods,
            config=WindowBuildConfig(
                length=self._config.window.length,
                step=self._config.window.step,
                min_coverage=self._config.window.min_coverage,
            ),
        )

        source_rows = self._merge_sources(sequences, assignments)
        baseline = fit_nominal_baseline(source_rows)
        # Why: compute_window_indicators only needs ``.medians``; a throwaway
        # class adapts the baseline object to that duck-typed contract.
        window_scores = compute_window_indicators(
            source_rows,
            windows,
            None if baseline is None else type('_Baseline', (), {'medians': baseline.medians()})(),
        )
        if window_scores.empty:
            return IndicatorPipelineResult(observation_periods, windows, window_scores, pd.DataFrame())
        if self._transient_model is not None:
            if analogue is None:
                raise ValueError("Electrical extension requires aligned analogue command telemetry")
            # Invariant: observer and reference are frozen; nothing is refitted.
            residuals = self._command_observer.residuals(analogue)
            electrical = self._transient_model.score_windows(residuals, window_scores)
            keys = ['window_id', 'asset_id']
            replace = [c for c in electrical if c in window_scores and c not in keys]
            window_scores = window_scores.drop(columns=replace).merge(electrical, on=keys, how='left', validate='one_to_one')
        normal_mask = window_scores.get('semantic_status', pd.Series(dtype=str)).ne('ANOMALOUS')
        self._scorer.fit(window_scores[normal_mask])
        window_scores = window_scores.copy()
        window_scores['deviation_score'] = self._scorer.score(window_scores)
        episodes = self._builder.build_episodes_from_window_scores(window_scores)
        episode_features = build_episode_features(episodes, window_scores)
        if not episode_features.empty and not episodes.empty:
            episodes = episodes.merge(episode_features, on=['episode_id', 'asset_id'], how='left')
        interpretation = None
        # Note: ``residuals`` is always bound here because the constructor
        # rejects an interpretation config without electrical models.
        if self._interpretation_config is not None:
            interpretation = EventFrequencyResponseInterpreter(
                self._transient_model, self._interpretation_config,
            ).explain(episodes=episodes, window_scores=window_scores, residuals=residuals)
        return IndicatorPipelineResult(observation_periods, windows, window_scores, episodes, interpretation)

    @staticmethod
    def _merge_sources(sequences: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
        """Create a unified row-wise source table from sequences and assignments.

        Both inputs are reset to a positional index and concatenated column-wise,
        so row ``i`` of ``assignments`` is assumed to describe row ``i`` of
        ``sequences`` (no key-based join).

        Args:
            sequences: Model_B sequence table.
            assignments: Model_C assignment table. ``window_start`` /
                ``window_end`` are aliased to ``start_time`` / ``end_time``;
                missing ``start_time``, ``end_time`` and ``asset_id`` are copied
                from ``sequences``.

        Returns:
            pandas.DataFrame: Sequence columns followed by assignment columns.
            Assignment columns that duplicate sequence columns are dropped,
            except ``semantic_status``, ``incident_family`` and
            ``anomaly_score``, which may therefore appear twice if both
            inputs carry them.
        """

        sequence_frame = sequences.copy().reset_index(drop=True)
        assignment_frame = assignments.copy().reset_index(drop=True)

        # Backfill compatibility aliases on the assignment side before merging.
        if 'window_start' in assignment_frame.columns and 'start_time' not in assignment_frame.columns:
            assignment_frame['start_time'] = assignment_frame['window_start']
        if 'window_end' in assignment_frame.columns and 'end_time' not in assignment_frame.columns:
            assignment_frame['end_time'] = assignment_frame['window_end']
        if 'start_time' not in assignment_frame.columns and 'start_time' in sequence_frame.columns:
            assignment_frame['start_time'] = sequence_frame['start_time']
        if 'end_time' not in assignment_frame.columns and 'end_time' in sequence_frame.columns:
            assignment_frame['end_time'] = sequence_frame['end_time']
        if 'asset_id' not in assignment_frame.columns and 'asset_id' in sequence_frame.columns:
            assignment_frame['asset_id'] = sequence_frame['asset_id']

        # Avoid duplicate column labels, which later break pandas datetime assembly.
        overlapping = [column for column in assignment_frame.columns if column in sequence_frame.columns]
        protected = {'semantic_status', 'incident_family', 'anomaly_score'}
        drop_from_assignment = [column for column in overlapping if column not in protected]
        if drop_from_assignment:
            assignment_frame = assignment_frame.drop(columns=drop_from_assignment)

        return pd.concat([sequence_frame, assignment_frame], axis=1)
