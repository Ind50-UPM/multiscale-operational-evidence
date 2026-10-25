"""Typed configuration objects for the Model_D incident-evidence stage.

Role in the pipeline:
    Model_D is the last stage of the multi-scale operational-evidence chain
    (Model_A state inference -> Model_B behavioural sequences -> Model_C
    semantic modes -> Model_D incident evidence). It combines rolling-window
    construction, deviation scoring, onset/recovery segmentation into
    episodes, explainable family assignment and occurrence summarization.
    This module centralizes the configuration contract so that command-line
    entry points, experiments (ablation, conformal fusion, sensitivity
    analysis) and tests share one immutable, typed settings surface.

Main public objects:
    * :class:`DeviationWeights` -- per-indicator aggregation weights.
    * :class:`WindowConfig` -- rolling-window length, step and support policy.
    * :class:`EpisodeDetectionConfig` -- onset/recovery hysteresis policy.
    * :class:`FamilyAssignmentConfig` -- rule-based family assignment policy.
    * :class:`OccurrenceConfig` -- occurrence-model fitting policy.
    * :class:`ModelDConfig` -- the aggregate of all of the above.
    * :func:`load_model_d_config` -- JSON loader with backward-compatible keys.

Data contract (JSON file accepted by :func:`load_model_d_config`):
    A single JSON object with optional top-level sections ``window``,
    ``detection`` (or the legacy ``episode_detection``), ``weights``,
    ``family_assignment`` and ``occurrence``. Durations in the JSON file are
    expressed in **hours** (keys suffixed ``_hours``) and hydrated into
    :class:`pandas.Timedelta`; missing keys fall back to loader defaults.

.. mermaid::

   classDiagram
       class ModelDConfig {
           +WindowConfig window
           +EpisodeDetectionConfig detection
           +DeviationWeights weights
           +FamilyAssignmentConfig family_assignment
           +OccurrenceConfig occurrence
           +to_dict() dict
       }
       ModelDConfig *-- WindowConfig
       ModelDConfig *-- EpisodeDetectionConfig
       ModelDConfig *-- DeviationWeights
       ModelDConfig *-- FamilyAssignmentConfig
       ModelDConfig *-- OccurrenceConfig

Maintenance notes:
    All dataclasses are ``frozen=True``; derive modified copies with
    :func:`dataclasses.replace` (as :mod:`iabm_incidents.ablation` does for
    calibrated thresholds) instead of mutating instances. When a field is
    added, update the dataclass default, the matching ``_load_*`` helper and,
    if serialization needs special handling, :func:`_serialize_dataclass`.
    Note that loader defaults are not always identical to dataclass defaults
    (``minimum_duration`` is 10 minutes in the dataclass but 0.5 hours when
    loaded from a JSON file without ``minimum_duration_hours``); keep this in
    mind when comparing programmatic and file-based runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DeviationWeights:
    r"""Weights used to aggregate window-level deviation indicators.

    The composite window score computed by
    :func:`~iabm_incidents.detection.compute_deviation_score` is the weighted
    linear sum

    .. math::

       S_w = \sum_{k} \omega_k \, x_{w,k},

    where :math:`x_{w,k}` is indicator :math:`k` of window :math:`w` (missing
    indicators contribute zero) and :math:`\omega_k` is the matching attribute
    of this class. The matched ablation
    (:class:`~iabm_incidents.ablation.FourLevelAblationRunner`) applies the
    same weights to robustly scaled directional excesses and requires them
    to be finite and non-negative. All weights are dimensionless.

    Attributes:
        sequence: Weight assigned to sequence-divergence evidence.
        duration: Weight assigned to duration-drift evidence.
        recurrence: Weight assigned to recurrence-based evidence.
        persistence: Weight assigned to persistence-based evidence.
        consumption: Weight assigned to analogue consumption deviation.
        state_error: Weight assigned to semantic or state error signals.
        mode: Weight assigned to operating-mode divergence.
        command_response: Weight assigned to command-response residual norms.
        transient_concentration: Weight assigned to fourth-order excess
            evidence (exact Tconc; the ablation reuses it for the VQE
            approximation of the same coordinate).
    """

    sequence: float = 1.0
    duration: float = 1.0
    recurrence: float = 1.0
    persistence: float = 1.0
    consumption: float = 1.0
    state_error: float = 1.0
    mode: float = 1.0
    command_response: float = 1.0
    transient_concentration: float = 1.0


@dataclass(frozen=True)
class WindowConfig:
    """Definition of the rolling analysis window policy.

    Windows are half-open intervals ``[start, start + length)`` whose starts
    advance by ``step`` from the first start time floored to ``step``. A
    window is emitted only when it satisfies both support constraints.

    Attributes:
        length: Window length (default 24 h).
        step: Stride between consecutive window starts (default 1 h). It is
            also used downstream as the decision-interval length attributed
            to each window (exposure accounting) and as the maximum start
            spacing that still counts as contiguous.
        min_coverage: Minimum fraction in ``[0, 1]`` of the window length
            that must be covered by overlapping source rows (sum of overlap
            durations divided by ``length``, capped at 1). Also used by the
            ablation as the minimum ``data_coverage`` for common support.
        min_active_sequences: Minimum number of source rows with strictly
            positive overlap required for the window to be emitted.
    """

    length: pd.Timedelta = pd.Timedelta(hours=24)
    step: pd.Timedelta = pd.Timedelta(hours=1)
    min_coverage: float = 0.90
    min_active_sequences: int = 1


@dataclass(frozen=True)
class EpisodeDetectionConfig:
    """Parameters governing onset persistence and recovery segmentation.

    The episode builder (:mod:`iabm_incidents.episodes`) applies a
    hysteresis policy on the window ``deviation_score``: an episode opens
    after ``onset_windows`` consecutive windows at or above
    ``onset_threshold`` and is considered recovered after
    ``recovery_windows`` consecutive windows below ``recovery_threshold``.

    Attributes:
        onset_threshold: Score level that starts an onset run (score units).
        recovery_threshold: Score level below which recovery is counted;
            should be lower than ``onset_threshold``.
        onset_windows: Number of consecutive windows needed to confirm onset.
        recovery_windows: Number of consecutive windows needed to confirm a
            stable recovery.
        minimum_duration: Shortest episode retained.
        maximum_gap: Largest gap that may be bridged inside one episode.
        maximum_duration: Configured upper bound on episode duration; the
            ablation reports how many episodes exceed it.
    """

    onset_threshold: float = 1.0
    recovery_threshold: float = 0.5
    onset_windows: int = 2
    recovery_windows: int = 3
    minimum_duration: pd.Timedelta = pd.Timedelta(minutes=10)
    maximum_gap: pd.Timedelta = pd.Timedelta(hours=2)
    maximum_duration: pd.Timedelta = pd.Timedelta(days=7)


@dataclass(frozen=True)
class FamilyAssignmentConfig:
    """Configuration for the explainable family-assignment stage.

    Attributes:
        method: Label recorded in ``assignment_method`` of each
            :class:`~iabm_incidents.classification.FamilyAssignment`. It is
            informational only; the classifier does not branch on it.
        minimum_confidence: Minimum family score in ``[0, 1]`` required for
            a primary family; below it the episode is labelled
            ``"unclassified_incident"``.
    """

    method: str = "rule_based"
    minimum_confidence: float = 0.60


@dataclass(frozen=True)
class OccurrenceConfig:
    """Configuration for family-level occurrence modelling outputs.

    Attributes:
        minimum_events_for_fit: Minimum number of episodes of a family
            required before an occurrence model is fitted for it.
    """

    minimum_events_for_fit: int = 5


@dataclass(frozen=True)
class ModelDConfig:
    """Top-level configuration for the Model_D pipeline.

    Aggregates the stage-specific configuration objects. Instances are
    immutable; use :func:`dataclasses.replace` to derive variants.

    Examples:
        >>> from dataclasses import replace
        >>> cfg = ModelDConfig()
        >>> tuned = replace(cfg, detection=replace(cfg.detection, onset_threshold=2.0))
        >>> tuned.detection.onset_threshold
        2.0

    Attributes:
        window: Rolling-window construction policy.
        detection: Episode-onset and recovery segmentation policy.
        weights: Aggregation weights for deviation indicators.
        family_assignment: Rule-based family-assignment settings.
        occurrence: Occurrence-summary settings.
    """

    window: WindowConfig = WindowConfig()
    detection: EpisodeDetectionConfig = EpisodeDetectionConfig()
    weights: DeviationWeights = DeviationWeights()
    family_assignment: FamilyAssignmentConfig = FamilyAssignmentConfig()
    occurrence: OccurrenceConfig = OccurrenceConfig()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration into JSON-friendly values.

        Returns:
            Nested dictionary with one key per section (``window``,
            ``detection``, ``weights``, ``family_assignment``,
            ``occurrence``) ready to be written into metadata artefacts.

        Notes:
            Timedelta objects are converted to ISO-8601 duration strings by
            :func:`_serialize_dataclass` so metadata remains portable across
            command-line runs and notebooks. The output is not directly
            re-loadable by :func:`load_model_d_config`, which expects
            ``*_hours`` numeric keys.
        """
        return {
            "window": _serialize_dataclass(self.window),
            "detection": _serialize_dataclass(self.detection),
            "weights": _serialize_dataclass(self.weights),
            "family_assignment": _serialize_dataclass(self.family_assignment),
            "occurrence": _serialize_dataclass(self.occurrence),
        }


def load_model_d_config(file_path: str | Path | None) -> ModelDConfig:
    """Load a Model_D configuration file when provided.

    Args:
        file_path: Optional path to a UTF-8 JSON configuration file. ``None``
            returns the programmatic defaults of :class:`ModelDConfig`.

    Returns:
        A fully typed :class:`ModelDConfig` instance.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        json.JSONDecodeError: If the file is not valid JSON.

    Notes:
        The loader keeps backward compatibility with both ``detection`` and the
        older ``episode_detection`` key so historical configuration files remain
        valid after refactors. When both are present, ``episode_detection``
        takes precedence. Missing sections are hydrated from empty
        dictionaries, i.e. from the loader defaults of each ``_load_*``
        helper.

    See Also:
        :meth:`ModelDConfig.to_dict` for the (non-round-trip) serialization.
    """
    if file_path is None:
        return ModelDConfig()

    path = Path(file_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ModelDConfig(
        window=_load_window_config(payload.get("window", {})),
        detection=_load_detection_config(payload.get("episode_detection", payload.get("detection", {}))),
        weights=_load_weights_config(payload.get("weights", {})),
        family_assignment=_load_family_assignment_config(payload.get("family_assignment", {})),
        occurrence=_load_occurrence_config(payload.get("occurrence", {})),
    )


def _load_window_config(payload: dict[str, Any]) -> WindowConfig:
    """Hydrate :class:`WindowConfig` from a JSON payload fragment.

    Args:
        payload: Mapping with optional keys ``length_hours`` (default 24),
            ``step_hours`` (default 1), ``min_coverage`` (default 0.90) and
            ``min_active_sequences`` (default 1).

    Returns:
        The hydrated :class:`WindowConfig`.
    """
    return WindowConfig(
        length=pd.Timedelta(hours=payload.get("length_hours", 24)),
        step=pd.Timedelta(hours=payload.get("step_hours", 1)),
        min_coverage=float(payload.get("min_coverage", 0.90)),
        min_active_sequences=int(payload.get("min_active_sequences", 1)),
    )


def _load_detection_config(payload: dict[str, Any]) -> EpisodeDetectionConfig:
    """Hydrate :class:`EpisodeDetectionConfig` from a JSON payload fragment.

    Args:
        payload: Mapping with optional keys ``onset_threshold`` (legacy alias
            ``onset_quantile``, default 1.0), ``recovery_threshold`` (legacy
            alias ``recovery_quantile``, default 0.5), ``onset_windows``
            (default 2), ``recovery_windows`` (default 3),
            ``minimum_duration_hours`` (default 0.5), ``maximum_gap_hours``
            (default 2) and ``maximum_duration_hours`` (default 168).

    Returns:
        The hydrated :class:`EpisodeDetectionConfig`.

    Notes:
        The legacy ``*_quantile`` aliases are read as absolute score
        thresholds, not as quantile levels; no quantile is computed here.
    """
    return EpisodeDetectionConfig(
        onset_threshold=float(payload.get("onset_threshold", payload.get("onset_quantile", 1.0))),
        recovery_threshold=float(payload.get("recovery_threshold", payload.get("recovery_quantile", 0.5))),
        onset_windows=int(payload.get("onset_windows", 2)),
        recovery_windows=int(payload.get("recovery_windows", 3)),
        minimum_duration=pd.Timedelta(hours=payload.get("minimum_duration_hours", 0.5)),
        maximum_gap=pd.Timedelta(hours=payload.get("maximum_gap_hours", 2)),
        maximum_duration=pd.Timedelta(hours=payload.get("maximum_duration_hours", 24 * 7)),
    )


def _load_weights_config(payload: dict[str, Any]) -> DeviationWeights:
    """Hydrate :class:`DeviationWeights` from a JSON payload fragment.

    Args:
        payload: Mapping from weight name (``sequence``, ``duration``,
            ``recurrence``, ``persistence``, ``consumption``,
            ``state_error``, ``mode``, ``command_response``,
            ``transient_concentration``) to a numeric weight; each missing
            key defaults to 1.0.

    Returns:
        The hydrated :class:`DeviationWeights`.
    """
    return DeviationWeights(
        sequence=float(payload.get("sequence", 1.0)),
        duration=float(payload.get("duration", 1.0)),
        recurrence=float(payload.get("recurrence", 1.0)),
        persistence=float(payload.get("persistence", 1.0)),
        consumption=float(payload.get("consumption", 1.0)),
        state_error=float(payload.get("state_error", 1.0)),
        mode=float(payload.get("mode", 1.0)),
        command_response=float(payload.get("command_response", 1.0)),
        transient_concentration=float(payload.get("transient_concentration", 1.0)),
    )


def _load_family_assignment_config(payload: dict[str, Any]) -> FamilyAssignmentConfig:
    """Hydrate :class:`FamilyAssignmentConfig` from a JSON payload fragment.

    Args:
        payload: Mapping with optional keys ``method`` (default
            ``"rule_based"``) and ``minimum_confidence`` (default 0.60).

    Returns:
        The hydrated :class:`FamilyAssignmentConfig`.
    """
    return FamilyAssignmentConfig(
        method=str(payload.get("method", "rule_based")),
        minimum_confidence=float(payload.get("minimum_confidence", 0.60)),
    )


def _load_occurrence_config(payload: dict[str, Any]) -> OccurrenceConfig:
    """Hydrate :class:`OccurrenceConfig` from a JSON payload fragment.

    Args:
        payload: Mapping with optional key ``minimum_events_for_fit``
            (default 5).

    Returns:
        The hydrated :class:`OccurrenceConfig`.
    """
    return OccurrenceConfig(
        minimum_events_for_fit=int(payload.get("minimum_events_for_fit", 5)),
    )


def _serialize_dataclass(instance: Any) -> dict[str, Any]:
    """Convert dataclass values into JSON-friendly primitives.

    Args:
        instance: Flat dataclass instance to serialize (one of the section
            dataclasses of this module).

    Returns:
        Dictionary mapping field names to values, where
        :class:`pandas.Timedelta` values are replaced by their ISO-8601
        string (``Timedelta.isoformat()``) and all other values are passed
        through unchanged.

    Notes:
        Only top-level :class:`pandas.Timedelta` fields are converted;
        nested dataclasses would be converted to plain dictionaries by
        :func:`dataclasses.asdict` but their timedeltas would not be
        stringified.
    """
    serialized: dict[str, Any] = {}
    for key, value in asdict(instance).items():
        if isinstance(value, pd.Timedelta):
            serialized[key] = value.isoformat()
        else:
            serialized[key] = value
    return serialized
