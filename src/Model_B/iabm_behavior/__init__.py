"""Public package surface for Model_B behavioral sequence analysis.

Model_B is the second layer of the multi-scale operational-evidence pipeline
(Model_A state inference -> **Model_B behavioural sequences** -> Model_C
semantic modes -> Model_D incident evidence/episodes). It consumes a
time-indexed timeline of integer (bitmask-encoded) operating states and turns
it into contiguous runs, active sequences ("state words"), nominal references
and anomaly/longitudinal/recovery metrics.

The package intentionally exposes a compact API centered on sequence-oriented
building blocks so downstream layers can depend on explicit typed dataclasses
instead of loosely structured intermediate dictionaries.

Exported objects:

* :class:`~iabm_behavior.sequences.BehavioralSequenceAnalyzer` -- stateless
  analysis engine (loading, smoothing, run/sequence extraction, comparison).
* :class:`~iabm_behavior.sequences.StateRun`,
  :class:`~iabm_behavior.sequences.ActiveSequence`,
  :class:`~iabm_behavior.sequences.NominalSequenceReference`,
  :class:`~iabm_behavior.sequences.SequenceComparison`,
  :class:`~iabm_behavior.sequences.LongitudinalSequenceMetrics`,
  :class:`~iabm_behavior.sequences.RecoveryMetrics` -- frozen dataclasses
  describing each abstraction level.

Examples:
    >>> from iabm_behavior import BehavioralSequenceAnalyzer
    >>> analyzer = BehavioralSequenceAnalyzer(state_column="Predicted_State")

Maintenance notes:
    Anything added to ``__all__`` becomes part of the public contract consumed
    by Model_C, Model_D and the repository batch utilities. The CLI
    (:mod:`iabm_behavior.main`) and i18n helpers (:mod:`iabm_behavior.utils`)
    are deliberately not re-exported.
"""

from .sequences import (
    ActiveSequence,
    BehavioralSequenceAnalyzer,
    LongitudinalSequenceMetrics,
    NominalSequenceReference,
    RecoveryMetrics,
    SequenceComparison,
    StateRun,
)

__all__ = [
    "ActiveSequence",
    "BehavioralSequenceAnalyzer",
    "LongitudinalSequenceMetrics",
    "NominalSequenceReference",
    "RecoveryMetrics",
    "SequenceComparison",
    "StateRun",
]
