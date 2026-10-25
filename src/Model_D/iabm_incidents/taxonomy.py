"""Incident-family taxonomy primitives for Model_D.

Role in the pipeline:
    The taxonomy is the static vocabulary shared by family assignment
    (:mod:`~iabm_incidents.classification`), the heuristic candidate registry
    (:mod:`~iabm_incidents.registry_builder`), risk targets
    (:class:`~iabm_incidents.ablation.AblationConfig`) and reporting. It
    declares five incident families together with qualitative severity, life
    regime and recovery regime, and a feature-level signature consumed by the
    explainable signature scorer.

Main public objects:
    * :class:`FamilySignature` - required, preferred and incompatible
      symbolic features of one family.
    * :class:`IncidentFamilyDefinition` - semantic/operational description.
    * :data:`DEFAULT_INCIDENT_TAXONOMY` - ``family -> definition`` mapping for
      ``pump_abrupt_failure``, ``float_recurrent_disturbance``,
      ``process_saturation``, ``post_intervention_recovery`` and
      ``external_ambiguous_disturbance``.
    * :data:`FAMILY_SIGNATURES` - ``family -> signature`` view derived from
      the taxonomy.
    * :func:`is_known_family` - membership test.

Notes:
    In :class:`~iabm_incidents.classification` a signature contributes
    ``0.45`` when all required features are present (``0.20`` times the
    satisfied fraction otherwise), up to ``0.35`` proportionally to the
    satisfied preferred features, and ``-0.25`` when any incompatible feature
    is present; the result is clipped to ``[0, 1]``.

Maintenance notes:
    * Families are *operational evidence categories*, not confirmed physical
      failure modes.
    * ``minimum_assignment_score`` is declared per family but, at the time of
      writing, no module in the package reads it; confidence gating uses
      :class:`~iabm_incidents.config.FamilyAssignmentConfig` instead.
    * Feature names are symbolic strings interpreted by
      ``classification._feature_present``; renaming a feature requires
      updating both places.
    * The dictionaries are module-level and mutable; treat them as read-only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FamilySignature:
    """Formal feature-level expectations used by family assigners.

    Attributes:
        family: Family identifier (key of :data:`DEFAULT_INCIDENT_TAXONOMY`).
        required_features: Features that must all be present for full
            required credit.
        preferred_features: Supporting features; each adds partial credit.
        incompatible_features: Features that penalise the family when present.
        minimum_assignment_score: Declared minimum score in ``[0, 1]`` for
            assignment (see the module maintenance notes).
    """

    family: str
    required_features: tuple[str, ...] = tuple()
    preferred_features: tuple[str, ...] = tuple()
    incompatible_features: tuple[str, ...] = tuple()
    minimum_assignment_score: float = 0.70


@dataclass(frozen=True)
class IncidentFamilyDefinition:
    """Semantic and operational description of one incident family.

    Attributes:
        family: Family identifier.
        severity: Qualitative severity (``"high"``, ``"medium"``, ``"low"``
            or ``"unknown"``).
        life_regime: Asset life regime associated with the family (e.g.
            ``"degradation_acceleration"``).
        recovery_regime: Expected recovery behaviour (e.g.
            ``"needs_intervention"``).
        signature: Feature-level signature.
    """

    family: str
    severity: str
    life_regime: str
    recovery_regime: str
    signature: FamilySignature

    def to_dict(self) -> dict[str, object]:
        """Serialize the definition, including its nested signature.

        Returns:
            dict[str, object]: JSON-friendly mapping of all fields; tuples of
            features remain tuples.
        """
        payload = asdict(self)
        payload["signature"] = asdict(self.signature)
        return payload


#: dict[str, IncidentFamilyDefinition]: Default five-family taxonomy, keyed
#: by family identifier.
DEFAULT_INCIDENT_TAXONOMY = {
    "pump_abrupt_failure": IncidentFamilyDefinition(
        family="pump_abrupt_failure",
        severity="high",
        life_regime="degradation_acceleration",
        recovery_regime="needs_intervention",
        signature=FamilySignature(
            family="pump_abrupt_failure",
            required_features=("abrupt_onset", "negative_consumption_deviation"),
            preferred_features=(
                "digital_electrical_mismatch",
                "high_sequence_divergence",
                "high_state_disagreement",
                "stable_word_regime",
                "far_from_nominal_word",
                "rare_state_presence",
            ),
            minimum_assignment_score=0.70,
        ),
    ),
    "float_recurrent_disturbance": IncidentFamilyDefinition(
        family="float_recurrent_disturbance",
        severity="medium",
        life_regime="recurrent_instability",
        recovery_regime="under_observation",
        signature=FamilySignature(
            family="float_recurrent_disturbance",
            required_features=("recurrence_excess",),
            preferred_features=("persistence_excess", "sequence_divergence", "state_stability", "word_regime_instability", "far_from_nominal_word", "word_distribution_shift"),
            minimum_assignment_score=0.60,
        ),
    ),
    "process_saturation": IncidentFamilyDefinition(
        family="process_saturation",
        severity="medium",
        life_regime="stress_accumulation",
        recovery_regime="under_observation",
        signature=FamilySignature(
            family="process_saturation",
            required_features=("duration_drift", "persistence_excess"),
            preferred_features=("mode_divergence", "non_negative_consumption_deviation", "persistent_word_regime", "near_nominal_word_with_duration_drift", "rare_word_pressure_without_state_shift"),
            minimum_assignment_score=0.60,
        ),
    ),
    "post_intervention_recovery": IncidentFamilyDefinition(
        family="post_intervention_recovery",
        severity="low",
        life_regime="nominal_operation",
        recovery_regime="stable_recovered",
        signature=FamilySignature(
            family="post_intervention_recovery",
            preferred_features=("moderate_residual_deviation", "restored_state_agreement"),
            incompatible_features=("abrupt_onset",),
            minimum_assignment_score=0.55,
        ),
    ),
    "external_ambiguous_disturbance": IncidentFamilyDefinition(
        family="external_ambiguous_disturbance",
        severity="unknown",
        life_regime="externally_perturbed_operation",
        recovery_regime="under_observation",
        signature=FamilySignature(
            family="external_ambiguous_disturbance",
            preferred_features=("mixed_family_evidence", "cross_domain_deviation", "word_distribution_shift"),
            minimum_assignment_score=0.50,
        ),
    ),
}


#: dict[str, FamilySignature]: Signatures keyed by family, in taxonomy order.
FAMILY_SIGNATURES = {
    family: definition.signature for family, definition in DEFAULT_INCIDENT_TAXONOMY.items()
}


def is_known_family(family: str) -> bool:
    """Return whether a family identifier is registered in the taxonomy.

    Args:
        family: Candidate family identifier.

    Returns:
        bool: ``True`` if ``family`` is a key of
        :data:`DEFAULT_INCIDENT_TAXONOMY`.

    Examples:
        >>> is_known_family("process_saturation")
        True
        >>> is_known_family("unknown_family")
        False
    """

    return family in DEFAULT_INCIDENT_TAXONOMY
