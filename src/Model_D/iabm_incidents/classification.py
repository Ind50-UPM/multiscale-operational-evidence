"""Explainable rule-based incident-family classification.

Role in the pipeline:
    After :mod:`iabm_incidents.episodes` segments scored windows into
    episodes, each episode receives a primary incident family, ranked
    secondary families, a confidence in ``[0, 1]`` and human-readable
    evidence strings. The classifier is deterministic and transparent: every
    score is a sum of fixed increments triggered by thresholded episode
    features, so assignments can be audited line by line. The matched
    ablation holds this classifier fixed across levels.

Main public objects:
    * :class:`FamilyScore` -- one candidate family with score and evidence.
    * :class:`FamilyAssignment` -- final assignment of one episode.
    * :class:`RuleBasedFamilyClassifier` -- the scorer.

Data contract:
    ``episode_frame`` holds the windows of one episode (columns as produced
    by :mod:`iabm_incidents.detection`; ``incident_family`` carries the
    Model_C semantic family label per window, ``deviation_score`` the
    composite score). ``episode_features`` is one row of
    :func:`~iabm_incidents.features.build_episode_features` (e.g.
    ``min_consumption_deviation``, ``max_state_error_rate``,
    ``onset_slope``, ``mean_state_word_transition_rate``, ``peak_score``,
    ``mixed_family_evidence``); missing features read as 0.

Scoring flow:
    1. Five explicit family scorers (pump abrupt failure, float recurrent
       disturbance, process saturation, post-intervention recovery,
       external ambiguous disturbance) and one declarative scorer per
       :data:`~iabm_incidents.taxonomy.FAMILY_SIGNATURES` entry.
    2. Scores of the same family are merged by maximum (capped at 1) and
       evidence lists are concatenated without duplicates.
    3. Semantic label frequencies are merged in the same way.
    4. Pump/saturation competition is resolved by episode-level rules.
    5. The best family is kept if its score reaches
       ``FamilyAssignmentConfig.minimum_confidence``.

.. mermaid::

   classDiagram
       class RuleBasedFamilyClassifier {
           +assign(episode_frame, config, episode_features) FamilyAssignment
           -_score_candidates(episode_frame, features) list~FamilyScore~
           -_resolve_competing_families(scores, features) dict
           -_signature_scores(features) list~FamilyScore~
       }
       class FamilyScore {
           +str family
           +float score
           +tuple evidence
       }
       class FamilyAssignment {
           +str primary_family
           +tuple secondary_families
           +float family_confidence
           +str assignment_method
           +tuple evidence
       }
       RuleBasedFamilyClassifier ..> FamilyScore : creates
       RuleBasedFamilyClassifier ..> FamilyAssignment : returns
       RuleBasedFamilyClassifier ..> FamilySignature : reads

Maintenance notes:
    Thresholds and increments are hard-coded heuristics; changing any of
    them changes published family counts, so update tests and changelog
    together. ``FamilySignature.minimum_assignment_score`` is not consulted
    here; only ``FamilyAssignmentConfig.minimum_confidence`` gates the
    primary family. :meth:`RuleBasedFamilyClassifier._feature_present`
    duplicates several thresholds of the explicit scorers (with some
    deliberate differences, e.g. ``far_from_nominal_word`` uses ``or`` while
    the pump scorer uses ``and``); keep them consistent when tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import FamilyAssignmentConfig
from .features import build_episode_features
from .taxonomy import FAMILY_SIGNATURES, FamilySignature


@dataclass(frozen=True)
class FamilyScore:
    """Score of one candidate family for one episode.

    Attributes:
        family: Family identifier.
        score: Heuristic support in ``[0, 1]``.
        evidence: Ordered human-readable reasons that contributed.
    """

    family: str
    score: float
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class FamilyAssignment:
    """Final family assignment of one episode.

    Attributes:
        primary_family: Best family, or ``"unclassified_incident"`` when its
            score is below the configured minimum confidence.
        secondary_families: Other families with positive score, in
            descending score order (populated even when the primary family
            is unclassified).
        family_confidence: Score of the primary family; 0.0 when
            unclassified.
        assignment_method: Copied from ``FamilyAssignmentConfig.method``.
        evidence: Evidence of the primary family; empty when unclassified.
    """

    primary_family: str
    secondary_families: tuple[str, ...]
    family_confidence: float
    assignment_method: str
    evidence: tuple[str, ...]


class RuleBasedFamilyClassifier:
    """Assign families using semantic labels, explicit evidence rules, and declarative signatures.

    The classifier is stateless; one instance can be reused across episodes.
    """

    def assign(
        self,
        episode_frame: pd.DataFrame,
        config: FamilyAssignmentConfig,
        episode_features: pd.Series | None = None,
    ) -> FamilyAssignment:
        """Assign a family to one episode.

        Args:
            episode_frame: Windows belonging to the episode.
            config: Minimum confidence and method label.
            episode_features: Precomputed episode feature row; derived via
                :meth:`_derive_features` when ``None``.

        Returns:
            The :class:`FamilyAssignment`.

        Notes:
            Ties between equal best scores resolve to the first candidate in
            insertion order (explicit scorers, then signatures, then
            semantic-only families), because :func:`max` keeps the first
            maximum.
        """
        features = episode_features if episode_features is not None else self._derive_features(episode_frame)
        scores = self._score_candidates(episode_frame, features)
        best = max(scores, key=lambda item: item.score)
        alternatives = tuple(
            item.family
            for item in sorted(scores, key=lambda item: item.score, reverse=True)
            if item.family != best.family and item.score > 0.0
        )
        primary_family = best.family if best.score >= config.minimum_confidence else "unclassified_incident"
        confidence = best.score if primary_family != "unclassified_incident" else 0.0
        evidence = best.evidence if primary_family != "unclassified_incident" else tuple()
        return FamilyAssignment(
            primary_family=primary_family,
            secondary_families=alternatives,
            family_confidence=confidence,
            assignment_method=config.method,
            evidence=evidence,
        )

    def _score_candidates(self, episode_frame: pd.DataFrame, features: pd.Series) -> list[FamilyScore]:
        """Score and merge all candidate families.

        Args:
            episode_frame: Episode windows.
            features: Episode feature row.

        Returns:
            One merged :class:`FamilyScore` per family (max score, capped at
            1, union of evidence), after competition resolution; a single
            ``unclassified_incident`` score of 0.0 if no candidate exists.
        """
        semantic_scores = self._semantic_family_scores(episode_frame, features)
        explicit_scores = [
            self._score_pump_abrupt_failure(episode_frame, features),
            self._score_float_recurrent_disturbance(episode_frame, features),
            self._score_process_saturation(episode_frame, features),
            self._score_post_intervention_recovery(episode_frame, features),
            self._score_external_ambiguous_disturbance(episode_frame, features),
        ]
        signature_scores = self._signature_scores(features)

        combined: dict[str, FamilyScore] = {}
        for score in explicit_scores + signature_scores:
            existing = combined.get(score.family)
            if existing is None:
                combined[score.family] = score
            else:
                merged_evidence = existing.evidence + tuple(item for item in score.evidence if item not in existing.evidence)
                combined[score.family] = FamilyScore(score.family, min(max(existing.score, score.score), 1.0), merged_evidence)

        for semantic_score in semantic_scores:
            existing = combined.get(semantic_score.family)
            if existing is None:
                combined[semantic_score.family] = semantic_score
                continue
            merged_evidence = existing.evidence + tuple(item for item in semantic_score.evidence if item not in existing.evidence)
            combined[semantic_score.family] = FamilyScore(
                family=existing.family,
                score=min(max(existing.score, semantic_score.score), 1.0),
                evidence=merged_evidence,
            )

        combined = self._resolve_competing_families(combined, features)
        return list(combined.values()) or [FamilyScore("unclassified_incident", 0.0, tuple())]

    def _resolve_competing_families(
        self,
        scores: dict[str, FamilyScore],
        features: pd.Series,
    ) -> dict[str, FamilyScore]:
        """Resolve competition between pump failure and process saturation.

        When both families compete (see
        :meth:`_episode_has_pump_saturation_competition`), the preferred one
        is raised above the other and the other is demoted:

        * saturation preferred: ``sat = min(max(sat + 0.10, pump + 0.01), 1)``
          and ``pump = max(pump - 0.12, 0)``;
        * pump preferred: ``pump = min(max(pump + 0.08, sat + 0.01), 1)`` and
          ``sat = max(sat - 0.10, 0)``.

        Saturation preference is checked first.

        Args:
            scores: Merged scores keyed by family (modified in place).
            features: Episode feature row.

        Returns:
            The (possibly updated) ``scores`` mapping.
        """
        pump = scores.get("pump_abrupt_failure")
        saturation = scores.get("process_saturation")
        if pump is None or saturation is None:
            return scores

        if not self._episode_has_pump_saturation_competition(features, pump, saturation):
            return scores

        if self._prefer_process_saturation_over_pump(features):
            scores["process_saturation"] = FamilyScore(
                "process_saturation",
                min(max(saturation.score + 0.10, pump.score + 0.01), 1.0),
                saturation.evidence + tuple(item for item in (
                    "episode-level family resolution favored sustained regime",
                    "persistent/degraded context dominates abrupt signature",
                ) if item not in saturation.evidence),
            )
            scores["pump_abrupt_failure"] = FamilyScore(
                "pump_abrupt_failure",
                max(pump.score - 0.12, 0.0),
                pump.evidence + tuple(item for item in (
                    "episode-level family resolution demoted abrupt family",
                ) if item not in pump.evidence),
            )
        elif self._prefer_pump_over_process_saturation(features):
            scores["pump_abrupt_failure"] = FamilyScore(
                "pump_abrupt_failure",
                min(max(pump.score + 0.08, saturation.score + 0.01), 1.0),
                pump.evidence + tuple(item for item in (
                    "episode-level family resolution favored abrupt rupture",
                ) if item not in pump.evidence),
            )
            scores["process_saturation"] = FamilyScore(
                "process_saturation",
                max(saturation.score - 0.10, 0.0),
                saturation.evidence + tuple(item for item in (
                    "episode-level family resolution demoted saturation family",
                ) if item not in saturation.evidence),
            )
        return scores

    def _episode_has_pump_saturation_competition(
        self,
        features: pd.Series,
        pump: FamilyScore,
        saturation: FamilyScore,
    ) -> bool:
        """Return whether pump and saturation genuinely compete.

        Args:
            features: Episode feature row.
            pump: Merged pump-abrupt-failure score.
            saturation: Merged process-saturation score.

        Returns:
            ``True`` if both scores are at least 0.45 and either
            ``mixed_family_evidence >= 1`` or the scores differ by at most
            0.2.
        """
        return bool(
            pump.score >= 0.45
            and saturation.score >= 0.45
            and (
                self._feature(features, "mixed_family_evidence") >= 1.0
                or abs(pump.score - saturation.score) <= 0.2
            )
        )

    def _prefer_process_saturation_over_pump(self, features: pd.Series) -> bool:
        """Return whether sustained-regime evidence favours process saturation.

        True when any holds: (persistent dominant word >= 0.65 and low word
        transition rate <= 0.35) with (duration drift >= 0.25 or persistence
        excess >= 0.15); or state 17 / off-nominal states (>= 0.25) with a
        persistent, low-transition regime; or no consumption drop (signed
        deviation >= -0.05) with duration drift and low transition rate.

        Args:
            features: Episode feature row.

        Returns:
            The preference flag.
        """
        persistent_regime = self._feature(features, "mean_dominant_state_word_fraction") >= 0.65
        low_transition = self._feature(features, "mean_state_word_transition_rate") <= 0.35
        duration_drift = self._feature(features, "median_duration_drift") >= 0.25
        persistence_excess = self._feature(features, "max_persistence_excess") >= 0.15
        state17_or_degraded = self._feature(features, "mean_state_17_fraction") > 0.0 or self._feature(features, "mean_off_nominal_state_fraction") >= 0.25
        no_drop = self._feature(features, "signed_consumption_deviation") >= -0.05
        return bool(
            ((persistent_regime and low_transition) and (duration_drift or persistence_excess))
            or (state17_or_degraded and persistent_regime and low_transition)
            or (no_drop and duration_drift and low_transition)
        )

    def _prefer_pump_over_process_saturation(self, features: pd.Series) -> bool:
        """Return whether abrupt-rupture evidence favours pump failure.

        Requires all of: severe consumption drop (min <= -0.7), abrupt onset
        (slope >= 0.25), a state break (max state error >= 0.25) or high
        sequence divergence (>= 0.8), and off-nominal words (match <= 0.25)
        or unstable words (transition rate >= 0.5).

        Args:
            features: Episode feature row.

        Returns:
            The preference flag.
        """
        severe_drop = self._feature(features, "min_consumption_deviation") <= -0.7
        abrupt_onset = self._feature(features, "onset_slope") >= 0.25
        state_break = self._feature(features, "max_state_error_rate") >= 0.25
        high_divergence = self._feature(features, "max_sequence_divergence") >= 0.8
        off_nominal = self._feature(features, "mean_nominal_state_word_match_fraction") <= 0.25
        unstable_words = self._feature(features, "mean_state_word_transition_rate") >= 0.5
        return bool(
            severe_drop and abrupt_onset and (state_break or high_divergence) and (off_nominal or unstable_words)
        )

    def _semantic_family_scores(self, episode_frame: pd.DataFrame, features: pd.Series) -> list[FamilyScore]:
        """Score families by their Model_C label frequency within the episode.

        Args:
            episode_frame: Episode windows with ``incident_family``.
            features: Episode feature row (uses ``peak_score``).

        Returns:
            One score per observed label equal to its relative frequency,
            plus 0.1 (capped at 1) when ``peak_score >= 2.0``.
        """
        family_counts = episode_frame.get("incident_family", pd.Series(dtype=str)).value_counts(normalize=True)
        scores: list[FamilyScore] = []
        peak_score = self._feature(features, "peak_score")
        for family, fraction in family_counts.items():
            evidence = [f"semantic label frequency {fraction:.2f}"]
            score = float(fraction)
            if peak_score >= 2.0:
                evidence.append("high peak deviation")
                score = min(score + 0.1, 1.0)
            scores.append(FamilyScore(family=family, score=score, evidence=tuple(evidence)))
        return scores

    def _signature_scores(self, features: pd.Series) -> list[FamilyScore]:
        """Score every declarative family signature.

        Args:
            features: Episode feature row.

        Returns:
            One :class:`FamilyScore` per entry of ``FAMILY_SIGNATURES``.
        """
        return [self._score_signature(signature, features) for signature in FAMILY_SIGNATURES.values()]

    def _score_signature(self, signature: FamilySignature, features: pd.Series) -> FamilyScore:
        r"""Score one declarative signature.

        With :math:`r` required, :math:`r_s` satisfied required,
        :math:`q` preferred and :math:`q_s` satisfied preferred features,

        .. math::

           s = \begin{cases} 0.45 & r_s = r > 0 \\
                             0.20\, r_s / r & 0 < r_s < r \\
                             0 & \text{otherwise} \end{cases}
               \; + \; 0.35\, q_s / q,

        then 0.25 is subtracted (floored at 0) if any incompatible feature is
        present, and the result is capped at 1.

        Args:
            signature: Declarative family signature.
            features: Episode feature row.

        Returns:
            The signature's :class:`FamilyScore`.
        """
        score = 0.0
        evidence: list[str] = []
        required = list(signature.required_features)
        preferred = list(signature.preferred_features)
        incompatible = list(signature.incompatible_features)

        if required:
            satisfied_required = [feature for feature in required if self._feature_present(features, feature)]
            if len(satisfied_required) == len(required):
                score += 0.45
                evidence.extend(f"required feature: {feature}" for feature in satisfied_required)
            elif satisfied_required:
                score += 0.20 * (len(satisfied_required) / len(required))
                evidence.extend(f"partial required feature: {feature}" for feature in satisfied_required)
        if preferred:
            satisfied_preferred = [feature for feature in preferred if self._feature_present(features, feature)]
            if satisfied_preferred:
                score += min(0.35, 0.35 * len(satisfied_preferred) / len(preferred))
                evidence.extend(f"preferred feature: {feature}" for feature in satisfied_preferred)
        violated = [feature for feature in incompatible if self._feature_present(features, feature)]
        if violated:
            score = max(score - 0.25, 0.0)
            evidence.extend(f"incompatible feature: {feature}" for feature in violated)
        return FamilyScore(signature.family, min(score, 1.0), tuple(evidence))

    def _score_pump_abrupt_failure(self, episode_frame: pd.DataFrame, features: pd.Series) -> FamilyScore:
        """Score the abrupt pump-failure family.

        Increments: consumption drop (min <= -0.5) +0.35; state prediction
        degradation (max error >= 0.2) +0.25; sequence rupture (max
        divergence >= 0.7) +0.25; abrupt onset (slope >= 0.2) +0.15; far
        from nominal words (match <= 0.35 and distance >= 0.8) +0.10; rare
        state presence +0.10. Capped at 1.

        Args:
            episode_frame: Episode windows (unused).
            features: Episode feature row.

        Returns:
            The family score.
        """
        score = 0.0
        evidence: list[str] = []
        if self._feature(features, "min_consumption_deviation") <= -0.5:
            score += 0.35
            evidence.append("large consumption drop")
        if self._feature(features, "max_state_error_rate") >= 0.2:
            score += 0.25
            evidence.append("state prediction degradation")
        if self._feature(features, "max_sequence_divergence") >= 0.7:
            score += 0.25
            evidence.append("sequence rupture")
        if self._feature(features, "onset_slope") >= 0.2:
            score += 0.15
            evidence.append("abrupt onset")
        if self._feature(features, "mean_nominal_state_word_match_fraction") <= 0.35 and self._feature(features, "mean_state_distance_to_nominal") >= 0.8:
            score += 0.10
            evidence.append("far from nominal word pattern")
        if self._feature(features, "mean_state_17_fraction") > 0.0 or self._feature(features, "mean_rare_state_fraction") >= 0.2:
            score += 0.10
            evidence.append("rare state presence")
        return FamilyScore("pump_abrupt_failure", min(score, 1.0), tuple(evidence))

    def _score_float_recurrent_disturbance(self, episode_frame: pd.DataFrame, features: pd.Series) -> FamilyScore:
        """Score the recurrent float-disturbance family.

        Combines semantic support (label frequency >= 0.6), recurrence and
        persistence excess, preserved state accuracy, behavioural
        divergence, unstable/multiple/off-nominal word regimes and word
        distribution shift. Capped at 1.

        Args:
            episode_frame: Episode windows (for the semantic frequency).
            features: Episode feature row.

        Returns:
            The family score.
        """
        score = 0.0
        evidence: list[str] = []
        semantic_frequency = self._semantic_frequency(episode_frame, "float_recurrent_disturbance")
        if semantic_frequency >= 0.6:
            score += 0.35
            evidence.append("dominant float disturbance semantic label")
        if self._feature(features, "max_recurrence_excess") >= 0.2:
            score += 0.25
            evidence.append("high recurrence excess")
        if self._feature(features, "median_persistence_excess") >= 0.1:
            score += 0.20
            evidence.append("persistent recurrent deviation")
        if self._feature(features, "mean_state_error_rate") <= 0.2:
            score += 0.10
            evidence.append("state accuracy preserved")
        if self._feature(features, "max_sequence_divergence") >= 0.75:
            score += 0.10
            evidence.append("behavioral divergence present")
        if self._feature(features, "mean_state_word_transition_rate") >= 0.55:
            score += 0.20
            evidence.append("unstable word regime")
        if self._feature(features, "dominant_state_word_count") >= 2.0:
            score += 0.05
            evidence.append("multiple dominant words across episode")
        if self._feature(features, "mean_nominal_state_word_match_fraction") <= 0.4:
            score += 0.10
            evidence.append("recurrent off-nominal word regime")
        if self._feature(features, "mean_word_regime_shift_score") >= 0.45 or self._feature(features, "mean_rare_word_fraction") >= 0.35:
            score += 0.15
            evidence.append("shifted word distribution")
        return FamilyScore("float_recurrent_disturbance", min(score, 1.0), tuple(evidence))

    def _score_process_saturation(self, episode_frame: pd.DataFrame, features: pd.Series) -> FamilyScore:
        """Score the sustained process-saturation family.

        Combines semantic support (frequency >= 0.4), longer-than-nominal
        runs, persistence excess, absence of a consumption drop, mode drift,
        a persistent stable word regime, near-nominal words with abnormal
        timing and rare-word pressure without a major state shift. Capped at
        1.

        Args:
            episode_frame: Episode windows (for the semantic frequency).
            features: Episode feature row.

        Returns:
            The family score.
        """
        score = 0.0
        evidence: list[str] = []
        semantic_frequency = self._semantic_frequency(episode_frame, "process_saturation")
        if semantic_frequency >= 0.4:
            score += 0.25
            evidence.append("process saturation semantic support")
        if self._feature(features, "median_duration_drift") >= 0.25:
            score += 0.25
            evidence.append("longer-than-nominal runs")
        if self._feature(features, "max_persistence_excess") >= 0.15:
            score += 0.25
            evidence.append("persistent overload signal")
        if self._feature(features, "signed_consumption_deviation") >= 0.0:
            score += 0.10
            evidence.append("no consumption drop during anomaly")
        if self._feature(features, "mean_mode_divergence") >= 0.15:
            score += 0.10
            evidence.append("mode drift during sustained event")
        if self._feature(features, "mean_dominant_state_word_fraction") >= 0.65:
            score += 0.12
            evidence.append("persistent degraded word regime")
        if self._feature(features, "mean_state_word_transition_rate") <= 0.35 and self._feature(features, "median_state_word_diversity") <= 2.0:
            score += 0.08
            evidence.append("stable non-nominal word persistence")
        if self._feature(features, "mean_nominal_state_word_match_fraction") >= 0.6 and self._feature(features, "mean_nominal_word_anomaly_score") >= 0.4:
            score += 0.10
            evidence.append("near nominal word with abnormal duration or timing")
        if self._feature(features, "mean_rare_word_fraction") >= 0.25 and self._feature(features, "mean_rare_state_fraction") < 0.15:
            score += 0.10
            evidence.append("rare word pressure without major state shift")
        return FamilyScore("process_saturation", min(score, 1.0), tuple(evidence))

    def _score_post_intervention_recovery(self, episode_frame: pd.DataFrame, features: pd.Series) -> FamilyScore:
        """Score the post-intervention recovery family.

        Semantic support (frequency >= 0.6) +0.50; moderate mean score
        (<= 0.75) +0.25; recovered state model (mean error <= 0.1) +0.15;
        stabilized mode (mean divergence <= 0.1) +0.10.

        Args:
            episode_frame: Episode windows (for the semantic frequency).
            features: Episode feature row.

        Returns:
            The family score.

        Notes:
            Without semantic support the score can still reach 0.50 from
            low-deviation features alone, which may exceed weaker families.
        """
        score = 0.0
        evidence: list[str] = []
        semantic_frequency = self._semantic_frequency(episode_frame, "post_intervention_recovery")
        if semantic_frequency >= 0.6:
            score += 0.50
            evidence.append("post-intervention semantic support")
        if self._feature(features, "mean_score") <= 0.75:
            score += 0.25
            evidence.append("moderate residual deviation")
        if self._feature(features, "mean_state_error_rate") <= 0.1:
            score += 0.15
            evidence.append("state model recovered")
        if self._feature(features, "mean_mode_divergence") <= 0.1:
            score += 0.10
            evidence.append("mode stabilized")
        return FamilyScore("post_intervention_recovery", min(score, 1.0), tuple(evidence))

    def _score_external_ambiguous_disturbance(self, episode_frame: pd.DataFrame, features: pd.Series) -> FamilyScore:
        """Score the external/ambiguous disturbance family.

        Semantic support (frequency >= 0.4) +0.35; cross-domain deviation
        (at least two of sequence divergence >= 0.6, mode divergence >= 0.2,
        state error >= 0.15) +0.35; mixed family evidence +0.20; word
        distribution shift (>= 0.5) +0.10; weak recurrence (< 0.2) +0.10.

        Args:
            episode_frame: Episode windows (for the semantic frequency).
            features: Episode feature row.

        Returns:
            The family score.
        """
        score = 0.0
        evidence: list[str] = []
        feature_support = sum(
            [
                self._feature(features, "max_sequence_divergence") >= 0.6,
                self._feature(features, "max_mode_divergence") >= 0.2,
                self._feature(features, "max_state_error_rate") >= 0.15,
            ]
        )
        semantic_frequency = self._semantic_frequency(episode_frame, "external_ambiguous_disturbance")
        if semantic_frequency >= 0.4:
            score += 0.35
            evidence.append("external disturbance semantic support")
        if feature_support >= 2:
            score += 0.35
            evidence.append("cross-domain deviation without dominant family")
        if self._feature(features, "mixed_family_evidence") >= 1.0:
            score += 0.20
            evidence.append("mixed family evidence")
        if self._feature(features, "mean_word_regime_shift_score") >= 0.5:
            score += 0.10
            evidence.append("word distribution shift without stable family")
        if self._feature(features, "max_recurrence_excess") < 0.2:
            score += 0.10
            evidence.append("not strongly recurrent")
        return FamilyScore("external_ambiguous_disturbance", min(score, 1.0), tuple(evidence))

    @staticmethod
    def _semantic_frequency(episode_frame: pd.DataFrame, family: str) -> float:
        """Return the fraction of episode windows labelled with ``family``.

        Args:
            episode_frame: Episode windows.
            family: Family identifier.

        Returns:
            Fraction in ``[0, 1]``; 0.0 without ``incident_family`` labels.
        """
        family_series = episode_frame.get("incident_family", pd.Series(dtype=str))
        if family_series.empty:
            return 0.0
        return float(family_series.eq(family).mean())

    @staticmethod
    def _derive_features(episode_frame: pd.DataFrame) -> pd.Series:
        """Derive episode features by wrapping the windows in a synthetic episode.

        Args:
            episode_frame: Episode windows.

        Returns:
            The single feature row returned by
            :func:`~iabm_incidents.features.build_episode_features` for an
            episode spanning ``min(start_time)``..``max(end_time)`` whose
            source windows are all rows of ``episode_frame`` (identified by
            index label) and whose peak/mean score are taken from
            ``deviation_score``.
        """
        synthetic_episodes = pd.DataFrame(
            {
                "episode_id": ["synthetic-episode"],
                "event_start": [pd.to_datetime(episode_frame["start_time"].min()) if "start_time" in episode_frame.columns and not episode_frame.empty else pd.NaT],
                "event_end": [pd.to_datetime(episode_frame["end_time"].max()) if "end_time" in episode_frame.columns and not episode_frame.empty else pd.NaT],
                "duration_seconds": [
                    (pd.to_datetime(episode_frame["end_time"].max()) - pd.to_datetime(episode_frame["start_time"].min())).total_seconds()
                    if {"start_time", "end_time"}.issubset(episode_frame.columns) and not episode_frame.empty
                    else 0.0
                ],
                "time_to_recovery_seconds": [None],
                "peak_score": [float(pd.to_numeric(episode_frame.get("deviation_score", pd.Series([0.0])), errors="coerce").fillna(0.0).max())],
                "mean_score": [float(pd.to_numeric(episode_frame.get("deviation_score", pd.Series([0.0])), errors="coerce").fillna(0.0).mean())],
                "source_window_ids": [tuple(str(index) for index in episode_frame.index)],
                "asset_id": [episode_frame.get("asset_id", pd.Series([None])).iloc[0] if not episode_frame.empty else None],
            }
        )
        return build_episode_features(synthetic_episodes, episode_frame).iloc[0]

    @staticmethod
    def _feature(features: pd.Series, key: str, cast=float):
        """Read a feature with a neutral default.

        Args:
            features: Episode feature row.
            key: Feature name.
            cast: Conversion callable (default :class:`float`).

        Returns:
            ``cast(value)``, or ``cast()`` (``0.0`` for float, ``""`` for
            str) when the value is missing, NaN or not convertible.
        """
        value = features.get(key, None)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return cast() if cast is not str else ""
        try:
            return cast(value)
        except Exception:
            return cast() if cast is not str else ""

    @staticmethod
    def _feature_present(features: pd.Series, feature_name: str) -> bool:
        """Evaluate a named qualitative feature used by family signatures.

        Args:
            features: Episode feature row.
            feature_name: Qualitative feature name referenced by
                :class:`~iabm_incidents.taxonomy.FamilySignature`.

        Returns:
            ``True`` when the threshold rule for ``feature_name`` holds;
            ``False`` for unknown names. NaN features compare as ``False``.

        Raises:
            TypeError: If an underlying feature value is explicitly ``None``
                (``float(None)`` is not guarded here, unlike in
                :meth:`_feature`).
        """
        mapping = {
            "abrupt_onset": lambda f: float(f.get("onset_slope", 0.0)) >= 0.2,
            "negative_consumption_deviation": lambda f: float(f.get("min_consumption_deviation", 0.0)) <= -0.5,
            "digital_electrical_mismatch": lambda f: float(f.get("max_state_error_rate", 0.0)) >= 0.2,
            "high_sequence_divergence": lambda f: float(f.get("max_sequence_divergence", 0.0)) >= 0.7,
            "high_state_disagreement": lambda f: float(f.get("max_state_error_rate", 0.0)) >= 0.2,
            "recurrence_excess": lambda f: float(f.get("max_recurrence_excess", 0.0)) >= 0.2,
            "persistence_excess": lambda f: float(f.get("max_persistence_excess", 0.0)) >= 0.15,
            "state_stability": lambda f: float(f.get("mean_state_error_rate", 0.0)) <= 0.2,
            "duration_drift": lambda f: float(f.get("median_duration_drift", 0.0)) >= 0.25,
            "mode_divergence": lambda f: float(f.get("mean_mode_divergence", 0.0)) >= 0.15,
            "non_negative_consumption_deviation": lambda f: float(f.get("signed_consumption_deviation", 0.0)) >= 0.0,
            "moderate_residual_deviation": lambda f: float(f.get("mean_score", 0.0)) <= 0.75,
            "restored_state_agreement": lambda f: float(f.get("mean_state_error_rate", 0.0)) <= 0.1,
            "mixed_family_evidence": lambda f: float(f.get("mixed_family_evidence", 0.0)) >= 1.0,
            "cross_domain_deviation": lambda f: sum([
                float(f.get("max_sequence_divergence", 0.0)) >= 0.6,
                float(f.get("max_mode_divergence", 0.0)) >= 0.2,
                float(f.get("max_state_error_rate", 0.0)) >= 0.15,
            ]) >= 2,
            "stable_word_regime": lambda f: float(f.get("mean_state_word_transition_rate", 0.0)) <= 0.25,
            "word_regime_instability": lambda f: float(f.get("mean_state_word_transition_rate", 0.0)) >= 0.55,
            "persistent_word_regime": lambda f: float(f.get("mean_dominant_state_word_fraction", 0.0)) >= 0.65,
            "far_from_nominal_word": lambda f: float(f.get("mean_nominal_state_word_match_fraction", 0.0)) <= 0.35 or float(f.get("mean_state_distance_to_nominal", 0.0)) >= 0.8,
            "near_nominal_word_with_duration_drift": lambda f: float(f.get("mean_nominal_state_word_match_fraction", 0.0)) >= 0.6 and float(f.get("median_duration_drift", 0.0)) >= 0.25,
            "rare_state_presence": lambda f: float(f.get("mean_state_17_fraction", 0.0)) > 0.0 or float(f.get("mean_rare_state_fraction", 0.0)) >= 0.2,
            "word_distribution_shift": lambda f: float(f.get("mean_word_regime_shift_score", 0.0)) >= 0.45,
            "rare_word_pressure_without_state_shift": lambda f: float(f.get("mean_rare_word_fraction", 0.0)) >= 0.25 and float(f.get("mean_rare_state_fraction", 0.0)) < 0.15,
        }
        checker = mapping.get(feature_name)
        return bool(checker(features)) if checker is not None else False
