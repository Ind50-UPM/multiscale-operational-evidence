"""Semantic interpretation utilities for Model_C.

The module bridges behavioral outputs from ``Model_B`` and the longitudinal
incident-processing logic in ``Model_D``. It enriches raw state words with
component semantics, operating modes, incident-family proxies, reliability
interpretations, and life/recovery regimes.

State encoding:
    Each integer state is a bitmask of simultaneously active components. The
    default mapping :data:`DEFAULT_COMPONENT_MAP` is::

        1  -> DIVING_PUMP_1      8  -> FEEDBACK_PUMP_2
        2  -> DIVING_PUMP_2      16 -> FLOCCULANT_PUMP
        4  -> FEEDBACK_PUMP_1    32 -> BASIN_PUMP

    A word such as ``(1, 5)`` therefore involves ``DIVING_PUMP_1`` and
    ``FEEDBACK_PUMP_1``. The component set of a word is the union over all its
    states (order and repetition are discarded).

Interpretation chain (per sequence row):

1. ``states`` -> component set (:meth:`SemanticModeInterpreter._decode_sequence_components`).
2. component set -> ``(operating_mode, working_mode)`` via user rules first,
   then built-in heuristics (:meth:`SemanticModeInterpreter._assign_modes`).
3. optional comparison row -> ``semantic_status`` (``"NORMAL"`` /
   ``"ANOMALOUS"``) and ``anomaly_score``.
4. status + components + score -> ``incident_family``.
5. family + status (+ components) -> ``reliability_interpretation``,
   ``life_regime`` and ``recovery_regime``.

Data contracts:

* Sequences dataframe: column ``states`` containing a tuple, a list, or a
  string literal of a tuple/list of integers (as written by the Model_B CLI,
  e.g. ``"(1, 2)"``). Other columns are ignored.
* Comparison dataframe (optional): columns ``anomaly_score`` (float) and
  ``is_anomalous`` (bool), aligned with the sequences **by row position**.
* Output: one row per sequence with the fields of
  :class:`SemanticAssignment` as columns.

.. mermaid::

   classDiagram
       class SemanticRule {
           +tuple required_components
           +str operating_mode
           +str working_mode
       }
       class SemanticAssignment {
           +tuple sequence_states
           +tuple components
           +str operating_mode
           +str working_mode
           +str semantic_status
           +float anomaly_score
           +str incident_family
           +str reliability_interpretation
           +str life_regime
           +str recovery_regime
       }
       class SemanticModeInterpreter {
           +dict component_map
           +list rules
           +load_active_sequences(file_path) DataFrame
           +load_comparison_report(file_path) DataFrame
           +load_rules(file_path) list~SemanticRule~
           +interpret_sequences(sequences, comparison) DataFrame
           +summarize_modes(assignments) DataFrame
           +summarize_life_regimes(assignments) DataFrame
       }
       SemanticModeInterpreter o-- SemanticRule : rules
       SemanticModeInterpreter ..> SemanticAssignment : creates

Maintenance notes:
    All label strings emitted here are part of the contract with Model_D and
    the published reports; treat them as an enumeration. The heuristics are
    hand-written for the wastewater pumping installation behind the default
    component map: a different plant should supply its own ``component_map``
    and ``rules``. Because a ``"NORMAL"`` status short-circuits every
    downstream mapping, several branches (e.g. the ``"recovering"`` recovery
    regime, or the ``"post_intervention_recovery"`` entry of the reliability
    mapping) are not reachable with the current family heuristic; keep that in
    mind before relying on them.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


# Note: keys are single-bit masks; a state may activate several components.
DEFAULT_COMPONENT_MAP = {
    1: "DIVING_PUMP_1",
    2: "DIVING_PUMP_2",
    4: "FEEDBACK_PUMP_1",
    8: "FEEDBACK_PUMP_2",
    16: "FLOCCULANT_PUMP",
    32: "BASIN_PUMP",
}


@dataclass(frozen=True)
class SemanticRule:
    """Declarative mapping from component requirements to semantic modes.

    A rule matches when all of its ``required_components`` are present in the
    decoded component set of a sequence (subset test; extra components are
    allowed). Rules are evaluated in list order and the first match wins.

    Attributes:
        required_components: Component names (values of the component map)
            that must all be active. An empty tuple matches every sequence.
        operating_mode: Operating-mode label returned on match.
        working_mode: Working-mode label returned on match.

    Examples:
        >>> SemanticRule(("BASIN_PUMP",), "CUSTOM_MODE", "CUSTOM_WORK")
        SemanticRule(required_components=('BASIN_PUMP',), operating_mode='CUSTOM_MODE', working_mode='CUSTOM_WORK')
    """

    required_components: tuple[str, ...]
    operating_mode: str
    working_mode: str


@dataclass(frozen=True)
class SemanticAssignment:
    """Row-level semantic interpretation emitted for one behavioral sequence.

    Attributes:
        sequence_states: Parsed state word (tuple of ``int``).
        components: Sorted tuple of decoded component names.
        operating_mode: Operating-mode label (e.g. ``"PUMPING_MODE"``,
            ``"IDLE"``, or a rule-defined label).
        working_mode: Working-mode label (e.g. ``"DIVING_ONLY"``).
        semantic_status: ``"ANOMALOUS"`` if the aligned comparison row flags
            the sequence, otherwise ``"NORMAL"`` (also when no comparison is
            available).
        anomaly_score: Model_B anomaly score of the aligned comparison row, or
            ``None`` when no comparison row is available.
        incident_family: Incident-family proxy, one of
            ``"post_intervention_recovery"``, ``"process_saturation"``,
            ``"float_recurrent_disturbance"``, ``"pump_abrupt_failure"`` or
            ``"external_ambiguous_disturbance"``.
        reliability_interpretation: Reliability label derived from the family
            (``"nominal_or_stable_regime"`` for normal rows).
        life_regime: Asset life-regime label (``"nominal_operation"`` for
            normal rows).
        recovery_regime: Recovery-regime label (``"stable_recovered"`` for
            normal rows).
    """

    sequence_states: tuple[int, ...]
    components: tuple[str, ...]
    operating_mode: str
    working_mode: str
    semantic_status: str
    anomaly_score: float | None
    incident_family: str
    reliability_interpretation: str
    life_regime: str
    recovery_regime: str


class SemanticModeInterpreter:
    """Interpret Model_B sequences as semantic and reliability-oriented labels.

    Args:
        component_map: Optional override for the bitmask-to-component mapping.
            Defaults to :data:`DEFAULT_COMPONENT_MAP`. An empty dict also falls
            back to the default (falsy check).
        rules: Optional declarative semantic rules evaluated before heuristics.

    Attributes:
        component_map: Copy of the bitmask -> component-name mapping in use.
        rules: List of :class:`SemanticRule` evaluated in order by
            :meth:`_assign_modes`; replaced wholesale by :meth:`load_rules`.

    Examples:
        >>> import pandas as pd
        >>> interp = SemanticModeInterpreter()
        >>> out = interp.interpret_sequences(pd.DataFrame({"states": ["(1, 2)"]}))
        >>> out.loc[0, "operating_mode"], out.loc[0, "working_mode"]
        ('PUMPING_MODE', 'DIVING_ONLY')

    Notes:
        The interpreter first applies explicit rules, then falls back to compact
        heuristic logic. This makes it easy to start from transparent defaults
        while still supporting future domain-specific rule injections.
    """

    def __init__(self, component_map: dict[int, str] | None = None, rules: Iterable[SemanticRule] | None = None) -> None:
        """Initialise the interpreter.

        Args:
            component_map: Optional bitmask -> component-name mapping; copied.
            rules: Optional iterable of :class:`SemanticRule`; materialised
                into a list.
        """
        self.component_map = dict(component_map or DEFAULT_COMPONENT_MAP)
        self.rules = list(rules or [])

    def load_active_sequences(self, file_path: str | Path) -> pd.DataFrame:
        """Load a Model_B sequence report from parquet, Excel, or CSV.

        Args:
            file_path: Path to the report; the reader is chosen from the
                case-insensitive suffix (``.parquet``, ``.xlsx``/``.xls``,
                ``.csv``).

        Returns:
            The report as read, without any column validation or parsing.

        Raises:
            ValueError: If the file extension is not supported.
        """
        path = Path(file_path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    def load_comparison_report(self, file_path: str | Path) -> pd.DataFrame:
        """Load a comparison report using the same tabular backends as sequences.

        Args:
            file_path: Path to a Model_B ``sequence_comparison`` report.

        Returns:
            The report as read.

        Raises:
            ValueError: If the file extension is not supported.

        See Also:
            :meth:`load_active_sequences`
        """
        return self.load_active_sequences(file_path)

    def load_rules(self, file_path: str | Path) -> list[SemanticRule]:
        """Load declarative semantic rules from JSON.

        The file must contain a JSON array of objects, each with keys
        ``required_components`` (array of component names),
        ``operating_mode`` and ``working_mode``. Extra keys are ignored.

        Args:
            file_path: Path to the UTF-8 encoded JSON file.

        Returns:
            The loaded rules, which also replace :attr:`rules`.

        Raises:
            json.JSONDecodeError: If the file is not valid JSON.
            KeyError: If a rule object lacks one of the required keys.

        Examples:
            A minimal rules file::

                [{"required_components": ["BASIN_PUMP"],
                  "operating_mode": "CUSTOM_MODE",
                  "working_mode": "CUSTOM_WORK"}]
        """
        payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
        self.rules = [
            SemanticRule(
                required_components=tuple(item["required_components"]),
                operating_mode=item["operating_mode"],
                working_mode=item["working_mode"],
            )
            for item in payload
        ]
        return self.rules

    def interpret_sequences(self, sequences: pd.DataFrame, *, comparison: pd.DataFrame | None = None) -> pd.DataFrame:
        """Interpret every behavioral sequence row into semantic and family labels.

        Args:
            sequences: Model_B active-sequence dataframe with a ``states``
                column.
            comparison: Optional anomaly comparison dataframe aligned row-wise,
                with ``anomaly_score`` and ``is_anomalous`` columns.

        Returns:
            One row per input sequence with semantic, reliability, and family
            fields required by downstream incident processing (columns follow
            :class:`SemanticAssignment`). An empty input yields a dataframe
            with no rows and no columns.

        Raises:
            KeyError: If ``states`` (or, when a comparison is given,
                ``anomaly_score``/``is_anomalous``) is missing.
            ValueError: If a ``states`` string cannot be parsed as a literal.

        Notes:
            The current implementation aligns comparison rows by positional
            order. That contract is acceptable because the repository pipeline
            generates both tables from the same ordered sequence export.
            Sequence rows beyond the length of ``comparison`` silently receive
            ``semantic_status="NORMAL"`` and ``anomaly_score=None``.
        """
        # Why: reset both indices so ``index`` below is a 0-based position that
        # can be used with ``comparison.loc`` for positional alignment.
        comparison = comparison.reset_index(drop=True) if comparison is not None else None
        assignments = []
        for index, row in sequences.reset_index(drop=True).iterrows():
            states = self._parse_states(row["states"])
            components = self._decode_sequence_components(states)
            operating_mode, working_mode = self._assign_modes(components)

            anomaly_score = None
            semantic_status = "NORMAL"
            if comparison is not None and index < len(comparison):
                anomaly_score = float(comparison.loc[index, "anomaly_score"])
                semantic_status = "ANOMALOUS" if bool(comparison.loc[index, "is_anomalous"]) else "NORMAL"

            incident_family = self._assign_incident_family(
                components,
                anomaly_score=anomaly_score,
                semantic_status=semantic_status,
            )
            assignments.append(
                SemanticAssignment(
                    sequence_states=states,
                    components=components,
                    operating_mode=operating_mode,
                    working_mode=working_mode,
                    semantic_status=semantic_status,
                    anomaly_score=anomaly_score,
                    incident_family=incident_family,
                    reliability_interpretation=self._assign_reliability_interpretation(incident_family, semantic_status=semantic_status),
                    life_regime=self._assign_life_regime(incident_family, components, semantic_status=semantic_status),
                    recovery_regime=self._assign_recovery_regime(incident_family, semantic_status=semantic_status),
                ).__dict__
            )

        return pd.DataFrame(assignments)

    def summarize_modes(self, assignments: pd.DataFrame) -> pd.DataFrame:
        """Aggregate semantic assignments by mode, status, family, and life regime.

        Args:
            assignments: Output of :meth:`interpret_sequences`.

        Returns:
            Dataframe with columns ``operating_mode``, ``working_mode``,
            ``semantic_status``, ``incident_family``, ``life_regime`` and
            ``count``, sorted by ``count`` descending. An empty dataframe with
            these columns is returned for empty input.
        """
        if assignments.empty:
            return pd.DataFrame(
                columns=["operating_mode", "working_mode", "semantic_status", "incident_family", "life_regime", "count"]
            )
        return (
            assignments.groupby(
                ["operating_mode", "working_mode", "semantic_status", "incident_family", "life_regime"],
                dropna=False,
            )
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

    def summarize_life_regimes(self, assignments: pd.DataFrame) -> pd.DataFrame:
        """Aggregate assignments by family, life regime, and recovery regime.

        Args:
            assignments: Output of :meth:`interpret_sequences`.

        Returns:
            Dataframe with columns ``incident_family``, ``life_regime``,
            ``recovery_regime``, ``reliability_interpretation`` and ``count``,
            sorted by ``count`` descending. An empty dataframe with these
            columns is returned for empty input.
        """
        if assignments.empty:
            return pd.DataFrame(
                columns=["incident_family", "life_regime", "recovery_regime", "reliability_interpretation", "count"]
            )
        return (
            assignments.groupby(
                ["incident_family", "life_regime", "recovery_regime", "reliability_interpretation"],
                dropna=False,
            )
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

    def _decode_sequence_components(self, states: tuple[int, ...]) -> tuple[str, ...]:
        """Decode a state word into the set of participating industrial components.

        Args:
            states: State word (tuple of non-negative integers).

        Returns:
            Alphabetically sorted, de-duplicated tuple of component names whose
            bitmask intersects at least one state of the word. Bits not present
            in :attr:`component_map` are ignored; an all-zero word yields
            ``()``.

        Examples:
            >>> SemanticModeInterpreter()._decode_sequence_components((1, 5))
            ('DIVING_PUMP_1', 'FEEDBACK_PUMP_1')
        """
        components: set[str] = set()
        for state in states:
            for bitmask, component in self.component_map.items():
                # Note: bitwise AND, so multi-bit masks match on any shared bit.
                if state & bitmask:
                    components.add(component)
        return tuple(sorted(components))

    def _assign_modes(self, components: tuple[str, ...]) -> tuple[str, str]:
        """Assign operating and working modes from the decoded component set.

        User rules (:attr:`rules`) are tried first, in order. Otherwise the
        built-in heuristics are applied in this priority order:

        ===========================================  =======================  ==============================
        Condition                                    operating_mode           working_mode
        ===========================================  =======================  ==============================
        no components                                ``IDLE``                 ``NO_ACTIVE_COMPONENTS``
        ``BASIN_PUMP`` present                       ``TRANSFER_MODE``        ``BASIN_TRANSFER``
        flocculant + any diving pump                 ``TREATMENT_MODE``       ``FLOCCULANT_ASSISTED_CYCLE``
        any feedback + any diving pump               ``RECIRCULATION_MODE``   ``DIVING_FEEDBACK_CYCLE``
        only diving pumps                            ``PUMPING_MODE``         ``DIVING_ONLY``
        only feedback pumps                          ``RECIRCULATION_MODE``   ``FEEDBACK_ONLY``
        anything else                                ``COMPOSITE_MODE``       components joined with ``+``
        ===========================================  =======================  ==============================

        Args:
            components: Decoded component names.

        Returns:
            ``(operating_mode, working_mode)``.

        Notes:
            The heuristics refer to the default component names; with a custom
            :attr:`component_map` using other names, only the ``IDLE`` and
            ``COMPOSITE_MODE`` branches (and user rules) apply.
        """
        component_set = set(components)
        for rule in self.rules:
            if set(rule.required_components).issubset(component_set):
                return rule.operating_mode, rule.working_mode

        if not components:
            return "IDLE", "NO_ACTIVE_COMPONENTS"
        if "BASIN_PUMP" in component_set:
            return "TRANSFER_MODE", "BASIN_TRANSFER"
        if "FLOCCULANT_PUMP" in component_set and component_set & {"DIVING_PUMP_1", "DIVING_PUMP_2"}:
            return "TREATMENT_MODE", "FLOCCULANT_ASSISTED_CYCLE"
        if component_set & {"FEEDBACK_PUMP_1", "FEEDBACK_PUMP_2"} and component_set & {"DIVING_PUMP_1", "DIVING_PUMP_2"}:
            return "RECIRCULATION_MODE", "DIVING_FEEDBACK_CYCLE"
        # Note: ``<=`` is the subset test; the empty set was handled above.
        if component_set <= {"DIVING_PUMP_1", "DIVING_PUMP_2"}:
            return "PUMPING_MODE", "DIVING_ONLY"
        if component_set <= {"FEEDBACK_PUMP_1", "FEEDBACK_PUMP_2"}:
            return "RECIRCULATION_MODE", "FEEDBACK_ONLY"
        return "COMPOSITE_MODE", "+".join(components)

    @staticmethod
    def _assign_incident_family(
        components: tuple[str, ...],
        *,
        anomaly_score: float | None,
        semantic_status: str,
    ) -> str:
        """Map semantic status and component mix into a coarse incident-family proxy.

        Rules are evaluated in order (``score`` is ``anomaly_score`` or ``0.0``
        when ``None``):

        1. ``semantic_status == "NORMAL"`` -> ``"post_intervention_recovery"``;
        2. ``FLOCCULANT_PUMP`` and ``score >= 1.0`` -> ``"process_saturation"``;
        3. any feedback pump and ``score >= 1.0`` ->
           ``"float_recurrent_disturbance"``;
        4. any diving pump and ``score >= 1.5`` -> ``"pump_abrupt_failure"``;
        5. otherwise -> ``"external_ambiguous_disturbance"``.

        Args:
            components: Decoded component names.
            anomaly_score: Model_B anomaly score, or ``None``.
            semantic_status: ``"NORMAL"`` or ``"ANOMALOUS"``.

        Returns:
            The incident-family label.

        Notes:
            Every non-anomalous row (including rows without comparison data)
            is labelled ``"post_intervention_recovery"``. A ``NaN`` score is
            truthy, so it is kept and fails every threshold test, leading to
            ``"external_ambiguous_disturbance"`` for anomalous rows.
        """
        component_set = set(components)
        score = anomaly_score or 0.0
        if semantic_status == "NORMAL":
            return "post_intervention_recovery"
        if "FLOCCULANT_PUMP" in component_set and score >= 1.0:
            return "process_saturation"
        if component_set & {"FEEDBACK_PUMP_1", "FEEDBACK_PUMP_2"} and score >= 1.0:
            return "float_recurrent_disturbance"
        if component_set & {"DIVING_PUMP_1", "DIVING_PUMP_2"} and score >= 1.5:
            return "pump_abrupt_failure"
        return "external_ambiguous_disturbance"

    @staticmethod
    def _assign_reliability_interpretation(incident_family: str, *, semantic_status: str) -> str:
        """Translate family-level semantics into a reliability interpretation label.

        Args:
            incident_family: Label from :meth:`_assign_incident_family`.
            semantic_status: ``"NORMAL"`` or ``"ANOMALOUS"``.

        Returns:
            ``"nominal_or_stable_regime"`` for normal rows; otherwise the
            ``*_proxy`` label mapped from the family, or
            ``"undetermined_reliability_proxy"`` for unknown families.
        """
        if semantic_status == "NORMAL":
            return "nominal_or_stable_regime"
        mapping = {
            "pump_abrupt_failure": "high_severity_failure_proxy",
            "float_recurrent_disturbance": "recurrent_control_disturbance_proxy",
            "process_saturation": "load_or_capacity_stress_proxy",
            "post_intervention_recovery": "recovery_stabilization_proxy",
            "external_ambiguous_disturbance": "ambiguous_external_disturbance_proxy",
        }
        return mapping.get(incident_family, "undetermined_reliability_proxy")

    @staticmethod
    def _assign_life_regime(incident_family: str, components: tuple[str, ...], *, semantic_status: str) -> str:
        """Assign a high-level life-regime interpretation for one semantic row.

        Args:
            incident_family: Label from :meth:`_assign_incident_family`.
            components: Decoded component names.
            semantic_status: ``"NORMAL"`` or ``"ANOMALOUS"``.

        Returns:
            One of ``"nominal_operation"`` (normal rows),
            ``"degradation_acceleration"`` (``pump_abrupt_failure``),
            ``"stress_accumulation"`` (``process_saturation``),
            ``"recurrent_instability"`` (``float_recurrent_disturbance``),
            ``"idle_or_missing_signal"`` (no components) or
            ``"externally_perturbed_operation"`` (anything else).
        """
        if semantic_status == "NORMAL":
            return "nominal_operation"
        if incident_family == "pump_abrupt_failure":
            return "degradation_acceleration"
        if incident_family == "process_saturation":
            return "stress_accumulation"
        if incident_family == "float_recurrent_disturbance":
            return "recurrent_instability"
        if not components:
            return "idle_or_missing_signal"
        return "externally_perturbed_operation"

    @staticmethod
    def _assign_recovery_regime(incident_family: str, *, semantic_status: str) -> str:
        """Assign a coarse recovery regime label for downstream longitudinal use.

        Args:
            incident_family: Label from :meth:`_assign_incident_family`.
            semantic_status: ``"NORMAL"`` or ``"ANOMALOUS"``.

        Returns:
            ``"stable_recovered"`` for normal rows, ``"recovering"`` for the
            ``post_intervention_recovery`` family, ``"needs_intervention"`` for
            ``pump_abrupt_failure`` and ``"under_observation"`` otherwise.
        """
        if semantic_status == "NORMAL":
            return "stable_recovered"
        # Note: unreachable with the built-in family heuristic, which only
        # emits post_intervention_recovery for NORMAL rows (handled above).
        if incident_family == "post_intervention_recovery":
            return "recovering"
        if incident_family == "pump_abrupt_failure":
            return "needs_intervention"
        return "under_observation"

    @staticmethod
    def _parse_states(value: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
        """Normalize serialized or in-memory state words into an integer tuple.

        Args:
            value: A tuple or list of integer-like values, or a string holding
                a Python literal of an iterable of integers (e.g. ``"(1, 2)"``
                or ``"[1, 2]"``).

        Returns:
            Tuple of ``int``.

        Raises:
            ValueError: If the string is not a valid literal, or an element
                cannot be converted to ``int``.
            SyntaxError: If the string is not syntactically valid.
            TypeError: If the parsed literal is not iterable (e.g. ``"5"``).

        Examples:
            >>> SemanticModeInterpreter._parse_states("(1, 2)")
            (1, 2)
        """
        if isinstance(value, tuple):
            return tuple(int(item) for item in value)
        if isinstance(value, list):
            return tuple(int(item) for item in value)
        # Note: any other type (e.g. a NumPy array from Parquet) reaches
        # literal_eval, which only accepts strings/AST nodes.
        parsed = ast.literal_eval(value)
        return tuple(int(item) for item in parsed)
