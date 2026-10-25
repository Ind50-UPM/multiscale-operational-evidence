"""Unit tests for Model_C semantic interpretation."""

from __future__ import annotations

from iabm_semantics import SemanticModeInterpreter, SemanticRule


def test_interpret_sequences_assigns_modes(active_sequences_report) -> None:
    interpreter = SemanticModeInterpreter()
    sequences = interpreter.load_active_sequences(active_sequences_report)
    assignments = interpreter.interpret_sequences(sequences)

    assert len(assignments) == 2
    assert assignments.loc[0, "operating_mode"] == "PUMPING_MODE"
    assert assignments.loc[0, "working_mode"] == "DIVING_ONLY"
    assert assignments.loc[1, "operating_mode"] == "TREATMENT_MODE"


def test_interpret_sequences_uses_custom_rules(active_sequences_report) -> None:
    interpreter = SemanticModeInterpreter(
        rules=[
            SemanticRule(
                required_components=("FLOCCULANT_PUMP",),
                operating_mode="CUSTOM_MODE",
                working_mode="CUSTOM_WORKFLOW",
            )
        ]
    )
    sequences = interpreter.load_active_sequences(active_sequences_report)
    assignments = interpreter.interpret_sequences(sequences)

    assert assignments.loc[1, "operating_mode"] == "CUSTOM_MODE"
    assert assignments.loc[1, "working_mode"] == "CUSTOM_WORKFLOW"


def test_interpret_sequences_enriches_semantic_status(
    active_sequences_report,
    comparison_report,
) -> None:
    interpreter = SemanticModeInterpreter()
    sequences = interpreter.load_active_sequences(active_sequences_report)
    comparison = interpreter.load_comparison_report(comparison_report)
    assignments = interpreter.interpret_sequences(sequences, comparison=comparison)
    summary = interpreter.summarize_modes(assignments)

    assert assignments.loc[0, "semantic_status"] == "NORMAL"
    assert assignments.loc[1, "semantic_status"] == "ANOMALOUS"
    assert "semantic_status" in summary.columns


def test_interpret_sequences_assigns_incident_families_and_life_regimes(
    active_sequences_report,
    comparison_report,
) -> None:
    interpreter = SemanticModeInterpreter()
    sequences = interpreter.load_active_sequences(active_sequences_report)
    comparison = interpreter.load_comparison_report(comparison_report)
    assignments = interpreter.interpret_sequences(sequences, comparison=comparison)
    life_summary = interpreter.summarize_life_regimes(assignments)

    assert "incident_family" in assignments.columns
    assert "reliability_interpretation" in assignments.columns
    assert "life_regime" in assignments.columns
    assert "recovery_regime" in assignments.columns
    assert not life_summary.empty
