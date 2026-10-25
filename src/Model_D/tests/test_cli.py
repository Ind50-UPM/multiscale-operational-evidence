"""CLI smoke tests for Model_D."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import iabm_incidents.main as cli_main
from iabm_incidents.main import WorkbookValidationError


@pytest.mark.parametrize(("input_format", "output_format"), [("xlsx", "xlsx"), ("csv", "csv")])
def test_cli_generates_incident_reports(
    sequences_frame: pd.DataFrame,
    assignments_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    input_format: str,
    output_format: str,
) -> None:
    sequences_path = tmp_path / f"active_sequences.{input_format}"
    assignments_path = tmp_path / f"semantic_assignments.{input_format}"
    if input_format == "csv":
        sequences_frame.to_csv(sequences_path, index=False)
        assignments_frame.to_csv(assignments_path, index=False)
    else:
        sequences_frame.to_excel(sequences_path, index=False)
        assignments_frame.to_excel(assignments_path, index=False)

    output_dir = tmp_path / "incident_reports"
    monkeypatch.setattr(
        "sys.argv",
        [
            "industrial-incidents",
            "--sequences",
            str(sequences_path),
            "--assignments",
            str(assignments_path),
            "--output-dir",
            str(output_dir),
            "--output-format",
            output_format,
        ],
    )

    cli_main.main()

    assert (output_dir / f"window_scores.{output_format}").exists()
    assert (output_dir / f"detected_episodes.{output_format}").exists()
    assert (output_dir / f"family_assignments.{output_format}").exists()
    assert (output_dir / f"recovery_assessment.{output_format}").exists()
    assert (output_dir / f"evaluation_summary.{output_format}").exists()
    assert (output_dir / f"occurrence_summary.{output_format}").exists()
    assert (output_dir / "model_d_run_metadata.json").exists()


def test_cli_accepts_single_workbook_with_multiple_sheets(
    sequences_frame: pd.DataFrame,
    assignments_frame: pd.DataFrame,
    known_incidents_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workbook_path = tmp_path / "model_d_inputs.xlsx"
    with pd.ExcelWriter(workbook_path) as writer:
        sequences_frame.to_excel(writer, sheet_name="sequences", index=False)
        assignments_frame.to_excel(writer, sheet_name="assignments", index=False)
        known_incidents_frame.to_excel(writer, sheet_name="registry", index=False)

    output_dir = tmp_path / "incident_reports_workbook"
    monkeypatch.setattr(
        "sys.argv",
        [
            "industrial-incidents",
            "--workbook",
            str(workbook_path),
            "--output-dir",
            str(output_dir),
            "--output-format",
            "csv",
        ],
    )

    cli_main.main()

    assert (output_dir / "incident_registry_matches.csv").exists()
    assert (output_dir / "registry_evaluation_summary.csv").exists()


def test_cli_accepts_real_workbook_sheet_names(
    assignments_frame: pd.DataFrame,
    known_incidents_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workbook_path = tmp_path / "data_per_family.xlsx"
    window_indicators = pd.DataFrame(
        {
            "asset_id": ["WW-01", "WW-01"],
            "window_start": ["2022-01-01 00:00:00", "2022-01-01 01:00:00"],
            "window_end": ["2022-01-01 00:59:59", "2022-01-01 01:59:59"],
            **assignments_frame.iloc[:2].reset_index(drop=True).to_dict(orient="list"),
            "data_coverage": [0.98, 0.97],
            "sequence_count": [3, 4],
            "active_sequence_count": [2, 3],
        }
    )
    exposure = pd.DataFrame(
        {
            "asset_id": ["WW-01"],
            "observation_start": ["2022-01-01 00:00:00"],
            "observation_end": ["2022-01-31 23:59:59"],
            "excluded_start": [None],
            "excluded_end": [None],
            "exclusion_reason": [None],
        }
    )
    with pd.ExcelWriter(workbook_path) as writer:
        known_incidents_frame.to_excel(writer, sheet_name="Incident_registry", index=False)
        window_indicators.to_excel(writer, sheet_name="Window_indicators", index=False)
        exposure.to_excel(writer, sheet_name="Observation_exposure", index=False)

    output_dir = tmp_path / "incident_reports_real_workbook"
    monkeypatch.setattr(
        "sys.argv",
        [
            "industrial-incidents",
            "--workbook",
            str(workbook_path),
            "--detection-mode",
            "indicators",
            "--output-dir",
            str(output_dir),
            "--output-format",
            "csv",
        ],
    )

    cli_main.main()

    assert (output_dir / "window_scores.csv").exists()
    assert (output_dir / "incident_registry_matches.csv").exists()
    assert (output_dir / "model_d_run_metadata.json").exists()



def test_cli_runs_indicator_pipeline_from_standard_inputs(
    sequences_frame: pd.DataFrame,
    assignments_frame: pd.DataFrame,
    known_incidents_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sequences_path = tmp_path / "active_sequences.csv"
    assignments_path = tmp_path / "semantic_assignments.csv"
    registry_path = tmp_path / "registry.csv"
    analogue_path = tmp_path / "analog.csv"
    digital_path = tmp_path / "digital.csv"

    sequences_frame.to_csv(sequences_path, index=False)
    assignments_frame.to_csv(assignments_path, index=False)
    known_incidents_frame.to_csv(registry_path, index=False)
    pd.DataFrame(
        {
            "asset_id": ["WW-01", "WW-01"],
            "timestamp": ["2022-01-01 00:00:00", "2022-01-01 03:00:00"],
        }
    ).to_csv(analogue_path, index=False)
    pd.DataFrame(
        {
            "asset_id": ["WW-01", "WW-01"],
            "timestamp": ["2022-01-01 00:30:00", "2022-01-01 03:30:00"],
        }
    ).to_csv(digital_path, index=False)

    output_dir = tmp_path / "indicator_reports"
    monkeypatch.setattr(
        "sys.argv",
        [
            "industrial-incidents",
            "--detection-mode",
            "indicators",
            "--sequences",
            str(sequences_path),
            "--assignments",
            str(assignments_path),
            "--registry",
            str(registry_path),
            "--analogue",
            str(analogue_path),
            "--digital",
            str(digital_path),
            "--output-dir",
            str(output_dir),
            "--output-format",
            "csv",
        ],
    )

    cli_main.main()

    assert (output_dir / "observation_periods.csv").exists()
    assert (output_dir / "analysis_windows.csv").exists()
    assert (output_dir / "window_scores.csv").exists()
    assert (output_dir / "detected_episodes.csv").exists()

def test_cli_rejects_real_workbook_with_missing_required_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workbook_path = tmp_path / "invalid_data_per_family.xlsx"
    invalid_window_indicators = pd.DataFrame(
        {
            "asset_id": ["WW-01"],
            "window_start": ["2022-01-01 00:00:00"],
            "window_end": ["2022-01-01 00:59:59"],
            "sequence_divergence": [0.8],
        }
    )
    with pd.ExcelWriter(workbook_path) as writer:
        invalid_window_indicators.to_excel(writer, sheet_name="Window_indicators", index=False)

    output_dir = tmp_path / "invalid_reports"
    monkeypatch.setattr(
        "sys.argv",
        [
            "industrial-incidents",
            "--workbook",
            str(workbook_path),
            "--output-dir",
            str(output_dir),
            "--output-format",
            "csv",
        ],
    )

    with pytest.raises(WorkbookValidationError, match="Missing required columns"):
        cli_main.main()


def test_cli_rejects_standard_workbook_with_missing_sheet(
    sequences_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workbook_path = tmp_path / "missing_sheet.xlsx"
    with pd.ExcelWriter(workbook_path) as writer:
        sequences_frame.to_excel(writer, sheet_name="sequences", index=False)

    output_dir = tmp_path / "missing_sheet_reports"
    monkeypatch.setattr(
        "sys.argv",
        [
            "industrial-incidents",
            "--workbook",
            str(workbook_path),
            "--output-dir",
            str(output_dir),
            "--output-format",
            "csv",
        ],
    )

    with pytest.raises(WorkbookValidationError, match="Missing required workbook sheets"):
        cli_main.main()
