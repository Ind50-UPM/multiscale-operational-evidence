"""CLI smoke tests for Model_B."""

from __future__ import annotations

from pathlib import Path

import pytest

import iabm_behavior.main as cli_main


@pytest.mark.parametrize("output_format", ["xlsx", "csv"])
def test_cli_generates_reports(
    synthetic_timeline: Path,
    output_format: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The CLI should generate all expected behavioral reports."""
    output_dir = tmp_path / "reports"
    monkeypatch.setattr(
        "sys.argv",
        [
            "industrial-behavior",
            "--input",
            str(synthetic_timeline),
            "--output-dir",
            str(output_dir),
            "--smooth-short-runs",
            "--lang",
            "en",
            "--output-format",
            output_format,
        ],
    )

    cli_main.main()

    assert (output_dir / f"state_runs.{output_format}").exists()
    assert (output_dir / f"active_sequences.{output_format}").exists()
    assert (output_dir / f"sequence_words.{output_format}").exists()


@pytest.mark.parametrize("output_format", ["xlsx", "csv"])
def test_cli_generates_nominal_comparison_report(
    synthetic_timeline: Path,
    nominal_timeline: Path,
    output_format: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The CLI should write a comparison report when a nominal timeline is provided."""
    output_dir = tmp_path / "reports_with_nominal"
    monkeypatch.setattr(
        "sys.argv",
        [
            "industrial-behavior",
            "--input",
            str(synthetic_timeline),
            "--nominal-input",
            str(nominal_timeline),
            "--output-dir",
            str(output_dir),
            "--anomaly-threshold",
            "0.5",
            "--output-format",
            output_format,
        ],
    )

    cli_main.main()

    assert (output_dir / f"sequence_comparison.{output_format}").exists()
