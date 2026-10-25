"""Tests for experimental evaluation utilities in Model_D."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from iabm_incidents.experimental import ModelDExperimentRunner


def test_experiment_runner_exports_comparison_report(
    sequences_frame: pd.DataFrame,
    assignments_frame: pd.DataFrame,
    known_incidents_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    runner = ModelDExperimentRunner()
    analogue = pd.DataFrame(
        {
            "asset_id": ["WW-01", "WW-01"],
            "timestamp": ["2022-01-01 00:00:00", "2022-01-01 03:00:00"],
        }
    )
    digital = pd.DataFrame(
        {
            "asset_id": ["WW-01", "WW-01"],
            "timestamp": ["2022-01-01 00:30:00", "2022-01-01 03:30:00"],
        }
    )

    results = runner.run_comparison(
        sequences=sequences_frame,
        assignments=assignments_frame,
        registry=known_incidents_frame,
        analogue=analogue,
        digital=digital,
    )
    written = runner.export_report(results, tmp_path / "experiment")

    assert set(results.keys()) == {"semantic", "indicators"}
    assert "comparison_summary" in written
    assert written["html_report"].exists()
    assert (tmp_path / "experiment" / "semantic" / "evaluation_summary.csv").exists()
    assert (tmp_path / "experiment" / "indicators" / "family_summary.csv").exists()
