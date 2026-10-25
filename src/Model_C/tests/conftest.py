"""Shared fixtures for Model_C tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


@pytest.fixture
def active_sequences_report(tmp_path: Path) -> Path:
    """Create a synthetic Model_B active-sequence report."""
    frame = pd.DataFrame(
        {
            "start_time": ["2022-01-01 00:00:00", "2022-01-01 01:00:00"],
            "end_time": ["2022-01-01 00:00:10", "2022-01-01 01:00:20"],
            "states": ["(1, 2)", "(16, 19)"],
            "total_duration_seconds": [10.0, 20.0],
            "run_count": [2, 2],
        }
    )
    file_path = tmp_path / "active_sequences.xlsx"
    frame.to_excel(file_path, index=False)
    return file_path


@pytest.fixture
def comparison_report(tmp_path: Path) -> Path:
    """Create a synthetic Model_B comparison report."""
    frame = pd.DataFrame(
        {
            "observed_states": ["(1, 2)", "(16, 19)"],
            "nominal_states": ["(1, 2)", "(16, 3)"],
            "exact_match": [True, False],
            "state_distance": [0, 1],
            "dtw_distance": [0.0, 1.0],
            "duration_ratio_delta": [0.0, 0.5],
            "anomaly_score": [0.0, 1.5],
            "is_anomalous": [False, True],
        }
    )
    file_path = tmp_path / "sequence_comparison.xlsx"
    frame.to_excel(file_path, index=False)
    return file_path
