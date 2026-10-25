"""Shared fixtures for Model_B tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


@pytest.fixture
def synthetic_timeline(tmp_path: Path) -> Path:
    """Create a simple state timeline suitable for sequence-analysis tests."""
    index = pd.date_range("2022-01-01 00:00:00", periods=8, freq="1s")
    frame = pd.DataFrame(
        {
            "Time": index,
            "Predicted_State": [0, 1, 1, 2, 0, 4, 4, 0],
        }
    )
    file_path = tmp_path / "predictions.xlsx"
    frame.to_excel(file_path, index=False)
    return file_path


@pytest.fixture
def nominal_timeline(tmp_path: Path) -> Path:
    """Create a nominal reference timeline for anomaly-comparison tests."""
    index = pd.date_range("2022-01-01 00:00:00", periods=8, freq="1s")
    frame = pd.DataFrame(
        {
            "Time": index,
            "Predicted_State": [0, 1, 1, 2, 2, 0, 4, 4],
        }
    )
    file_path = tmp_path / "nominal_predictions.xlsx"
    frame.to_excel(file_path, index=False)
    return file_path
