"""Shared fixtures for the Model_A automated test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


@pytest.fixture
def synthetic_datasets(tmp_path: Path) -> dict[str, Path]:
    """Create compact analog and digital datasets for CLI and model tests."""
    index = pd.date_range("2022-01-01 00:00:00", periods=60, freq="1min", tz="Europe/Madrid")
    states = np.array(([1] * 20) + ([2] * 20) + ([4] * 20), dtype=np.int32)

    analog_df = pd.DataFrame(
        {
            "Vrms1": 220.0 + states,
            "Vrms2": 221.0 + states,
            "Vrms3": 222.0 + states,
            "Irms1": 5.0 + (states * 0.2),
            "Irms2": 5.2 + (states * 0.2),
            "Irms3": 5.4 + (states * 0.2),
            "PF1": 0.80 + (states * 0.01),
            "PF2": 0.81 + (states * 0.01),
            "PF3": 0.82 + (states * 0.01),
            "RP1": 100.0 + (states * 5.0),
            "RP2": 102.0 + (states * 5.0),
            "RP3": 104.0 + (states * 5.0),
        },
        index=index,
    )
    digital_df = pd.DataFrame({"estado": states}, index=index)

    analog_path = tmp_path / "analog.parquet"
    digital_path = tmp_path / "digital.parquet"
    analog_df.to_parquet(analog_path)
    digital_df.to_parquet(digital_path)

    return {
        "analog_path": analog_path,
        "digital_path": digital_path,
        "start": "2022-01-01 00:00:00",
        "train_end": "2022-01-01 00:39:00",
        "predict_start": "2022-01-01 00:40:00",
        "predict_end": "2022-01-01 00:59:00",
    }
