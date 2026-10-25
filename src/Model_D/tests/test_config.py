"""Configuration tests for Model_D."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from iabm_incidents.config import ModelDConfig, load_model_d_config


def test_default_model_d_config_is_stable() -> None:
    """Default configuration should expose the expected baseline values."""

    config = ModelDConfig()

    assert config.window.length == pd.Timedelta(hours=24)
    assert config.window.step == pd.Timedelta(hours=1)
    assert config.detection.onset_windows == 2
    assert config.detection.minimum_duration == pd.Timedelta(minutes=10)
    assert config.family_assignment.minimum_confidence == 0.60
    assert config.occurrence.minimum_events_for_fit == 5


def test_load_model_d_config_from_json(tmp_path: Path) -> None:
    """Configuration loader should parse JSON overrides."""

    config_path = tmp_path / "model_d.json"
    config_path.write_text(
        json.dumps(
            {
                "window": {
                    "length_hours": 12,
                    "step_hours": 2,
                    "min_coverage": 0.95,
                    "min_active_sequences": 2,
                },
                "episode_detection": {
                    "onset_threshold": 1.4,
                    "recovery_threshold": 0.7,
                    "onset_windows": 3,
                    "recovery_windows": 2,
                    "maximum_gap_hours": 4,
                    "minimum_duration_hours": 1.5,
                },
                "weights": {
                    "sequence": 2.0,
                    "mode": 0.5,
                },
                "family_assignment": {
                    "minimum_confidence": 0.75,
                },
                "occurrence": {
                    "minimum_events_for_fit": 8,
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_model_d_config(config_path)

    assert config.window.length == pd.Timedelta(hours=12)
    assert config.window.step == pd.Timedelta(hours=2)
    assert config.window.min_coverage == 0.95
    assert config.window.min_active_sequences == 2
    assert config.detection.onset_threshold == 1.4
    assert config.detection.recovery_threshold == 0.7
    assert config.detection.onset_windows == 3
    assert config.detection.recovery_windows == 2
    assert config.detection.maximum_gap == pd.Timedelta(hours=4)
    assert config.detection.minimum_duration == pd.Timedelta(hours=1.5)
    assert config.weights.sequence == 2.0
    assert config.weights.mode == 0.5
    assert config.family_assignment.minimum_confidence == 0.75
    assert config.occurrence.minimum_events_for_fit == 8
