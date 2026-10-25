"""Shared fixtures for Model_D tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


@pytest.fixture
def sequences_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_time": [
                "2022-01-01 00:00:00",
                "2022-01-01 00:10:00",
                "2022-01-01 00:20:00",
                "2022-01-01 02:00:00",
                "2022-01-01 02:10:00",
                "2022-01-01 02:20:00",
            ],
            "end_time": [
                "2022-01-01 00:05:00",
                "2022-01-01 00:15:00",
                "2022-01-01 00:25:00",
                "2022-01-01 02:05:00",
                "2022-01-01 02:15:00",
                "2022-01-01 02:25:00",
            ],
            "states": ["(1, 2)", "(16, 19)", "(1, 2)", "(1, 2)", "(4, 12)", "(1, 2)"],
            "total_duration_seconds": [300.0, 300.0, 300.0, 300.0, 300.0, 300.0],
            "run_count": [2, 2, 2, 2, 2, 2],
            "nominal_match": ["(1, 2)", "(1, 2)", "(1, 2)", "(1, 2)", "(1, 2)", "(1, 2)"],
            "state_distance": [0, 1, 0, 0, 2, 0],
            "dtw_distance": [0.0, 1.0, 0.0, 0.0, 2.0, 0.0],
            "anomaly_score": [0.0, 1.2, 0.0, 0.0, 1.7, 0.0],
        }
    )


@pytest.fixture
def assignments_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "semantic_status": ["ANOMALOUS", "ANOMALOUS", "NORMAL", "ANOMALOUS", "ANOMALOUS", "NORMAL"],
            "incident_family": [
                "process_saturation",
                "float_recurrent_disturbance",
                "post_intervention_recovery",
                "float_recurrent_disturbance",
                "float_recurrent_disturbance",
                "post_intervention_recovery",
            ],
            "sequence_divergence": [0.8, 0.9, 0.1, 0.8, 0.85, 0.1],
            "duration_drift": [0.3, 0.4, 0.0, 0.3, 0.3, 0.0],
            "recurrence_excess": [0.05, 0.22, 0.0, 0.25, 0.3, 0.0],
            "persistence_excess": [0.1, 0.18, 0.0, 0.14, 0.18, 0.0],
            "consumption_deviation": [0.02, -0.05, 0.0, 0.01, 0.02, 0.0],
            "state_error_rate": [0.08, 0.09, 0.0, 0.08, 0.1, 0.0],
            "mode_divergence": [0.12, 0.2, 0.0, 0.18, 0.2, 0.0],
        }
    )


@pytest.fixture
def pump_failure_assignments_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "semantic_status": ["ANOMALOUS", "ANOMALOUS"],
            "incident_family": ["external_ambiguous_disturbance", "pump_abrupt_failure"],
            "sequence_divergence": [0.75, 0.95],
            "duration_drift": [0.1, 0.15],
            "recurrence_excess": [0.0, 0.05],
            "persistence_excess": [0.02, 0.04],
            "consumption_deviation": [-0.55, -0.7],
            "state_error_rate": [0.22, 0.3],
            "mode_divergence": [0.1, 0.12],
        }
    )


@pytest.fixture
def known_incidents_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "incident_id": ["INC-001", "INC-002"],
            "asset_id": ["WW-01", "WW-01"],
            "event_time": ["2022-01-01 00:12:00", "2022-01-01 02:12:00"],
            "documented_start": ["2022-01-01 00:10:00", "2022-01-01 02:10:00"],
            "documented_end": ["2022-01-01 00:20:00", "2022-01-01 02:20:00"],
            "recovery_time": ["2022-01-01 00:25:00", "2022-01-01 02:25:00"],
            "incident_family": ["process_saturation", "float_recurrent_disturbance"],
            "label_strength": ["confirmed", "strong"],
        }
    )


@pytest.fixture
def out_of_window_incidents_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "incident_id": ["INC-003"],
            "asset_id": ["WW-01"],
            "event_time": ["2022-01-01 05:00:00"],
            "incident_family": ["process_saturation"],
            "documented_start": ["2022-01-01 05:00:00"],
            "documented_end": ["2022-01-01 05:10:00"],
            "label_strength": ["weak"],
        }
    )


@pytest.fixture
def regime_sequences_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_time": [
                "2022-01-01 01:00:00",
                "2022-01-01 01:10:00",
                "2022-01-01 12:00:00",
                "2022-01-01 12:10:00",
            ],
            "end_time": [
                "2022-01-01 01:05:00",
                "2022-01-01 01:15:00",
                "2022-01-01 12:05:00",
                "2022-01-01 12:15:00",
            ],
            "states": ["(1, 2)", "(1, 2)", "(17, 17)", "(17, 17)"],
            "total_duration_seconds": [300.0, 300.0, 300.0, 300.0],
            "run_count": [1, 1, 4, 4],
            "nominal_match": ["(1, 2)", "(1, 2)", "(17, 17)", "(17, 17)"],
            "state_distance": [0, 0, 0, 0],
            "dtw_distance": [0.0, 0.0, 0.0, 0.0],
            "anomaly_score": [0.0, 0.0, 0.0, 0.0],
        }
    )


@pytest.fixture
def regime_assignments_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "semantic_status": ["NORMAL", "NORMAL", "NORMAL", "NORMAL"],
            "incident_family": [
                "post_intervention_recovery",
                "post_intervention_recovery",
                "post_intervention_recovery",
                "post_intervention_recovery",
            ],
            "sequence_divergence": [0.0, 0.0, 0.0, 0.0],
            "duration_drift": [0.0, 0.0, 0.0, 0.0],
            "recurrence_excess": [0.0, 0.0, 0.0, 0.0],
            "persistence_excess": [0.0, 0.0, 0.0, 0.0],
            "consumption_deviation": [0.0, 0.0, 0.0, 0.0],
            "state_error_rate": [0.0, 0.0, 0.0, 0.0],
            "mode_divergence": [0.0, 0.0, 0.0, 0.0],
        }
    )
