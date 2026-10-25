"""Tests for indicator-driven helper modules in Model_D."""

from __future__ import annotations

import pandas as pd

from iabm_incidents.exposure import ExposureConfig, derive_observation_periods
from iabm_incidents.indicators import compute_window_indicators, fit_indicator_baseline
from iabm_incidents.windows import WindowBuildConfig, build_rolling_windows


def test_derive_observation_periods_includes_registry_exclusions() -> None:
    analogue = pd.DataFrame(
        {
            "asset_id": ["WW-01", "WW-01"],
            "timestamp": ["2022-01-01 00:00:00", "2022-01-01 06:00:00"],
        }
    )
    digital = pd.DataFrame(
        {
            "asset_id": ["WW-01", "WW-01"],
            "timestamp": ["2022-01-01 00:30:00", "2022-01-01 05:30:00"],
        }
    )
    registry = pd.DataFrame(
        {
            "incident_id": ["INC-1"],
            "asset_id": ["WW-01"],
            "downtime_start": ["2022-01-01 02:00:00"],
            "downtime_end": ["2022-01-01 03:00:00"],
            "maintenance_time": ["2022-01-01 04:00:00"],
        }
    )

    periods = derive_observation_periods(
        analogue,
        digital,
        registry=registry,
        config=ExposureConfig(gap_tolerance=pd.Timedelta(hours=8)),
    )

    assert periods["period_type"].eq("observed").any()
    assert periods["reason"].eq("documented_downtime").any()
    assert periods["reason"].eq("documented_maintenance").any()


def test_build_rolling_windows_respects_excluded_exposure() -> None:
    periods = pd.DataFrame(
        {
            "asset_id": ["WW-01", "WW-01"],
            "start_time": ["2022-01-01 00:00:00", "2022-01-01 03:00:00"],
            "end_time": ["2022-01-01 06:00:00", "2022-01-01 04:00:00"],
            "period_type": ["observed", "excluded"],
            "exclude_from_exposure": [False, True],
        }
    )

    windows = build_rolling_windows(
        periods,
        config=WindowBuildConfig(length=pd.Timedelta(hours=2), step=pd.Timedelta(hours=1), min_coverage=0.5),
    )

    assert not windows.empty
    assert windows["data_coverage"].between(0.0, 1.0).all()
    assert windows["is_valid"].any()


def test_compute_window_indicators_centers_on_nominal_baseline() -> None:
    source_rows = pd.DataFrame(
        {
            "asset_id": ["WW-01", "WW-01", "WW-01"],
            "start_time": ["2022-01-01 00:00:00", "2022-01-01 01:00:00", "2022-01-01 02:00:00"],
            "end_time": ["2022-01-01 00:59:59", "2022-01-01 01:59:59", "2022-01-01 02:59:59"],
            "semantic_status": ["NORMAL", "ANOMALOUS", "ANOMALOUS"],
            "incident_family": ["post_intervention_recovery", "process_saturation", "process_saturation"],
            "sequence_divergence": [0.1, 0.8, 0.9],
            "duration_drift": [0.0, 0.3, 0.4],
            "recurrence_excess": [0.0, 0.2, 0.25],
            "persistence_excess": [0.0, 0.15, 0.2],
            "consumption_deviation": [0.0, 0.1, 0.15],
            "state_error_rate": [0.0, 0.08, 0.1],
            "mode_divergence": [0.0, 0.12, 0.15],
        }
    )
    windows = pd.DataFrame(
        {
            "window_id": ["w1"],
            "asset_id": ["WW-01"],
            "start_time": ["2022-01-01 00:00:00"],
            "end_time": ["2022-01-01 03:00:00"],
            "data_coverage": [1.0],
        }
    )

    baseline = fit_indicator_baseline(source_rows)
    indicators = compute_window_indicators(source_rows, windows, baseline)

    assert indicators.loc[0, "incident_family"] == "process_saturation"
    assert indicators.loc[0, "semantic_status"] == "ANOMALOUS"
    assert indicators.loc[0, "sequence_divergence"] > 0.0
