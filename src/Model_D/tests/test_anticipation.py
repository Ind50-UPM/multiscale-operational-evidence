from __future__ import annotations

from pathlib import Path

import pandas as pd

from iabm_incidents.anticipation import (
    AnticipationConfig,
    AnticipatoryMaintenanceForecaster,
    build_anticipation_frame,
    temporal_holdout_split,
)


def _window_scores_frame() -> pd.DataFrame:
    rows = []
    for idx in range(12):
        ts = pd.Timestamp("2022-01-01 00:00:00") + pd.Timedelta(hours=idx)
        rows.append(
            {
                "asset_id": "WW-01",
                "start_time": ts,
                "end_time": ts + pd.Timedelta(hours=1),
                "semantic_status": "ANOMALOUS" if idx in {3, 4, 8, 9} else "NORMAL",
                "incident_family": "process_saturation" if idx in {3, 4, 8, 9} else "post_intervention_recovery",
                "deviation_score": 0.2 + 0.1 * idx,
                "sequence_divergence": 0.05 * idx,
                "duration_drift": 0.02 * idx,
                "recurrence_excess": 0.01 * idx,
                "persistence_excess": 0.015 * idx,
                "consumption_deviation": -0.05 if idx >= 8 else 0.01 * idx,
                "state_error_rate": 0.02 * idx,
                "mode_divergence": 0.01 * idx,
                "state_word_diversity": 1 + (idx % 3),
                "dominant_state_word_fraction": 0.9 - 0.03 * min(idx, 6),
                "state_word_transition_rate": 0.05 * (idx % 4),
                "nominal_state_word_match_fraction": max(0.0, 0.95 - 0.08 * idx),
                "mean_state_distance": 0.1 * idx,
                "mean_dtw_distance": 0.12 * idx,
                "mean_nominal_anomaly_score": 0.04 * idx,
                "rare_word_fraction": 0.02 * (idx % 5),
                "rare_state_fraction": 0.01 * (idx % 4),
                "state_entropy": 0.2 * idx,
                "state_17_fraction": 0.0 if idx < 8 else 0.08,
                "off_nominal_state_fraction": 0.01 * idx,
                "word_regime_shift_score": min(0.1 * idx, 0.99),
            }
        )
    return pd.DataFrame(rows)


def _episodes_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_id": ["ep-1", "ep-2"],
            "asset_id": ["WW-01", "WW-01"],
            "event_start": ["2022-01-01 05:30:00", "2022-01-01 10:30:00"],
            "event_end": ["2022-01-01 07:30:00", "2022-01-01 11:30:00"],
            "primary_family": ["process_saturation", "pump_abrupt_failure"],
        }
    )


def test_build_anticipation_frame_creates_horizon_targets() -> None:
    frame = build_anticipation_frame(
        window_scores=_window_scores_frame(),
        episodes=_episodes_frame(),
        horizons_hours=(24, 72),
        target_families=("process_saturation", "pump_abrupt_failure"),
    )

    assert "feat_deviation_score" in frame.columns
    assert "feat_deviation_score_mean_6h" in frame.columns
    assert "target_h24" in frame.columns
    assert "target_h72" in frame.columns
    assert frame["target_h24"].sum() > 0


def test_anticipatory_forecaster_produces_validation_and_actions(tmp_path: Path) -> None:
    window_scores = _window_scores_frame()
    episodes = _episodes_frame()
    forecaster = AnticipatoryMaintenanceForecaster(
        AnticipationConfig(horizons_hours=(24, 72), risk_threshold=0.5, warning_threshold=0.3)
    )

    result = forecaster.run(window_scores=window_scores, episodes=episodes)
    written = forecaster.export_report(result, tmp_path / "anticipation")

    assert not result.scored_windows.empty
    assert {"risk_h24", "risk_h72", "max_risk"}.issubset(result.scored_windows.columns)
    assert not result.validation_summary.empty
    assert {"maintenance_action", "maintenance_priority"}.issubset(result.maintenance_actions.columns)
    assert written["html_report"].exists()
    assert (tmp_path / "anticipation" / "scored_windows.csv").exists()


def test_temporal_holdout_split_preserves_order() -> None:
    frame = build_anticipation_frame(
        window_scores=_window_scores_frame(),
        episodes=_episodes_frame(),
        horizons_hours=(24,),
        target_families=("process_saturation",),
    )
    train, test = temporal_holdout_split(frame, 0.25)

    assert not train.empty
    assert not test.empty
    assert train["window_end"].max() <= test["window_end"].min()
