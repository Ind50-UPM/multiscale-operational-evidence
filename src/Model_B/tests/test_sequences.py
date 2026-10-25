"""Unit tests for Model_B sequence analysis."""

from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from iabm_behavior import ActiveSequence, BehavioralSequenceAnalyzer


def test_extract_runs_and_sequences(synthetic_timeline) -> None:
    analyzer = BehavioralSequenceAnalyzer()
    timeline = analyzer.load_state_timeline(synthetic_timeline)

    runs = analyzer.extract_runs(timeline)
    sequences = analyzer.extract_active_sequences(timeline)
    summary = analyzer.summarize_sequence_words(sequences)

    assert len(runs) == 6
    assert runs[1].state == 1
    assert runs[2].state == 2
    assert len(sequences) == 2
    assert sequences[0].states == (1, 2)
    assert sequences[1].states == (4,)
    assert list(summary["count"]) == [1, 1]


def test_smooth_short_runs_merges_transients() -> None:
    analyzer = BehavioralSequenceAnalyzer()
    timeline = pd.DataFrame(
        {"Predicted_State": [0, 1, 2, 2, 0]},
        index=pd.date_range("2022-01-01", periods=5, freq="500ms"),
    )

    smoothed = analyzer.smooth_short_runs(
        timeline,
        min_duration_seconds=1.0,
        min_samples=1,
    )

    assert smoothed["Predicted_State"].tolist() == [0, 2, 2, 2, 0]


def test_compare_to_nominal_quantifies_sequence_differences() -> None:
    analyzer = BehavioralSequenceAnalyzer()
    nominal_sequences = [
        ActiveSequence(
            start_time=pd.Timestamp("2022-01-01 00:00:00"),
            end_time=pd.Timestamp("2022-01-01 00:00:03"),
            states=(1, 2),
            total_duration_seconds=3.0,
            run_count=2,
        )
    ]
    observed_sequences = [
        ActiveSequence(
            start_time=pd.Timestamp("2022-01-02 00:00:00"),
            end_time=pd.Timestamp("2022-01-02 00:00:03"),
            states=(1, 4),
            total_duration_seconds=3.0,
            run_count=2,
        )
    ]

    nominal_reference = analyzer.build_nominal_reference(nominal_sequences)
    comparison = analyzer.compare_to_nominal(observed_sequences, nominal_reference)

    assert len(comparison) == 1
    assert comparison.loc[0, "observed_states"] == (1, 4)
    assert comparison.loc[0, "nominal_states"] == (1, 2)
    assert comparison.loc[0, "state_distance"] >= 1
    assert comparison.loc[0, "dtw_distance"] >= comparison.loc[0, "state_distance"]
    assert comparison.loc[0, "anomaly_score"] >= comparison.loc[0, "dtw_distance"]
    assert bool(comparison.loc[0, "is_anomalous"]) is True


def test_compare_to_nominal_accepts_exact_nominal_sequence() -> None:
    analyzer = BehavioralSequenceAnalyzer()
    nominal_sequences = [
        ActiveSequence(
            start_time=pd.Timestamp("2022-01-01 00:00:00"),
            end_time=pd.Timestamp("2022-01-01 00:00:03"),
            states=(1, 2),
            total_duration_seconds=3.0,
            run_count=2,
        )
    ]
    observed_sequences = [
        ActiveSequence(
            start_time=pd.Timestamp("2022-01-02 00:00:00"),
            end_time=pd.Timestamp("2022-01-02 00:00:03"),
            states=(1, 2),
            total_duration_seconds=3.0,
            run_count=2,
        )
    ]

    nominal_reference = analyzer.build_nominal_reference(nominal_sequences)
    comparison = analyzer.compare_to_nominal(
        observed_sequences,
        nominal_reference,
        anomaly_threshold=0.5,
    )

    assert bool(comparison.loc[0, "exact_match"]) is True
    assert comparison.loc[0, "dtw_distance"] == 0.0
    assert comparison.loc[0, "duration_ratio_delta"] == 0.0
    assert bool(comparison.loc[0, "is_anomalous"]) is False


def test_longitudinal_metrics_recurrence_and_recovery_are_reproducible() -> None:
    analyzer = BehavioralSequenceAnalyzer()
    nominal_sequences = [
        ActiveSequence(
            start_time=pd.Timestamp("2022-01-01 00:00:00"),
            end_time=pd.Timestamp("2022-01-01 00:10:00"),
            states=(1, 2),
            total_duration_seconds=600.0,
            run_count=2,
        )
    ]
    observed_sequences = [
        ActiveSequence(
            start_time=pd.Timestamp("2022-01-02 00:00:00"),
            end_time=pd.Timestamp("2022-01-02 00:10:00"),
            states=(1, 2),
            total_duration_seconds=600.0,
            run_count=2,
        ),
        ActiveSequence(
            start_time=pd.Timestamp("2022-01-02 01:00:00"),
            end_time=pd.Timestamp("2022-01-02 01:20:00"),
            states=(1, 4),
            total_duration_seconds=1200.0,
            run_count=2,
        ),
        ActiveSequence(
            start_time=pd.Timestamp("2022-01-02 02:00:00"),
            end_time=pd.Timestamp("2022-01-02 02:10:00"),
            states=(1, 2),
            total_duration_seconds=600.0,
            run_count=2,
        ),
    ]

    nominal_reference = analyzer.build_nominal_reference(nominal_sequences)
    longitudinal = analyzer.compute_longitudinal_metrics(
        observed_sequences,
        nominal_reference,
        anomaly_threshold=0.5,
    ).reset_index(drop=True)
    recovery = analyzer.compute_recovery_metrics(
        observed_sequences,
        nominal_reference,
        anomaly_threshold=0.5,
    ).reset_index(drop=True)

    expected_longitudinal = pd.DataFrame(
        {
            "sequence_index": [0, 1, 2],
            "states": [(1, 2), (1, 4), (1, 2)],
            "recurrence_interval_seconds": [None, None, 7200.0],
            "sequence_persistence_ratio": [300.0, 600.0, 300.0],
            "over_activation_rate": [1.0, 1.0, 1.0],
            "transition_instability": [0.0, 0.5, 0.5],
            "duration_drift": [0.0, 1.0, 0.0],
            "rolling_divergence": [0.0, 1.0, 2.0 / 3.0],
            "nominal_match": [(1, 2), (1, 2), (1, 2)],
            "anomaly_flag": [False, True, False],
        }
    )
    expected_recovery = pd.DataFrame(
        {
            "anomaly_sequence_index": [1],
            "recovered_sequence_index": [2],
            "time_to_recovery_seconds": [3000.0],
            "partial_recovery_score": [1.0],
            "post_intervention_regime_shift": [True],
        }
    )

    assert_frame_equal(longitudinal, expected_longitudinal, check_dtype=False)
    assert_frame_equal(recovery, expected_recovery, check_dtype=False)


def test_family_signature_classification_is_rule_based_and_stable() -> None:
    analyzer = BehavioralSequenceAnalyzer()
    longitudinal_metrics = pd.DataFrame(
        {
            "sequence_index": [0, 1, 2],
            "states": [(1, 4), (16, 19), (1, 2)],
            "recurrence_interval_seconds": [pd.NA, pd.NA, 3600.0],
            "sequence_persistence_ratio": [300.0, 300.0, 300.0],
            "over_activation_rate": [1.0, 1.0, 1.0],
            "transition_instability": [0.8, 0.2, 0.5],
            "duration_drift": [0.2, 0.6, -0.5],
            "rolling_divergence": [1.1, 1.2, 0.9],
            "nominal_match": [(1, 2), (16, 3), (1, 2)],
            "anomaly_flag": [True, True, True],
        }
    )

    signatures = analyzer.summarize_family_signatures(longitudinal_metrics).reset_index(drop=True)
    expected = pd.DataFrame(
        {
            "incident_family_candidate": [
                "recurrent_spurious_cycle",
                "pump_abrupt_failure",
                "process_saturation",
            ],
            "signature_word": [(1, 4), (1, 2), (16, 19)],
            "count": [1, 1, 1],
            "mean_transition_instability": [0.8, 0.5, 0.2],
            "mean_duration_drift": [0.2, -0.5, 0.6],
        }
    )

    assert_frame_equal(signatures, expected, check_dtype=False)
