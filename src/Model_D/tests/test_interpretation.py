"""Post hoc FIR identities, event provenance and detection isolation."""
import json

import numpy as np
import pandas as pd
import pytest

from iabm_incidents import (EventFrequencyResponseInterpreter, FrequencyInterpretationConfig,
    residual_delay_taps, residual_frequency_response, IndicatorPipeline)
from iabm_incidents.config import ModelDConfig, WindowConfig, EpisodeDetectionConfig
from test_electrical import telemetry, fitted, windows


def detected(sample, model, observer):
    residuals = observer.residuals(sample)
    w = windows(sample)
    scores = w.merge(model.score_windows(residuals, w), on=["window_id", "asset_id"])
    episodes = pd.DataFrame([dict(episode_id="e", asset_id="pump",
        event_start=w.iloc[0].start_time, event_end=w.iloc[0].end_time, source_window_ids=("0",))])
    return episodes, scores, residuals


def test_known_delay_gain_phase_and_nyquist():
    taps = np.zeros((3, 4)); taps[0, 2] = 2
    f = np.linspace(0, .5, 17)
    h = residual_frequency_response(taps, f, 1., np.array([1., 0., 0.]))
    assert np.allclose(h, 2*np.exp(-4j*np.pi*f))
    assert np.allclose(abs(h), 2)
    difference = np.zeros((3, 4)); difference[0, :2] = [1, -1]
    h = residual_frequency_response(difference, f, 1., np.array([1., 0., 0.]))
    assert h[0] == 0
    assert np.isclose(abs(h[-1]), 2)
    with pytest.raises(ValueError, match="Nyquist"):
        residual_frequency_response(taps, np.array([.6]), 1., np.ones(3))


def test_delay_taps_equal_embedded_readout():
    rng = np.random.default_rng(1)
    r, w = rng.normal(size=(50, 3)), rng.normal(size=8)
    from iabm_incidents import causal_delay
    projection = causal_delay(r, np.ones(len(r), bool)) @ w
    taps = residual_delay_taps(w)
    fir = np.array([sum(r[n-k] @ taps[:, k] for k in range(4)) for n in range(3, len(r))])
    assert np.allclose(projection, fir)


def test_explanation_localizes_impulse_and_reconstructs_fourth_order():
    observer, model = fitted()
    sample = telemetry(3)
    sample.loc[80, "RP1"] += 30
    episodes, scores, residuals = detected(sample, model, observer)
    before = scores.copy(deep=True)
    result = EventFrequencyResponseInterpreter(model).explain(episodes=episodes, window_scores=scores, residuals=residuals)
    event = result.events.iloc[0]
    assert event.status == "explained"
    assert 80 <= event.peak_sample_index <= 83
    assert event.half_height_width_seconds >= 1
    assert event.decision_available_at > event.peak_time
    assert np.isclose(event.projected_fourth_moment-event.reference_projected_fourth_moment, event.dominant_eigenvalue)
    assert np.isclose(result.taps.coefficient_energy_share.sum(), 1)
    assert set(result.frequency_response.input_direction) == {"RP1", "RP2", "RP3", "common", "differential12"}
    assert result.frequency_response.frequency_hz.max() == .5
    tap = result.taps.pivot(index="channel", columns="lag_samples", values="coefficient").loc[list(model.config.channels)].to_numpy()
    rr = residuals[["residual_1", "residual_2", "residual_3"]].to_numpy()
    for row in result.sample_projections.itertuples():
        y = sum(rr[row.sample_index-k] @ tap[:, k] for k in range(4))+event.projection_offset
        assert np.isclose(row.projection, y)
    pd.testing.assert_frame_equal(before, scores)


def test_zero_gain_phase_is_undefined_and_segments_unwrap():
    from iabm_incidents.interpretation import _phase
    h = np.array([1., 0., -1.+.1j, -1.-.1j])
    p = _phase(h, 1e-12)
    assert p[0] == 0 and np.isnan(p[1])
    assert abs(p[3]-p[2]) < 15


def test_unsupported_and_irregular_events_remain_explicit():
    observer, model = fitted()
    sample = telemetry(3); sample.RP1 += 3
    # Within electrical gap tolerance, but incompatible with nominal Bode cadence.
    sample.timestamp = sample.timestamp.min()+pd.to_timedelta(np.arange(len(sample))*1.1, unit="s")
    episodes, scores, residuals = detected(sample, model, observer)
    result = EventFrequencyResponseInterpreter(model).explain(episodes=episodes, window_scores=scores, residuals=residuals)
    assert result.events.iloc[0].status == "irregular_cadence"
    assert result.frequency_response.empty
    scores["transient_concentration"] = np.nan
    result = EventFrequencyResponseInterpreter(model).explain(episodes=episodes, window_scores=scores, residuals=residuals)
    assert result.events.iloc[0].status == "no_supported_transient_window"


def test_replay_does_not_silently_explain_changed_inputs():
    observer, model = fitted()
    sample = telemetry(3); sample.RP1 += 4
    episodes, scores, residuals = detected(sample, model, observer)
    residuals.residual_1 *= 2
    with pytest.raises(ValueError, match="replay differs"):
        EventFrequencyResponseInterpreter(model).explain(episodes=episodes, window_scores=scores, residuals=residuals)


def test_pipeline_post_hoc_does_not_change_scores_or_episodes(tmp_path):
    observer, model = fitted()
    sample = telemetry(2, n=600); sample.RP1 += 4
    seq = pd.DataFrame([dict(asset_id="pump", start_time=sample.timestamp.min(),
        end_time=sample.timestamp.max()+pd.Timedelta(seconds=1), sequence_divergence=0.)])
    assignments = pd.DataFrame([dict(semantic_status="NORMAL", incident_family="unclassified_incident")])
    cfg = ModelDConfig(detection=EpisodeDetectionConfig(minimum_duration=pd.Timedelta(seconds=10)),
        window=WindowConfig(length=pd.Timedelta(seconds=60), step=pd.Timedelta(seconds=30), min_coverage=.8))
    inputs = dict(sequences=seq, assignments=assignments, analogue=sample, digital=sample)
    plain = IndicatorPipeline(cfg, command_observer=observer, transient_model=model).run(**inputs)
    explained = IndicatorPipeline(cfg, command_observer=observer, transient_model=model,
        interpretation_config=FrequencyInterpretationConfig(frequency_points=33)).run(**inputs)
    pd.testing.assert_frame_equal(plain.window_scores, explained.window_scores)
    pd.testing.assert_frame_equal(plain.episodes, explained.episodes)
    assert plain.event_interpretation is None
    assert explained.event_interpretation.events.status.eq("explained").any()
    artifacts = explained.event_interpretation.export(tmp_path)
    metadata = json.loads(artifacts["metadata"].read_text())
    assert not metadata["reliability_score_modified"]
    assert "not pump" in metadata["meaning"]
    assert pd.read_csv(artifacts["frequency_response"]).shape[0] > 0


def test_source_window_selection_excludes_stronger_outside_evidence():
    observer, model = fitted()
    sample = telemetry(3, n=360); sample.RP1 += 3
    sample.loc[250, "RP2"] += 100
    residuals = observer.residuals(sample)
    w = pd.DataFrame([
        dict(window_id="first", asset_id="pump", start_time=sample.timestamp.iloc[0], end_time=sample.timestamp.iloc[180]),
        dict(window_id="second", asset_id="pump", start_time=sample.timestamp.iloc[180], end_time=sample.timestamp.iloc[-1]+pd.Timedelta(seconds=1)),
    ])
    scores = w.merge(model.score_windows(residuals, w), on=["window_id", "asset_id"])
    assert scores.transient_concentration.iloc[1] > scores.transient_concentration.iloc[0]
    episodes = pd.DataFrame([dict(episode_id="e", asset_id="pump", source_window_ids=("0",),
        event_start=w.start_time.min(), event_end=w.end_time.max())])
    result = EventFrequencyResponseInterpreter(model).explain(episodes=episodes, window_scores=scores, residuals=residuals)
    assert result.events.iloc[0].window_id == "first"
    assert result.sample_projections.timestamp.max() < sample.timestamp.iloc[180]


def test_experiment_exports_post_hoc_html_and_empty_tables(tmp_path):
    from iabm_incidents import ModelDExperimentRunner
    observer, model = fitted()
    sample = telemetry(2, n=600); sample.RP1 += 4
    seq = pd.DataFrame([dict(asset_id="pump", start_time=sample.timestamp.min(),
        end_time=sample.timestamp.max()+pd.Timedelta(seconds=1), sequence_divergence=0.)])
    assignments = pd.DataFrame([dict(semantic_status="NORMAL", incident_family="unclassified_incident")])
    cfg = ModelDConfig(detection=EpisodeDetectionConfig(minimum_duration=pd.Timedelta(seconds=10)),
        window=WindowConfig(length=pd.Timedelta(seconds=60), step=pd.Timedelta(seconds=30), min_coverage=.8))
    runner = ModelDExperimentRunner(cfg, command_observer=observer, transient_model=model,
        interpretation_config=FrequencyInterpretationConfig(frequency_points=17))
    result = runner.run_indicators(sequences=seq, assignments=assignments, analogue=sample, digital=sample)
    paths = runner.export_report({"indicators": result}, tmp_path)
    report = paths["indicators_interpretation_html_report"].read_text()
    assert "not pump or plant transfer functions" in report
    assert "<svg" in report
    empty = EventFrequencyResponseInterpreter(model).explain(
        episodes=pd.DataFrame(), window_scores=result.window_scores, residuals=observer.residuals(sample))
    empty_paths = empty.export(tmp_path / "empty")
    assert pd.read_csv(empty_paths["events"]).empty
    assert pd.read_csv(empty_paths["frequency_response"]).empty
