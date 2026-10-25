"""Scientific identities, support boundaries and Model D electrical integration."""
import numpy as np
import pandas as pd
import pytest

from iabm_incidents import (ElectricalConfig, CommandResponseObserver,
    TransientConcentration, VQEDirectionSolver, causal_delay, fourth_scatter,
    IndicatorPipeline)
from iabm_incidents.config import ModelDConfig, WindowConfig, DeviationWeights, EpisodeDetectionConfig
from iabm_incidents.detection import compute_deviation_score


def telemetry(seed=0, n=180, start="2022-07-01"):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(10, .2, (n, 3)), columns=["RP1", "RP2", "RP3"])
    frame["timestamp"] = pd.date_range(start, periods=n, freq="s")
    frame["controller_state"] = 1
    frame["asset_id"] = "pump"
    return frame


def fitted():
    cfg = ElectricalConfig(minimum_samples=12, minimum_safe_fraction=.8)
    observer = CommandResponseObserver(cfg).fit(telemetry())
    model = TransientConcentration(cfg).fit(observer.residuals(telemetry(1, start="2022-07-02")))
    return observer, model


def windows(frame):
    return pd.DataFrame([dict(window_id="w", asset_id="pump", start_time=frame.timestamp.min(),
        end_time=frame.timestamp.max()+pd.Timedelta(seconds=1))])


def test_fourth_order_rayleigh_and_scaling():
    z = np.random.default_rng(1).normal(size=(100, 8))
    v = np.random.default_rng(2).normal(size=8); v /= np.linalg.norm(v)
    f = fourth_scatter(z)
    assert np.allclose(f, f.T)
    assert np.linalg.eigvalsh(f).min() >= -1e-12
    assert np.isclose(v @ f @ v, np.mean(np.sum(z*z, axis=1)/8*(z @ v)**2))
    assert np.allclose(fourth_scatter(2*z), 16*f)


def test_equal_second_moments_different_fourth_order():
    a = np.array([[-1.], [-1.], [1.], [1.]])
    b = np.array([[-np.sqrt(2)], [0.], [0.], [np.sqrt(2)]])
    assert np.allclose(a.T @ a, b.T @ b)
    assert not np.allclose(fourth_scatter(a), fourth_scatter(b))


def test_causal_embedding_masks_every_source_sample():
    r = np.arange(30).reshape(10, 3)
    safe = np.ones(10, bool); safe[4] = False
    x = causal_delay(r, safe)
    assert len(x) == 3
    assert np.allclose(x[0, :6], np.r_[r[3], r[2]])
    assert np.allclose(x[0, 6:], [r[1].sum()/np.sqrt(3), r[0].sum()/np.sqrt(3)])


def test_signed_response_unknown_states_and_reference_frozen():
    observer, model = fitted()
    sample = telemetry(2, start="2022-08-01")
    sample.RP1 -= 2
    sample.RP2 += 2
    residuals = observer.residuals(sample)
    before = model.references[("pump", 1)].scatter.copy()
    result = model.score_windows(residuals, windows(sample)).iloc[0]
    assert result.command_response_deviation > 5
    assert result.command_response_signed_RP1 < 0 < result.command_response_signed_RP2
    assert result.transient_concentration > 0
    assert np.array_equal(before, model.references[("pump", 1)].scatter)
    sample.controller_state = 99
    result = model.score_windows(observer.residuals(sample), windows(sample)).iloc[0]
    assert result.electrical_status == "unsupported"
    assert np.isnan(result.command_response_deviation)
    assert np.isnan(result.transient_concentration)


def test_gaps_and_assets_do_not_mix():
    observer, model = fitted()
    sample = telemetry(2)
    sample.loc[90:, "timestamp"] += pd.Timedelta(hours=1)
    residuals = observer.residuals(sample)
    assert not residuals.loc[90:92, "safe"].any()
    other = telemetry(3); other.asset_id = "other"
    assert not observer.residuals(other).safe.any()
    with pytest.raises(ValueError, match="Duplicate"):
        observer.residuals(pd.concat([sample, sample]))


def test_vqe_is_audited_separately_on_identical_operator():
    observer, model = fitted()
    model.solver = VQEDirectionSolver(layers=2, sweeps=3)
    sample = telemetry(2, start="2022-08-01")
    sample.loc[80, "RP1"] += 20
    result = model.score_windows(observer.residuals(sample), windows(sample)).iloc[0]
    assert 0 <= result.transient_concentration_vqe <= result.transient_concentration+1e-10
    assert result.transient_solver_deficit >= -1e-10
    model.solver = None
    exact = model.score_windows(observer.residuals(sample), windows(sample)).iloc[0]
    assert exact.transient_concentration == result.transient_concentration
    d = np.diag(np.arange(8.))
    v = VQEDirectionSolver().solve(d)
    assert np.isclose(v @ v, 1)
    assert v @ d @ v > 6.9


def test_extended_weights_and_pipeline_episode_features():
    observer, model = fitted()
    sample = telemetry(2, n=600, start="2022-08-01"); sample.RP1 += 4
    seq = pd.DataFrame([dict(asset_id="pump", start_time=sample.timestamp.min(),
        end_time=sample.timestamp.max()+pd.Timedelta(seconds=1), sequence_divergence=0.)])
    assignments = pd.DataFrame([dict(semantic_status="NORMAL", incident_family="unclassified_incident")])
    cfg = ModelDConfig(detection=EpisodeDetectionConfig(minimum_duration=pd.Timedelta(seconds=10)), window=WindowConfig(length=pd.Timedelta(seconds=60),
        step=pd.Timedelta(seconds=30), min_coverage=.8))
    result = IndicatorPipeline(cfg, command_observer=observer, transient_model=model).run(
        sequences=seq, assignments=assignments, analogue=sample, digital=sample)
    assert result.window_scores.command_response_deviation.notna().any()
    assert result.window_scores.transient_concentration.notna().any()
    assert not result.episodes.empty
    assert "max_transient_concentration" in result.episodes
    frame = pd.DataFrame(dict(command_response_deviation=[2], transient_concentration=[3]))
    assert compute_deviation_score(frame, DeviationWeights(command_response=2, transient_concentration=3)).iloc[0] == 13


def test_no_transient_excess_is_clipped_and_sparse_coverage_abstains():
    observer, model = fitted()
    sample = telemetry(2)
    residuals = observer.residuals(sample)
    residuals[["residual_1", "residual_2", "residual_3"]] = 0.
    # Zero-centered coordinates and identity nominal scatter give negative D.
    from iabm_incidents import TransientReference
    reference = TransientReference(np.zeros(8), np.eye(8), np.eye(8), 1.)
    model.references[("pump", 1)] = reference
    result = model.score_windows(residuals, windows(sample)).iloc[0]
    assert result.transient_concentration == 0
    sparse = windows(sample)
    sparse["end_time"] += pd.Timedelta(hours=1)
    result = model.score_windows(residuals, sparse).iloc[0]
    assert result.electrical_status == "unsupported"
    assert np.isnan(result.transient_concentration)


def test_precomputed_electrical_scores_are_not_recentered():
    from iabm_incidents import compute_window_indicators
    frame = pd.DataFrame(dict(asset_id=["pump", "pump"],
        start_time=pd.to_datetime(["2022-01-01", "2022-01-01 01:00"], format="mixed"),
        end_time=pd.to_datetime(["2022-01-01 01:00", "2022-01-01 02:00"]),
        command_response_deviation=[2., 4.], transient_concentration=[3., 7.],
        command_response_signed_RP1=[-1., -3.]))
    w = pd.DataFrame([dict(window_id="w", asset_id="pump", start_time=frame.start_time.min(), end_time=frame.end_time.max())])
    row = compute_window_indicators(frame, w).iloc[0]
    assert row.command_response_deviation == 3
    assert row.transient_concentration == 7
    assert row.command_response_signed_RP1 == -2


def test_electrical_weight_json_roundtrip(tmp_path):
    import json
    from iabm_incidents.config import load_model_d_config
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"weights": {"command_response": 0., "transient_concentration": 2.5}}))
    cfg = load_model_d_config(path)
    assert cfg.to_dict()["weights"]["command_response"] == 0
    assert cfg.to_dict()["weights"]["transient_concentration"] == 2.5
