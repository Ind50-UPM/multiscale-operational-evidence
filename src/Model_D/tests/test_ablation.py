"""Software fixtures for strict nested ablations and evidence-aware evaluation."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from iabm_incidents import FourLevelAblationRunner, AblationConfig, VQEDirectionSolver
from iabm_incidents.ablation import BEHAVIORAL_COLUMNS, _healthy_warning_hours
from iabm_incidents.config import ModelDConfig, WindowConfig, EpisodeDetectionConfig


def fixture():
    rows=[]
    for phase,offset in [('reference',0),('calibration',24),('test',48)]:
        for i in range(12):
            start=pd.Timestamp('2022-01-01')+pd.Timedelta(hours=offset+i)
            row=dict(asset_id='a',window_id=f'{phase}_{i}',phase=phase,start_time=start,
                end_time=start+pd.Timedelta(hours=1),data_coverage=1.,electrical_status='supported')
            for j,c in enumerate(BEHAVIORAL_COLUMNS):
                row[c]=.1*(i%4)+.01*j+(i>=5 and phase=='test')*2
            row['command_response_deviation']=.2*(i%4)+(i>=4 and phase=='test')*2
            row['transient_concentration']=.3*(i%4)+(i>=3 and phase=='test')*4
            row['transient_concentration_vqe']=row['transient_concentration']*.6
            rows.append(row)
    frame=pd.DataFrame(rows)
    config=AblationConfig(minimum_phase_windows=4,risk_horizons_hours=(1,))
    model=ModelDConfig(window=WindowConfig(length=pd.Timedelta(hours=1),step=pd.Timedelta(hours=1)),
        detection=EpisodeDetectionConfig(onset_windows=1,recovery_windows=2,minimum_duration=pd.Timedelta(minutes=1)))
    return frame,FourLevelAblationRunner(config,model)


def test_four_levels_substitute_vqe_and_do_not_leak_excluded_coordinates():
    frame,runner=fixture();before=frame.copy(deep=True)
    result=runner.run(indicators=frame)
    pd.testing.assert_frame_equal(frame,before)
    assert 'command_response_deviation' not in result.levels['M0'].window_scores
    assert 'transient_concentration' not in result.levels['M1'].window_scores
    assert 'transient_concentration_vqe' not in result.levels['M3'].window_scores
    scores={k:v.window_scores.deviation_score for k,v in result.levels.items()}
    assert (scores['M0']<=scores['M1']).all()
    assert (scores['M1']<=scores['M2']).all()
    assert (scores['M3']<=scores['M2']).all()
    cal=result.calibration.set_index('level')
    assert cal.loc['M3','onset_threshold']==cal.loc['M2','onset_threshold']
    pair=result.pairwise.query("left == 'M2' and right == 'M3'").iloc[0]
    assert pair.right_only_warnings==0
    assert result.summary.maintenance_overlap_fraction.isna().all()
    assert result.summary.false_warning_hours.isna().all()


def test_missing_dimension_excluded_equally_and_phases_cannot_leak():
    frame,runner=fixture()
    frame.loc[frame.phase.eq('test').idxmax(),'mode_divergence']=np.nan
    result=runner.run(indicators=frame)
    assert all(len(level.window_scores)==11 for level in result.levels.values())
    assert result.support.common_support.sum()==35
    frame.loc[frame.phase.eq('reference'),'end_time']=pd.Timestamp('2022-01-05')
    with pytest.raises(ValueError,match='chronologically disjoint'):
        runner.run(indicators=frame)


def test_test_outliers_do_not_change_nominal_scaling_or_thresholds():
    frame,runner=fixture();first=runner.run(indicators=frame)
    frame.loc[frame.phase.eq('test'),'command_response_deviation']+=1000
    second=runner.run(indicators=frame)
    assert first.metadata['centers']==second.metadata['centers']
    assert first.metadata['scales']==second.metadata['scales']
    pd.testing.assert_frame_equal(first.calibration,second.calibration)


def test_vqe_batch_matches_single_runs_and_variational_bound():
    rng=np.random.default_rng(1);a=rng.normal(size=(3,8,8));d=(a+a.transpose(0,2,1))/2
    solver=VQEDirectionSolver(layers=2,sweeps=3,restarts=2,seed=5)
    vectors=solver.solve_batch(d)
    assert np.allclose(vectors,np.stack([solver.solve(item) for item in d]))
    assert np.allclose(np.sum(vectors*vectors,axis=1),1)
    assert (np.einsum('bi,bij,bj->b',vectors,d,vectors)<=np.linalg.eigvalsh(d)[:,-1]+1e-10).all()


def test_healthy_intervals_union_does_not_double_count():
    windows=pd.DataFrame(dict(asset_id=['a','a'],end_time=pd.to_datetime(['2022-01-01 01:00','2022-01-01 02:00']),warning=[True,False]))
    healthy=pd.DataFrame(dict(asset_id=['a','a'],start_time=pd.to_datetime(['2022-01-01 01:00']*2),end_time=pd.to_datetime(['2022-01-01 03:00']*2)))
    observed,false=_healthy_warning_hours(windows,healthy,pd.Timedelta(hours=1))
    assert observed==2 and false==1


def test_fixed_external_risk_labels_are_identical_for_every_level(tmp_path):
    frame,runner=fixture()
    refs=pd.DataFrame([dict(incident_id='r1',asset_id='a',event_time=pd.Timestamp('2022-01-02 04:30'),incident_family='pump_abrupt_failure'),
        dict(incident_id='r2',asset_id='a',event_time=pd.Timestamp('2022-01-03 04:30'),incident_family='pump_abrupt_failure')])
    result=runner.run(indicators=frame,incidents=refs,study_end=pd.Timestamp('2022-01-03 13:00'))
    targets=[v.risk_predictions.target_h1 for v in result.levels.values()]
    assert all(targets[0].equals(item) for item in targets[1:])
    assert all('weak_reference' in v.risk_metrics.iloc[0].target_provenance for v in result.levels.values())
    paths=result.export(tmp_path)
    assert paths['html'].exists()
    assert 'not independent physical-fault labels' in paths['assessment'].read_text()


def test_family_evidence_retains_real_peak_scores():
    frame,runner=fixture()
    result=runner.run(indicators=frame)
    assert result.levels['M2'].family_summary.mean_peak_score.gt(0).all()
    assert 'peak_score' in result.levels['M2'].episodes


def test_behavioral_partial_nominal_support_and_gap_mask():
    from iabm_incidents import BehavioralWindowEncoder, CommandResponseObserver, ElectricalConfig
    from iabm_incidents.behavioral import _runs
    from test_electrical import telemetry
    cfg=ElectricalConfig(minimum_samples=12,minimum_safe_fraction=.9)
    nominal=telemetry(n=360)
    encoder=BehavioralWindowEncoder(cfg).fit(nominal,pd.Timedelta(seconds=60),pd.Timedelta(seconds=30))
    observer=CommandResponseObserver(cfg).fit(nominal)
    sample=telemetry(2,n=360);sample.loc[100,'controller_state']=63
    residuals=observer.residuals(sample);residuals['pred_estado']=1.
    row=encoder.encode(residuals)
    assert np.isfinite(row['duration_drift'])
    assert row['behavioral_nominal_state_fraction']>.99
    assert row['sequence_divergence']>0
    residuals.loc[200,'controller_state']=np.nan
    masked=encoder.encode(residuals)
    assert np.isfinite(masked['duration_drift'])
    assert masked['controller_known_fraction']>.99
    states,duration,pairs=_runs(np.array([1,1,2,2]),np.array([False,False,True,False]),1.)
    assert not pairs.any()


def test_tied_auc_and_pr_scores_are_handled_as_groups():
    from iabm_incidents.ablation import _roc_auc,_average_precision
    truth=np.array([0.,1.,0.,1.]);p=np.ones(4)*.5
    assert _roc_auc(truth,p)==.5
    assert _average_precision(truth,p)==.5


def test_maintenance_outside_matched_exposure_is_unavailable_not_a_miss():
    frame,runner=fixture()
    future=pd.DataFrame([dict(asset_id='a',start_time=pd.Timestamp('2024-01-01'),end_time=pd.Timestamp('2024-01-02'))])
    result=runner.run(indicators=frame,maintenance=future)
    assert result.summary.maintenance_periods.isna().all()
    within=pd.DataFrame([dict(asset_id='a',start_time=pd.Timestamp('2022-01-03 07:00'),end_time=pd.Timestamp('2022-01-03 10:00'))])
    result=runner.run(indicators=frame,maintenance=within)
    assert result.summary.maintenance_period_hits.eq(1).all()
