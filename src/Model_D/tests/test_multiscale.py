"""Prespecified-scale controls and matched-outcome invariants (software fixtures)."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from iabm_incidents import (MultiscaleSensitivityConfig, MultiscaleSensitivityRunner, AblationConfig)
from iabm_incidents.ablation import BEHAVIORAL_COLUMNS


def fixtures():
    ends = pd.date_range('2022-01-03', '2022-01-09 02:45', freq='15min')
    frames = {}
    for minutes in (15, 60, 360, 1440):
        rows = []
        for i,end in enumerate(ends):
            phase = 'excluded_boundary'
            for name,date in [('reference','2022-01-03'),('calibration','2022-01-06'),('test','2022-01-09')]:
                if pd.Timestamp(date) <= end < pd.Timestamp(date)+pd.Timedelta(hours=3):
                    phase = name
            local = i%12
            row = dict(asset_id='a',window_id=f'{minutes}_{i}',start_time=end-pd.Timedelta(minutes=minutes),
                end_time=end,phase=phase,data_coverage=1.,electrical_status='supported')
            for j,column in enumerate(BEHAVIORAL_COLUMNS):
                row[column] = (.1*(local%4)+.01*j+2*(phase == 'test' and local>=5))/np.sqrt(minutes)
            row['command_response_deviation'] = (.2*(local%4)+2*(phase == 'test' and local>=4))/np.sqrt(minutes)
            row['transient_concentration'] = (.3*(local%4)+4*(phase == 'test' and local>=3))/np.sqrt(minutes)
            row['transient_concentration_vqe'] = row['transient_concentration']*.7
            rows.append(row)
        frames[minutes] = pd.DataFrame(rows)
    config = MultiscaleSensitivityConfig(risk_holdout_start=pd.Timestamp('2022-01-08'))
    runner = MultiscaleSensitivityRunner(config,ablation_config=AblationConfig(minimum_phase_windows=3,risk_horizons_hours=(1,)))
    refs = pd.DataFrame([dict(incident_id='r1',asset_id='a',event_time=pd.Timestamp('2022-01-06 01:30'),incident_family='pump_abrupt_failure'),
        dict(incident_id='r2',asset_id='a',event_time=pd.Timestamp('2022-01-09 01:30'),incident_family='pump_abrupt_failure')])
    return frames,runner,refs


def test_one_shared_support_intersection_and_unmodified_input():
    frames,runner,refs = fixtures()
    missing = frames[15].index[frames[15].end_time.eq(pd.Timestamp('2022-01-09 01:00'))][0]
    frames[15].loc[missing,'mode_divergence'] = np.nan
    before = {key:value.copy(deep=True) for key,value in frames.items()}
    result = runner.run(indicators_by_scale=frames,incidents=refs,study_end=pd.Timestamp('2022-01-09 05:00'))
    assert len(result.summary) == 16
    assert result.summary.test_windows.eq(11).all()
    assert result.metadata['common_phase_endpoints']['test'] == 11
    for minutes in frames:
        pd.testing.assert_frame_equal(frames[minutes],before[minutes])
    available = result.availability[result.availability.phase.eq('test')]
    assert available.cross_scale_support.sum() == 44
    assert available.own_support.sum() == 47
    for scale in result.scales.values():
        pair = scale.pairwise.query("left == 'M2' and right == 'M3'").iloc[0]
        assert pair.right_only_warnings == 0


def test_fixed_risk_labels_and_purge_are_identical_across_scales():
    frames,runner,refs = fixtures()
    result = runner.run(indicators_by_scale=frames,incidents=refs,study_end=pd.Timestamp('2022-01-09 05:00'))
    base = result.scales[15].levels['M0'].risk_predictions
    for scale in result.scales.values():
        for level in scale.levels.values():
            assert level.risk_predictions.window_end.equals(base.window_end)
            assert level.risk_predictions.target_h1.equals(base.target_h1)
        assert scale.metadata['risk_holdout_start'] == pd.Timestamp('2022-01-08')
        assert scale.metadata['model_d_config']['window']['step'] == 'P0DT0H15M0S'
    counts = result.risk_metrics.groupby('horizon_hours')[['train_windows','test_windows']].nunique()
    assert counts.eq(1).all().all()


def test_reject_mixed_endpoint_grid_and_unplanned_scale():
    frames,runner,refs = fixtures()
    frames[60][['start_time','end_time']] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError,match='endpoints'):
        runner.run(indicators_by_scale=frames,incidents=refs)
    frames.pop(60)
    with pytest.raises(ValueError,match='prespecified'):
        runner.run(indicators_by_scale=frames,incidents=refs)


def test_scales_and_step_do_not_allow_short_window_blind_spots():
    with pytest.raises(ValueError,match='cadence'):
        MultiscaleSensitivityConfig(decision_step_minutes=60)
    with pytest.raises(ValueError,match='Unique'):
        MultiscaleSensitivityConfig(windows_minutes=(15,15))


def test_held_out_change_cannot_modify_scale_calibration(tmp_path):
    frames,runner,refs = fixtures()
    first = runner.run(indicators_by_scale=frames,incidents=refs,study_end=pd.Timestamp('2022-01-09 05:00'))
    frames[15].loc[frames[15].phase.eq('test'),'command_response_deviation'] += 100
    second = runner.run(indicators_by_scale=frames,incidents=refs,study_end=pd.Timestamp('2022-01-09 05:00'))
    for minutes in frames:
        pd.testing.assert_frame_equal(first.scales[minutes].calibration,second.scales[minutes].calibration)
    paths = first.export(tmp_path)
    assert paths['html'].exists()
    assert 'primary 24-hour/hourly experiment is retained unchanged' in paths['assessment'].read_text()


def test_long_horizon_purge_uses_declared_boundary_not_shorter_window_start():
    frames,runner,refs = fixtures()
    runner = MultiscaleSensitivityRunner(runner.config,
        ablation_config=AblationConfig(minimum_phase_windows=3,risk_horizons_hours=(48,)))
    result = runner.run(indicators_by_scale=frames,incidents=refs,study_end=pd.Timestamp('2022-01-12'))
    # Only the first calibration endpoint finishes its 48-h target before Jan 8.
    assert result.risk_metrics.train_windows.eq(1).all()
    assert result.risk_metrics.test_windows.eq(12).all()


def test_retrospective_reference_cohort_does_not_change_future_target_registry():
    frames,runner,refs=fixtures()
    result=runner.run(indicators_by_scale=frames,incidents=refs,study_end=pd.Timestamp('2022-01-09 05:00'))
    assert result.metadata['retrospective_reference_cohort_count']==1
    assert result.summary.weak_reference_reference_events.eq(1).all()
    # The calibration event remains a positive forecast target, even though
    # it is not a held-out retrospective matching reference.
    assert result.risk_metrics.train_positive_rate.gt(0).all()
    assert result.risk_metrics.train_positive_rate.lt(1).all()
