"""Execute one prespecified real-data multiscale M0--M3 sensitivity.

Use ``PYTHONPATH=src/Model_D python src/run_model_d_multiscale.py`` from the repo
root. The primary study is read-only. All supplemental scales use a fixed
15-minute decision grid and shared endpoint support, with the original frozen
observer/electrical reference and exactly the original VQE resource budget.

Purpose:
    Window-length sensitivity of the M0--M3 ablation. The same nine evidence
    coordinates (plus the VQE estimate) are materialised for window lengths
    of 15 min, 1 h, 6 h and 24 h
    (:class:`iabm_incidents.multiscale.MultiscaleSensitivityConfig`
    defaults), all ending on a common 15-minute decision grid, and evaluated
    with :class:`iabm_incidents.multiscale.MultiscaleSensitivityRunner` on
    the endpoints supported at every scale. It reports exposure-normalised
    warning burden and episode rates per scale; it deliberately does not
    select a "best" scale.

Command line (from the repository root)::

    PYTHONPATH=src/Model_D python src/run_model_d_multiscale.py

    # Re-evaluate cached per-scale indicators (checked against frozen inputs):
    PYTHONPATH=src/Model_D python src/run_model_d_multiscale.py --reuse-indicators

``src`` itself is on ``sys.path`` because it is the script folder, which
makes ``run_model_d_ablation`` importable.

Chronological boundaries:
    Identical to :data:`run_model_d_ablation.BOUNDARIES`. The phase of every
    endpoint is decided with the *widest* window (24 h) at all scales, so the
    fit/calibration/test endpoints are identical across scales and no
    shorter window is labelled with a phase its 24 h counterpart straddles.
    The first endpoint is ``observer_end + 24 h``.

Inputs:
    * ``--primary-dir`` (default ``reports/four_level_ablation/2022_focused``,
      read-only): ``input_provenance.json``,
      ``observer_reference_validation.csv``,
      ``frozen_electrical_reference.npz``.
    * ``--data-dir`` (default ``data``): the same raw Parquet files as the
      ablation (see :func:`run_model_d_ablation.load_aligned`); their SHA-256
      must equal the hashes stored in the primary provenance.
    * ``src/predictions/Model_D/candidate_incident_registry_refined.csv``.
    * With ``--reuse-indicators``: ``sensitivity_metadata.json``,
      ``input_source_audit.json`` and ``window_<m>min/raw_indicators.parquet``
      from ``--output-dir``.

Outputs (``--output-dir``, default ``reports/four_level_ablation/multiscale_sensitivity``):
    * Per scale ``window_<m>min/``: ``raw_indicators.parquet``,
      ``operator_context_scores.csv``,
      ``identical_operators_and_vqe_directions.npz`` and the per-scale
      ablation export (``summary.csv``, ``calibration.csv``,
      ``ablation_metadata.json``, level folders, ...).
    * ``summary.csv``, ``pairwise.csv``, ``risk_metrics.csv``,
      ``availability.csv`` (own and cross-scale support per endpoint),
      ``sensitivity_metadata.json``, ``assessment.md``,
      ``sensitivity_report.html``, ``multiscale_sensitivity.png``,
      ``input_source_audit.json`` and ``primary_artifact_integrity.json``
      (SHA-256 of every primary file, verified unchanged).

Script dependencies:
    Run after ``run_model_d_ablation.py`` (primary artifacts) and
    ``generate_candidate_incident_registry_batch.py`` (registry). Must run
    before ``run_model_d_extensions.py``, which reads ``availability.csv`` and
    the ``window_<m>min`` folders.

.. mermaid::

    flowchart TD
        P[primary ablation artifacts] --> RE[restore_electrical: frozen observer + reference]
        RAW[raw telemetry] --> H{hashes equal primary?}
        H -->|yes| RES[observer residuals]
        RE --> RES
        RES --> MS[materialize_scale for 15, 60, 360, 1440 min]
        RE --> MS
        MS --> RUN[MultiscaleSensitivityRunner on common support]
        REG[candidate registry] --> RUN
        RUN --> OUT[summary, pairwise, risk, availability, HTML, PNG]
        P --> INT[before/after hash integrity check]
        INT --> OUT

Maintenance notes:
    The primary directory is hashed before and after the run; any change
    raises :class:`RuntimeError`. ``--reuse-indicators`` fails closed when the
    cached metadata disagrees with the frozen primary manifest, observer,
    electrical reference, VQE budget, window lengths or decision step. The
    plot labels (``15 min``, ``1 h``, ``6 h``, ``24 h``) are hard-coded and
    assume the default window lengths. The retained metadata record a
    materialisation time of about 710 s for the four scales, while the
    re-evaluation with ``--reuse-indicators`` took about 24 s.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import json
import platform
import time

import numpy as np
import pandas as pd

from run_model_d_ablation import BOUNDARIES, load_aligned, _hash, _phase
from iabm_incidents.ablation import BEHAVIORAL_COLUMNS
from iabm_incidents.behavioral import BehavioralWindowEncoder
from iabm_incidents.electrical import (ElectricalConfig, CommandResponseObserver,
    TransientConcentration, TransientReference, VQEDirectionSolver, causal_delay, _safe_history)
from iabm_incidents.multiscale import MultiscaleSensitivityConfig, MultiscaleSensitivityRunner


def restore_electrical(primary):
    """Restore the primary observer and operator reference without refitting either.

    Args:
        primary (Path): Primary ablation folder.

    Returns:
        tuple: ``(cfg, observer, model, provenance)`` where ``cfg`` is the
        stored :class:`ElectricalConfig`, ``observer`` a
        :class:`CommandResponseObserver` whose per-(asset, state) expected
        levels and residual scales are loaded from
        ``observer_reference_validation.csv``, ``model`` a
        :class:`TransientConcentration` whose references are loaded from
        ``frozen_electrical_reference.npz`` (keyed by
        ``('asset-unknown', float_state)``), and ``provenance`` the parsed
        ``input_provenance.json``.

    Raises:
        FileNotFoundError: If one of the three primary files is missing.
        KeyError: If the npz lacks a field for a stored state.
    """
    provenance = json.loads((primary/'input_provenance.json').read_text())
    cfg = ElectricalConfig(**{**provenance['electrical_config'],
        'channels':tuple(provenance['electrical_config']['channels'])})
    observer = CommandResponseObserver(cfg)
    for record in pd.read_csv(primary/'observer_reference_validation.csv').to_dict('records'):
        observer.profiles[(record['asset_id'],record['controller_state'])] = (
            np.array([record[channel+'_expected'] for channel in cfg.channels]),
            np.array([record[channel+'_residual_scale'] for channel in cfg.channels]))
    model = TransientConcentration(cfg)
    with np.load(primary/'frozen_electrical_reference.npz') as stored:
        for key in stored.files:
            if key.endswith('_center'):
                # Invariant: keys were written as f'state_{float_state}_<field>',
                # so re-formatting the parsed float reproduces the prefix.
                state = float(key[len('state_'):-len('_center')])
                prefix = f'state_{state}_'
                model.references[('asset-unknown',state)] = TransientReference(
                    stored[prefix+'center'],stored[prefix+'whitening'],stored[prefix+'scatter'],
                    float(stored[prefix+'scale']))
    return cfg, observer, model, provenance


def materialize_scale(residuals, nominal, model, minutes, sensitivity, budget, output):
    """Compute scale-specific behavioral/electrical evidence on shared endpoints.

    The widest window determines phase labels/warmup at all scales. This keeps
    identical fit/calibration/test endpoints and avoids partition-straddling
    observations. Individual state and telemetry support policies are unchanged.

    Args:
        residuals (pandas.DataFrame): Frozen-observer residuals of the whole
            study period (``timestamp``, ``residual_1``-``residual_3``,
            ``controller_state``, ``safe``), sorted by time.
        nominal (pandas.DataFrame): Aligned telemetry of the observer-fit
            partition with known commands, used to fit the behavioural
            encoder for this window length.
        model (TransientConcentration): Restored frozen transient model.
        minutes (int): Window length in minutes.
        sensitivity (MultiscaleSensitivityConfig): Provides
            ``decision_step_minutes`` and ``windows_minutes``.
        budget (dict): VQE resource budget (``layers``, ``sweeps``,
            ``restarts``, ``seed``, ``batch_size``) copied from the primary
            provenance.
        output (Path): Per-scale folder (``window_<m>min``), created if
            needed.

    Returns:
        pandas.DataFrame: One row per decision endpoint with identifiers,
        phase, coverage, behavioural columns, command-response deviation,
        exact and VQE transient concentration, support fractions, status and
        exact/VQE projection peak times.

    Raises:
        ValueError: If no context operator is supported at this scale (the
            support policy is never relaxed).

    Notes:
        Writes ``raw_indicators.parquet``, ``operator_context_scores.csv`` and
        ``identical_operators_and_vqe_directions.npz`` into ``output``.
    """
    cfg = model.config
    length = pd.Timedelta(minutes=minutes)
    step = pd.Timedelta(minutes=sensitivity.decision_step_minutes)
    widest = pd.Timedelta(minutes=max(sensitivity.windows_minutes))
    encoder = BehavioralWindowEncoder(cfg).fit(nominal, window_length=length, step=step)
    times = residuals.timestamp.to_numpy(dtype='datetime64[ns]')
    end = pd.Timestamp(BOUNDARIES['observer_end'])+widest
    stop = pd.Timestamp(BOUNDARIES['end'])
    rows, matrices, contexts = [], [], []
    while end <= stop:
        start = end-length
        phase = _phase(end-widest,end)
        lo,hi = np.searchsorted(times,[start.to_datetime64(),end.to_datetime64()])
        row = dict(window_id=f'{minutes}min_{len(rows):05d}',asset_id='asset-unknown',
            start_time=start,end_time=end,phase=phase,window_minutes=minutes,
            data_coverage=min((hi-lo)*cfg.sample_period_seconds/length.total_seconds(),1.),
            **{name:np.nan for name in BEHAVIORAL_COLUMNS},
            command_response_deviation=np.nan,transient_concentration=np.nan,
            transient_concentration_vqe=np.nan,electrical_status='unsupported',
            electrical_safe_fraction=0.,transient_supported_fraction=0.)
        if hi > lo and phase != 'excluded_boundary':
            part = residuals.iloc[lo:hi]
            row.update(encoder.encode(part))
            r = part[['residual_1','residual_2','residual_3']].to_numpy(float)
            states = part.controller_state.to_numpy()
            safe = part.safe.to_numpy(bool) & _safe_history(part,cfg) & np.isfinite(r).all(1)
            coverage = min(float(safe.mean()),safe.sum()*cfg.sample_period_seconds/length.total_seconds(),1.)
            row['electrical_safe_fraction'] = coverage
            if coverage >= cfg.minimum_safe_fraction:
                row['command_response_deviation'] = float(np.linalg.norm(r[safe],axis=1).mean())
                supported = np.zeros(len(r),bool);pending=[]
                for state in pd.unique(states):
                    reference = model.references.get(('asset-unknown',state))
                    ok = safe & (states == state)
                    x = causal_delay(r,ok)
                    if reference is None or len(x) < cfg.minimum_samples:
                        continue
                    supported |= ok
                    d = model.operator(x,reference)
                    values,vectors = np.linalg.eigh(d)
                    projection = ((x-reference.center) @ reference.whitening) @ vectors[:,-1]
                    # Invariant: embedding row t uses samples t-3..t (causal_delay order).
                    valid = ok[3:] & ok[2:-1] & ok[1:-2] & ok[:-3]
                    ids = np.arange(3,len(r))[valid]
                    peak = lo+int(ids[np.argmax(abs(projection))])
                    pending.append((d,dict(window_position=len(rows),controller_state=float(state),
                        exact_score=max(float(values[-1]),0.)**.25,
                        exact_peak_time=pd.Timestamp(times[peak]),coordinate_count=len(x))))
                row['transient_supported_fraction'] = min(float(supported.mean()),
                    supported.sum()*cfg.sample_period_seconds/length.total_seconds(),1.)
                if pending and row['transient_supported_fraction'] >= cfg.minimum_safe_fraction:
                    row['electrical_status'] = 'supported'
                    best = max(pending,key=lambda item:item[1]['exact_score'])[1]
                    row['transient_concentration'] = best['exact_score']
                    row['exact_projection_peak_time'] = best['exact_peak_time']
                    for matrix,context in pending:
                        matrices.append(matrix); contexts.append(context)
        rows.append(row);end += step
        if len(rows)%400 == 0:
            print(f'{minutes} min: encoded {len(rows)} endpoints; {len(matrices)} supported context operators',flush=True)
    frame = pd.DataFrame(rows)
    if not matrices:
        raise ValueError(f'No supported operators at {minutes} min; support policy was not relaxed')
    matrices = np.stack(matrices)
    solver = VQEDirectionSolver(**{name:budget[name] for name in ('layers','sweeps','restarts','seed')})
    batch_size = budget['batch_size']
    vectors = []
    for lo in range(0,len(matrices),batch_size):
        vectors.append(solver.solve_batch(matrices[lo:lo+batch_size]))
        if lo%(batch_size*10) == 0:
            print(f'{minutes} min: VQE {min(lo+batch_size,len(matrices))}/{len(matrices)} contexts',flush=True)
    vectors = np.concatenate(vectors)
    vqe = np.maximum(np.einsum('bi,bij,bj->b',vectors,matrices,vectors),0)**.25
    context_frame = pd.DataFrame(contexts)
    context_frame['vqe_score'] = vqe
    # Note: VQE peak localisation replays the best VQE context of each window.
    for position,group in context_frame.groupby('window_position'):
        best = group.vqe_score.idxmax()
        frame.loc[position,'transient_concentration_vqe'] = group.vqe_score.max()
        state = context_frame.loc[best,'controller_state']
        row = frame.loc[position]
        lo,hi = np.searchsorted(times,[row.start_time.to_datetime64(),row.end_time.to_datetime64()])
        part = residuals.iloc[lo:hi]
        r = part[['residual_1','residual_2','residual_3']].to_numpy(float)
        safe = part.safe.to_numpy(bool) & _safe_history(part,cfg) & np.isfinite(r).all(1)
        ok = safe & part.controller_state.eq(state).to_numpy()
        x = causal_delay(r,ok)
        reference = model.references[('asset-unknown',state)]
        projection = ((x-reference.center) @ reference.whitening) @ vectors[best]
        valid = ok[3:] & ok[2:-1] & ok[1:-2] & ok[:-3]
        ids = np.arange(3,len(r))[valid]
        peak = lo+int(ids[np.argmax(abs(projection))])
        frame.loc[position,'vqe_projection_peak_time'] = pd.Timestamp(times[peak])
    frame['exact_vqe_peak_displacement_seconds'] = (
        pd.to_datetime(frame.vqe_projection_peak_time)-pd.to_datetime(frame.exact_projection_peak_time)).dt.total_seconds()
    output.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(output/'raw_indicators.parquet',index=False)
    context_frame.to_csv(output/'operator_context_scores.csv',index=False)
    np.savez_compressed(output/'identical_operators_and_vqe_directions.npz',operators=matrices,
        vqe_directions=vectors,window_positions=context_frame.window_position.to_numpy(),
        controller_states=context_frame.controller_state.to_numpy())
    return frame


def plot_sensitivity(result, output):
    """Plot comparable exposure-normalized quantities, never rank a best scale.

    Args:
        result: Multiscale sensitivity result with ``summary`` and
            ``metadata['windows_minutes']``.
        output (Path): Destination folder.

    Returns:
        list[Path]: ``[output / 'multiscale_sensitivity.png']``, or an empty
        list when Matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    colors=['#64748b','#2563eb','#059669','#9333ea']
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    windows=result.metadata['windows_minutes'];x=np.arange(len(windows));labels=['15 min','1 h','6 h','24 h']
    for i,level in enumerate(['M0','M1','M2','M3']):
        values=result.summary[result.summary.level.eq(level)].set_index('window_minutes').loc[list(windows)]
        axes[0].plot(x,values.warning_hours_per_100_observed_hours,'o-',color=colors[i],label=level)
        axes[1].plot(x,values.episodes_per_100_observed_hours,'o-',color=colors[i],label=level)
    axes[0].set(title='Total warning burden (not false warnings)',ylabel='Warning h per 100 matched decision h',ylim=(0,100))
    axes[1].set(title='Candidate episode rate on common support',ylabel='Episodes per 100 matched decision h')
    for axis in axes:
        axis.set(xticks=x,xticklabels=labels,xlabel='Prespecified window length');axis.legend();axis.grid(alpha=.2)
    path=output/'multiscale_sensitivity.png';fig.savefig(path,dpi=180);plt.close(fig)
    return [path]


def main():
    """Run (or reuse) the four-scale materialisation and the sensitivity evaluation.

    Raises:
        ValueError: If primary and output directories overlap, cached
            indicators disagree with the frozen inputs, or raw-input hashes
            differ from the primary study.
        RuntimeError: If any primary artifact changed during the run.

    Notes:
        The module docstring is also the ``--help`` text. Options:
        ``--primary-dir``, ``--output-dir``, ``--data-dir`` (Paths) and
        ``--reuse-indicators``. The registry path is relative to the current
        working directory.
    """
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary-dir',type=Path,default=Path('reports/four_level_ablation/2022_focused'))
    parser.add_argument('--output-dir',type=Path,default=Path('reports/four_level_ablation/multiscale_sensitivity'))
    parser.add_argument('--data-dir',type=Path,default=Path('data'))
    parser.add_argument('--reuse-indicators',action='store_true',help='Reuse completed sensitivity indicators; never read/write primary indicators as sensitivity data')
    args=parser.parse_args();primary=args.primary_dir.resolve();output=args.output_dir.resolve()
    # Invariant: sensitivity outputs may never be written inside the primary study.
    if output == primary or primary in output.parents or output in primary.parents:
        raise ValueError('Primary and sensitivity artifact directories must be separate')
    output.mkdir(parents=True,exist_ok=True);tic=time.perf_counter()
    before={str(path.relative_to(primary)):_hash(path) for path in primary.rglob('*') if path.is_file()}
    config=MultiscaleSensitivityConfig();cfg,observer,model,provenance=restore_electrical(primary)
    frames={};audit=None
    previous_metadata = None
    if args.reuse_indicators:
        previous_metadata = json.loads((output/'sensitivity_metadata.json').read_text())
        expected = {
            'primary_manifest_sha256':_hash(primary/'input_provenance.json'),
            'frozen_observer_sha256':_hash(primary/'observer_reference_validation.csv'),
            'frozen_electrical_reference_sha256':_hash(primary/'frozen_electrical_reference.npz'),
            'vqe_resource_budget':provenance['vqe_resource_budget'],
            'windows_minutes':list(config.windows_minutes),
            'decision_step_minutes':config.decision_step_minutes,
        }
        if any(previous_metadata.get(name) != value for name,value in expected.items()):
            raise ValueError('Cached sensitivity differs from frozen inputs, grid or circuit budget')
    if not args.reuse_indicators:
        print('Loading real telemetry once and restoring the frozen primary electrical models...',flush=True)
        telemetry,audit=load_aligned(args.data_dir)
        # Fail closed if the immutable study's upstream observations changed.
        for name in ('analogicas_nonans.parquet','digitales.parquet'):
            if audit[name]['sha256'] != provenance['input_audit'][name]['sha256']:
                raise ValueError('Sensitivity inputs differ from primary study source hashes')
        if audit['retained_Model_A_predictions']['sha256'] != provenance['input_audit']['retained_Model_A_predictions']['sha256']:
            raise ValueError('Retained Model A predictions differ from primary study')
        residuals=observer.residuals(telemetry)
        nominal=telemetry[(telemetry.timestamp < pd.Timestamp(BOUNDARIES['observer_end'])) & telemetry.controller_state.notna()].copy()
    for minutes in config.windows_minutes:
        folder=output/f'window_{minutes}min'
        if args.reuse_indicators:
            frames[minutes]=pd.read_parquet(folder/'raw_indicators.parquet')
        else:
            frames[minutes]=materialize_scale(residuals,nominal,model,minutes,config,
                provenance['vqe_resource_budget'],folder)
    if audit is not None:
        (output/'input_source_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    else:
        audit=json.loads((output/'input_source_audit.json').read_text())
    incidents=pd.read_csv('src/predictions/Model_D/candidate_incident_registry_refined.csv')
    result=MultiscaleSensitivityRunner(config).run(indicators_by_scale=frames,incidents=incidents,
        independent_incidents=False,study_end=pd.Timestamp(BOUNDARIES['end']))
    after={str(path.relative_to(primary)):_hash(path) for path in primary.rglob('*') if path.is_file()}
    if before != after:
        raise RuntimeError('Primary artifact integrity check failed')
    integrity=dict(primary_directory=str(primary),unchanged=True,files=before)
    (output/'primary_artifact_integrity.json').write_text(json.dumps(integrity,indent=2),encoding='utf-8')
    result.metadata.update(boundaries_utc=BOUNDARIES,primary_artifact_files_unchanged=True,
        primary_manifest_sha256=_hash(primary/'input_provenance.json'),
        frozen_observer_sha256=_hash(primary/'observer_reference_validation.csv'),
        frozen_electrical_reference_sha256=_hash(primary/'frozen_electrical_reference.npz'),
        vqe_resource_budget=provenance['vqe_resource_budget'],input_source_audit=audit,
        runtime_versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        elapsed_seconds=time.perf_counter()-tic,
        materialization_elapsed_seconds=(previous_metadata.get('materialization_elapsed_seconds',previous_metadata['elapsed_seconds']) if previous_metadata else time.perf_counter()-tic),
        source_code_sha256={str(path):_hash(path) for path in [Path(__file__),
            Path('src/Model_D/iabm_incidents/multiscale.py'),Path('src/Model_D/iabm_incidents/ablation.py'),
            Path('src/Model_D/iabm_incidents/behavioral.py')]})
    result.export(output)
    plots=plot_sensitivity(result,output)
    if plots:
        report=output/'sensitivity_report.html'
        report.write_text(report.read_text().replace('</html>',
            "<h2>Prespecified scale comparison</h2><img src='multiscale_sensitivity.png' alt='Warning burden and episode rate by window length' style='max-width:100%'></html>"),encoding='utf-8')
    print(result.summary[['window_minutes','level','test_windows','warning_windows','episode_count',
        'warning_hours_per_100_observed_hours']].to_string(index=False),flush=True)
    print(f'Sensitivity report: {output}\nPrimary artifacts unchanged: {len(before)} files',flush=True)


if __name__ == '__main__':
    main()
