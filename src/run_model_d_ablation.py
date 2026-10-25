"""Reproduce the focused real-data M0--M3 ablation (English artifacts).

Run from the repository root with ``PYTHONPATH=src/Model_D python
src/run_model_d_ablation.py``. The study keeps the notebook's UTC partitions,
24-hour Model D windows updated hourly, and an independently budgeted VQE on
identical fourth-order operators. No synthetic telemetry or per-level labels.

Purpose:
    Primary experiment of the manuscript. On the 2022-07-01 to 2022-09-11
    campaign it materialises, for every 24-hour window ending on an hourly
    grid, the nine evidence coordinates of Model D (seven behavioural
    coordinates, the command-response deviation and the exact fourth-order
    transient concentration) plus the variational (VQE) estimate of the
    latter, and then evaluates four nested evidence levels with
    :class:`iabm_incidents.ablation.FourLevelAblationRunner`:
    M0 (behavioural), M1 (+ command-response deviation), M2 (+ exact
    transient concentration) and M3 (M2 with the VQE estimate substituted
    for the exact one).

Command line (from the repository root)::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src/Model_D \
        python src/run_model_d_ablation.py

    # Re-evaluate the stored indicators without re-encoding or rerunning VQE:
    PYTHONPATH=src/Model_D python src/run_model_d_ablation.py --reuse-indicators

    # Recompute only the behavioural columns, keeping frozen electrical operators:
    PYTHONPATH=src/Model_D python src/run_model_d_ablation.py --reuse-indicators --refresh-behavioral

Chronological partitions (:data:`BOUNDARIES`, naive UTC):
    observer fit ``[2022-07-01, 2022-07-22)``; reference
    ``[2022-07-22, 2022-08-05)``; calibration ``[2022-08-05, 2022-08-19)``;
    test ``[2022-08-19, 2022-09-11)``. A window is assigned to a phase only
    if it lies entirely inside it, otherwise it is ``excluded_boundary``.
    The observer and behavioural baseline are fitted on the first
    partition, the transient reference (centering/whitening) on the
    reference partition, thresholds on calibration, and results are
    reported on test.

Inputs:
    * ``data/analogicas_nonans.parquet``: columns ``0`` (timestamp), ``RP1``,
      ``RP2``, ``RP3`` (phase electrical channels).
    * ``data/digitales.parquet``: columns ``0`` (timestamp) and ``estado``
      (controller command state).
    * ``data/estados_nonans.parquet``: columns ``source_month``, ``date``,
      ``pred_estado`` (retained Model_A predictions; months 202207-202209).
      Joined for provenance only; never used as commands.
    * ``src/predictions/Model_D/candidate_incident_registry_refined.csv``: the
      weak reference registry (fixed across levels).
    * With ``--reuse-indicators``: ``raw_indicators.parquet`` and
      ``input_provenance.json`` from ``--output-dir`` (plus, when VQE peaks
      are missing, ``observer_reference_validation.csv``,
      ``operator_context_scores.csv``,
      ``identical_operators_and_vqe_directions.npz`` and
      ``frozen_electrical_reference.npz``).

Outputs (``--output-dir``, default ``reports/four_level_ablation/2022_focused``):
    * Materialisation: ``raw_indicators.parquet``,
      ``frozen_electrical_reference.npz`` (per-state ``center``,
      ``whitening``, ``scatter``, ``scale``),
      ``observer_reference_validation.csv``, ``operator_context_scores.csv``,
      ``identical_operators_and_vqe_directions.npz`` and
      ``input_provenance.json`` (boundaries, SHA-256 of inputs and source
      files, electrical configuration, VQE budget, encoding time).
    * Audit: ``reference_source_audit.csv`` (existence and SHA-256 of the
      candidate maintenance/incident reference sources).
    * Evaluation (:meth:`FourLevelAblationResult.export`): ``summary.csv``,
      ``pairwise.csv``, ``calibration.csv``, ``support.csv``,
      ``ablation_metadata.json``, ``assessment.md``, ``ablation_report.html``
      and per-level folders ``M0``..``M3`` (``window_scores``, ``episodes``,
      ``family_summary``, ``risk_predictions``, ``risk_metrics``,
      ``risk_coefficients`` CSVs).
    * Figures (if Matplotlib is installed): ``detection_and_warning_burden.png``,
      ``episode_timeline.png``, ``weak_reference_risk_performance.png``,
      appended to ``ablation_report.html``.

Script dependencies:
    Requires ``generate_candidate_incident_registry_batch.py`` (registry) and
    the data files above (``estados_nonans.parquet`` from
    ``concat_monthly_csv_to_parquet.py``). Must run before
    ``run_model_d_multiscale.py`` and ``run_model_d_extensions.py``, which
    restore its frozen observer, electrical reference and VQE budget and
    verify its input hashes.

.. mermaid::

    flowchart TD
        A[analogicas_nonans.parquet] --> AL[load_aligned: causal as-of join, 2 s]
        D[digitales.parquet] --> AL
        E[estados_nonans.parquet] --> AL
        AL --> OB[CommandResponseObserver + BehavioralWindowEncoder fit on observer partition]
        OB --> RES[signed residuals]
        RES --> TC[TransientConcentration fit on reference partition]
        RES --> WIN[24 h windows, 1 h step: 9 coordinates]
        TC --> WIN
        WIN --> VQE[VQEDirectionSolver on identical operators]
        VQE --> IND[raw_indicators.parquet + npz + provenance]
        IND --> LOC[localize_variational_projection]
        LOC --> RUN[FourLevelAblationRunner M0..M3]
        REG[candidate_incident_registry_refined.csv] --> RUN
        RUN --> OUT[summary, pairwise, calibration, support, levels, HTML]

Maintenance notes:
    The ``source_month`` filter of the Model_A predictions
    (``202207``-``202209``) is hard-coded and must follow :data:`BOUNDARIES`
    if the study period changes. ``input_provenance.json`` stores SHA-256
    hashes of this file and of the Model_D modules, so any edit (including
    documentation) changes the hashes recorded by the next run. Downstream
    scripts treat the output folder as immutable and verify it by hash; do
    not rerun the materialisation into the same folder after the multiscale
    and extension studies have been produced unless they are regenerated as
    well. The full materialisation (alignment, encoding and VQE) recorded
    ``encoding_seconds`` of about 160 s in the retained provenance;
    ``--reuse-indicators`` only re-evaluates. Setting
    ``OPENBLAS_NUM_THREADS=1``/``OMP_NUM_THREADS=1`` is recommended for
    reproducible linear-algebra timings.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from iabm_incidents.ablation import FourLevelAblationRunner
from iabm_incidents.behavioral import BehavioralWindowEncoder
from iabm_incidents.electrical import (ElectricalConfig, CommandResponseObserver,
    TransientConcentration, VQEDirectionSolver, causal_delay, _safe_history)

#: Chronological study partitions (naive UTC dates); see module docstring.
BOUNDARIES = dict(start='2022-07-01', observer_end='2022-07-22', reference_end='2022-08-05',
                  calibration_end='2022-08-19', end='2022-09-11')


def _hash(path):
    """Return the SHA-256 hex digest of a file, read in 8 MiB blocks.

    Args:
        path (str | Path): File to hash.

    Returns:
        str: Lower-case hexadecimal SHA-256 digest.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    h = hashlib.sha256()
    with open(path,'rb') as source:
        for block in iter(lambda:source.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_aligned(data_dir):
    """Causal digital as-of alignment within two seconds; never use predicted states as commands.

    Each analogue sample receives the most recent controller command
    (``estado``) observed at most two seconds earlier (backward as-of join).
    Retained Model_A predictions are attached by exact timestamp only for
    provenance and coverage statistics.

    Args:
        data_dir (Path): Folder with ``analogicas_nonans.parquet``,
            ``digitales.parquet`` and ``estados_nonans.parquet``.

    Returns:
        tuple[pandas.DataFrame, dict]: ``aligned`` with naive-UTC
        ``timestamp``, ``RP1``-``RP3``, ``command_time``,
        ``controller_state`` (NaN when no command within 2 s),
        ``asset_id`` (``"asset-unknown"``) and ``pred_estado``; and
        ``audit`` with, per raw file, ``rows``, ``duplicate_timestamps`` and
        ``sha256``, plus ``unmatched_command_samples`` and
        ``retained_Model_A_predictions`` (hash, exact-timestamp match
        fraction, provenance note).

    Raises:
        pandas.errors.MergeError: If the prediction join is not one-to-one.

    Notes:
        Rows are filtered to ``[BOUNDARIES['start'], BOUNDARIES['end'])``
        at read time (pyarrow predicate push-down). Duplicate timestamps are
        counted, then removed keeping the last occurrence. Every input file
        is fully hashed, which dominates I/O on large inputs.
    """
    frames = []; audit = {}
    start,end = pd.Timestamp(BOUNDARIES['start'],tz='UTC'),pd.Timestamp(BOUNDARIES['end'],tz='UTC')
    for name, columns in [('analogicas_nonans.parquet',['0','RP1','RP2','RP3']),('digitales.parquet',['0','estado'])]:
        path = data_dir/name; dataset = ds.dataset(path,format='parquet');typ = dataset.schema.field('0').type
        tab = dataset.to_table(columns=columns,filter=(ds.field('0') >= pa.scalar(start.to_pydatetime(),type=typ)) & (ds.field('0') < pa.scalar(end.to_pydatetime(),type=typ)))
        frame = tab.to_pandas().reset_index().rename(columns={'0':'timestamp','estado':'controller_state'})
        frame['timestamp'] = pd.to_datetime(frame.timestamp,utc=True).dt.tz_convert(None)
        duplicates = int(frame.timestamp.duplicated().sum())
        frame = frame.sort_values('timestamp').drop_duplicates('timestamp',keep='last')
        frames.append(frame)
        audit[name] = dict(rows=len(frame), duplicate_timestamps=duplicates,sha256=_hash(path))
    analog,digital = frames
    aligned = pd.merge_asof(analog,digital[['timestamp','controller_state']].rename(columns={'timestamp':'command_time'}),left_on='timestamp',right_on='command_time',direction='backward',tolerance=pd.Timedelta(seconds=2))
    aligned['asset_id']='asset-unknown'
    audit['unmatched_command_samples'] = int(aligned.controller_state.isna().sum())
    path = data_dir/'estados_nonans.parquet'
    # Note: month filter is hard-coded to the study period in BOUNDARIES.
    pred = ds.dataset(path,format='parquet').to_table(columns=['date','pred_estado'],filter=ds.field('source_month').isin(['202207','202208','202209'])).to_pandas()
    pred['timestamp'] = pd.to_datetime(pred.date,utc=True,format='mixed').dt.tz_convert(None)
    pred = pred[(pred.timestamp >= start.tz_convert(None)) & (pred.timestamp < end.tz_convert(None))].sort_values('timestamp').drop_duplicates('timestamp')
    aligned = aligned.merge(pred[['timestamp','pred_estado']],on='timestamp',how='left',validate='one_to_one')
    audit['retained_Model_A_predictions'] = dict(sha256=_hash(path),exact_timestamp_fraction=float(aligned.pred_estado.notna().mean()),
        provenance='retained pred_estado; original Model A chronological training split not recovered; not independent health labels')
    return aligned,audit


def _phase(start,end):
    """Assign a window to the chronological partition that fully contains it.

    Args:
        start (pandas.Timestamp): Window start (naive UTC).
        end (pandas.Timestamp): Window end (naive UTC).

    Returns:
        str: ``'reference'``, ``'calibration'`` or ``'test'`` when
        ``[start, end]`` lies inside that partition, else
        ``'excluded_boundary'`` (windows straddling a boundary or inside the
        observer-fit partition).
    """
    for name,low,high in [('reference','observer_end','reference_end'),('calibration','reference_end','calibration_end'),('test','calibration_end','end')]:
        if start >= pd.Timestamp(BOUNDARIES[low]) and end <= pd.Timestamp(BOUNDARIES[high]):
            return name
    return 'excluded_boundary'


def reference_audit(root):
    """Inventory genuine maintenance-source references without recycling old performance.

    Args:
        root (Path): Repository root used to resolve relative paths.

    Returns:
        pandas.DataFrame: One row per candidate source with ``path``,
        ``exists``, ``interpretation`` and ``sha256`` (``None`` when absent).

    Notes:
        Only existence and hashes are recorded; no metric is read from these
        sources. Absolute ``/tmp`` entries are the original Word documents,
        which are normally absent.
    """
    records = []
    sources = [
        ('src/generate_probe_incident_timeline_report.py','maintenance extraction code for 2024-02--2026-04; references absent temporary Word sources'),
        ('src/predictions/episode_eval_full/comparison_report/comparison_summary.csv','retained aggregate comparison: 14 periods and 11 work-order points; lacks interval records'),
        ('src/predictions/episode_eval_full/comparison_report/incident_timeline_report.html','historical maintenance comparison, outside 2022 focused study'),
        ('src/predictions/Model_D/candidate_incident_registry_refined.csv','weak automatically generated signal events, not independent incident truth'),
        ('ontology/examples/wheel_washer_incidents.ttl','ontology example; not an operational incident registry'),
        ('/tmp/Tabla_Incidencias.docx','original referenced maintenance intervals'),
        ('/tmp/OT - INCIDENCIAS LAVA RUEDAS.docx','original referenced work orders'),
    ]
    for name,description in sources:
        path = Path(name) if name.startswith('/') else root/name
        records.append(dict(path=name,exists=path.exists(),interpretation=description,
            sha256=_hash(path) if path.is_file() else None))
    return pd.DataFrame(records)


def localize_variational_projection(indicators, output, data_dir, provenance):
    """Replay stored VQE directions for retrospective peak-location comparison.

    The original frozen observer and whitening are restored, not refitted.
    Different peak locations are not physical-onset errors or online latency.

    For every window with supported operators, the context with the largest
    VQE score is selected, its delay-embedded residuals are projected onto
    the stored VQE direction and the time of the largest absolute projection
    is stored as ``vqe_projection_peak_time``.

    Args:
        indicators (pandas.DataFrame): Raw indicators indexed by window
            position, containing ``start_time``, ``end_time`` and
            ``exact_projection_peak_time``.
        output (Path): Folder holding the frozen artifacts
            (``observer_reference_validation.csv``,
            ``operator_context_scores.csv``,
            ``identical_operators_and_vqe_directions.npz``,
            ``frozen_electrical_reference.npz``).
        data_dir (Path): Raw data folder passed to :func:`load_aligned`.
        provenance (dict): Parsed ``input_provenance.json`` (provides the
            ``electrical_config``).

    Returns:
        pandas.DataFrame: ``indicators`` with ``vqe_projection_peak_time``
        and ``exact_vqe_peak_displacement_seconds`` (VQE minus exact peak
        time, seconds); also written to ``output/raw_indicators.parquet``.

    Notes:
        Reloads and re-aligns the raw telemetry.
    """
    print('Auditing exact/VQE retrospective projection peaks from frozen directions...',flush=True)
    telemetry,_ = load_aligned(data_dir)
    cfg = ElectricalConfig(**{**provenance['electrical_config'], 'channels':tuple(provenance['electrical_config']['channels'])})
    observer = CommandResponseObserver(cfg)
    for row in pd.read_csv(output/'observer_reference_validation.csv').itertuples(index=False):
        values = row._asdict()
        observer.profiles[(row.asset_id,row.controller_state)] = (
            np.array([values[channel+'_expected'] for channel in cfg.channels]),
            np.array([values[channel+'_residual_scale'] for channel in cfg.channels]))
    residuals = observer.residuals(telemetry)
    times = residuals.timestamp.to_numpy(dtype='datetime64[ns]')
    contexts = pd.read_csv(output/'operator_context_scores.csv')
    stored = np.load(output/'identical_operators_and_vqe_directions.npz')
    refs = np.load(output/'frozen_electrical_reference.npz')
    vectors = stored['vqe_directions']
    for position,group in contexts.groupby('window_position'):
        selected = group.vqe_score.idxmax();state = contexts.loc[selected,'controller_state']
        row = indicators.loc[position]
        lo,hi = np.searchsorted(times,[row.start_time.to_datetime64(),row.end_time.to_datetime64()])
        part = residuals.iloc[lo:hi]
        r = part[['residual_1','residual_2','residual_3']].to_numpy(float)
        safe = part.safe.to_numpy(bool) & _safe_history(part,cfg) & np.isfinite(r).all(1)
        ok = safe & part.controller_state.eq(state).to_numpy()
        x = causal_delay(r,ok)
        prefix = f'state_{state}_'
        z = (x-refs[prefix+'center']) @ refs[prefix+'whitening']
        projection = z @ vectors[selected]
        # Invariant: a delay-embedded row at sample t needs samples t-3..t to be
        # valid, matching the row order produced by causal_delay.
        valid = ok[3:] & ok[2:-1] & ok[1:-2] & ok[:-3]
        indices = np.arange(3,len(r))[valid]
        peak = lo+int(indices[np.argmax(abs(projection))])
        indicators.loc[position,'vqe_projection_peak_time'] = pd.Timestamp(times[peak])
    indicators['exact_vqe_peak_displacement_seconds'] = (
        pd.to_datetime(indicators.vqe_projection_peak_time)-pd.to_datetime(indicators.exact_projection_peak_time)).dt.total_seconds()
    indicators.to_parquet(output/'raw_indicators.parquet',index=False)
    return indicators


def plot_results(result, output):
    """Export standard scientific plots; optional Matplotlib has no role in scoring.

    Args:
        result (FourLevelAblationResult): Evaluated ablation.
        output (Path): Destination folder.

    Returns:
        list[Path]: Written figures (``detection_and_warning_burden.png``,
        ``episode_timeline.png`` and, when 6-hour risk metrics with a Brier
        score exist, ``weak_reference_risk_performance.png``); empty when
        Matplotlib is not installed.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        return []
    paths=[]
    summary=result.summary.set_index('level').loc[['M0','M1','M2','M3']]
    colors=['#64748b','#2563eb','#059669','#9333ea']
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    axes[0].bar(summary.index,summary.episode_count,color=colors)
    axes[0].set(title='Detected candidate episodes',ylabel='Episode count')
    axes[1].bar(summary.index,summary.warning_hours_per_100_observed_hours,color=colors)
    axes[1].set(title='Total warning burden (not false warnings)',ylabel='Warning hours per 100 decision hours',ylim=(0,100))
    path=output/'detection_and_warning_burden.png';fig.savefig(path,dpi=170);plt.close(fig);paths.append(path)
    fig,axis=plt.subplots(figsize=(12,4),constrained_layout=True)
    for number,(level,data) in enumerate(result.levels.items()):
        for row in data.episodes.itertuples():
            axis.barh(number,(pd.Timestamp(row.event_end)-pd.Timestamp(row.event_start)).total_seconds()/86400,
                left=mdates.date2num(pd.Timestamp(row.event_start)),height=.45,color=colors[number],alpha=.8)
    axis.set(yticks=range(4),yticklabels=list(result.levels),title='Lookback-based episode spans; not independent physical failures',xlabel='UTC date')
    axis.xaxis_date();axis.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    path=output/'episode_timeline.png';fig.savefig(path,dpi=170);plt.close(fig);paths.append(path)
    risk=[]
    for level,data in result.levels.items():
        part=data.risk_metrics[data.risk_metrics.horizon_hours.eq(6)]
        if not part.empty and 'brier_score' in part:
            risk.append(dict(level=level,**part.iloc[0].to_dict()))
    if risk:
        table=pd.DataFrame(risk)
        fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        axes[0].bar(table.level,table.roc_auc,color=colors)
        axes[0].axhline(.5,color='black',linestyle='--',label='Random ranking')
        axes[0].set(title='6-hour weak-reference ROC AUC',ylim=(0,1));axes[0].legend()
        axes[1].bar(table.level,table.brier_score,color=colors)
        axes[1].axhline(table.prevalence_baseline_brier.iloc[0],color='black',linestyle='--',label='Training-prevalence comparator')
        axes[1].set(title='6-hour weak-reference Brier score',ylabel='Lower is better',ylim=(0,1));axes[1].legend()
        path=output/'weak_reference_risk_performance.png';fig.savefig(path,dpi=170);plt.close(fig);paths.append(path)
    return paths


def main():
    """Run materialisation (or reuse), VQE localisation and the M0--M3 evaluation.

    Raises:
        FileNotFoundError: If inputs or, with ``--reuse-indicators``, stored
            artifacts are missing.
        ValueError: If no supported operator is found (``np.stack`` of an
            empty list).

    Notes:
        The module docstring is also the ``--help`` text. Options:
        ``--output-dir`` (Path), ``--reuse-indicators``,
        ``--refresh-behavioral`` (only with ``--reuse-indicators``) and
        ``--data-dir`` (Path, default ``data``). Paths of the registry and of
        hashed source files are resolved against the current working
        directory, so run from the repository root.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('reports/four_level_ablation/2022_focused'))
    parser.add_argument('--reuse-indicators',action='store_true',help='Reuse completed real indicators/operators from this output directory')
    parser.add_argument('--refresh-behavioral',action='store_true',help='Recompute behavioral columns while retaining completed frozen electrical operators')
    parser.add_argument('--data-dir',type=Path,default=Path('data'))
    args = parser.parse_args();output = args.output_dir;output.mkdir(parents=True,exist_ok=True)
    root = Path.cwd(); tic = time.perf_counter()
    if args.reuse_indicators:
        indicators = pd.read_parquet(output/'raw_indicators.parquet')
        provenance = json.loads((output/'input_provenance.json').read_text())
        if args.refresh_behavioral:
            print('Refreshing seven behavioral coordinates from original real telemetry; electrical reference/operators stay frozen...',flush=True)
            telemetry,audit = load_aligned(args.data_dir)
            cfg = ElectricalConfig(**{**provenance['electrical_config'], 'channels':tuple(provenance['electrical_config']['channels'])})
            fit = telemetry[(telemetry.timestamp < pd.Timestamp(BOUNDARIES['observer_end'])) & telemetry.controller_state.notna()].copy()
            # Note: the observer is refitted on the same observer partition with
            # the stored configuration; only behavioural columns are rewritten.
            observer = CommandResponseObserver(cfg).fit(fit)
            behavioral = BehavioralWindowEncoder(cfg).fit(fit)
            residuals = observer.residuals(telemetry)
            times = residuals.timestamp.to_numpy(dtype='datetime64[ns]')
            for i,row in indicators.iterrows():
                if row.phase == 'excluded_boundary':
                    continue
                lo,hi = np.searchsorted(times,[row.start_time.to_datetime64(),row.end_time.to_datetime64()])
                values = behavioral.encode(residuals.iloc[lo:hi])
                for name,value in values.items():
                    indicators.loc[i,name] = value
            indicators.to_parquet(output/'raw_indicators.parquet',index=False)
            provenance['behavioral_refresh'] = dict(reason='mask rare unknown commands and exclude their transitions; sufficient nominal-state coverage for dwell',
                source_hashes={name:_hash(root/name) for name in ['src/run_model_d_ablation.py','src/Model_D/iabm_incidents/behavioral.py']})
            (output/'input_provenance.json').write_text(json.dumps(provenance,indent=2,default=str),encoding='utf-8')
    else:
        print('Loading and causally aligning real analogue/digital telemetry...',flush=True)
        telemetry,audit = load_aligned(args.data_dir)
        # Note: the nominal sample period is the median inter-sample spacing.
        dt = float(telemetry.timestamp.diff().dt.total_seconds().median())
        cfg = ElectricalConfig(sample_period_seconds=dt,maximum_gap_seconds=2*dt,minimum_samples=32,minimum_safe_fraction=.90)
        fit = telemetry[(telemetry.timestamp < pd.Timestamp(BOUNDARIES['observer_end'])) & telemetry.controller_state.notna()].copy()
        # Removing unknown commands must leave visible timestamp gaps.
        observer = CommandResponseObserver(cfg).fit(fit)
        behavioral = BehavioralWindowEncoder(cfg).fit(fit)
        print('Frozen observer and seven-coordinate behavioral baseline fitted; extracting signed residuals...',flush=True)
        residuals = observer.residuals(telemetry)
        reference = residuals[(residuals.timestamp >= pd.Timestamp(BOUNDARIES['observer_end'])) & (residuals.timestamp < pd.Timestamp(BOUNDARIES['reference_end']))].copy()
        model = TransientConcentration(cfg).fit(reference)
        np.savez_compressed(output/'frozen_electrical_reference.npz',**{
            f'state_{state}_{field}':getattr(ref,field) for (_,state),ref in model.references.items() for field in ('center','whitening','scatter','scale')})
        observer_rows = []
        for (asset,state),(level,scale) in observer.profiles.items():
            part = reference[reference.controller_state.eq(state) & reference.safe]
            r = part[['residual_1','residual_2','residual_3']].to_numpy()
            observer_rows.append(dict(asset_id=asset,controller_state=state,reference_samples=len(part),
                **{f'{name}_expected':float(level[j]) for j,name in enumerate(cfg.channels)},
                **{f'{name}_residual_scale':float(scale[j]) for j,name in enumerate(cfg.channels)},
                maximum_phase_median_absolute_standardized_error=float(np.median(abs(r),axis=0).max()) if len(r) else np.nan))
        pd.DataFrame(observer_rows).to_csv(output/'observer_reference_validation.csv',index=False)
        times = residuals.timestamp.to_numpy(dtype='datetime64[ns]')
        rows,operators,operator_info = [],[],[]
        current = pd.Timestamp(BOUNDARIES['observer_end']); stop = pd.Timestamp(BOUNDARIES['end'])
        length,step = pd.Timedelta(hours=24),pd.Timedelta(hours=1)
        # Invariant: windows end on an hourly grid starting at observer_end;
        # window_position == row index, used to link operators to windows.
        while current+length <= stop:
            end = current+length;phase = _phase(current,end)
            lo,hi = np.searchsorted(times,[current.to_datetime64(),end.to_datetime64()])
            row = dict(window_id=f'w_{len(rows):05d}',asset_id='asset-unknown',start_time=current,end_time=end,phase=phase,
                data_coverage=min((hi-lo)*dt/length.total_seconds(),1.),
                command_response_deviation=np.nan,transient_concentration=np.nan,transient_concentration_vqe=np.nan,
                electrical_status='unsupported',electrical_safe_fraction=0.,transient_supported_fraction=0.)
            if hi > lo and phase != 'excluded_boundary':
                part = residuals.iloc[lo:hi]
                row.update(behavioral.encode(part))
                r = part[['residual_1','residual_2','residual_3']].to_numpy(float)
                u = part.controller_state.to_numpy()
                safe = part.safe.to_numpy(bool) & _safe_history(part,cfg) & np.isfinite(r).all(1)
                coverage = min(float(safe.mean()),float(safe.sum()*dt/length.total_seconds()),1.)
                row['electrical_safe_fraction']=coverage
                if coverage >= cfg.minimum_safe_fraction:
                    row['command_response_deviation']=float(np.linalg.norm(r[safe],axis=1).mean())
                    supported = np.zeros(len(r),bool);pending=[]
                    # Note: one operator per controller state present in the window;
                    # the window score is the maximum over these contexts.
                    for state in pd.unique(u):
                        ref = model.references.get(('asset-unknown',state))
                        ok = safe & (u == state);x = causal_delay(r,ok)
                        if ref is None or len(x) < cfg.minimum_samples:
                            continue
                        supported |= ok
                        d = model.operator(x,ref)
                        eigenvalues,eigenvectors = np.linalg.eigh(d)
                        v = eigenvectors[:,-1]
                        # Why: the operator is a fourth-order scatter deviation, so the
                        # score is the fourth root of its top eigenvalue (clipped at 0).
                        exact = max(float(eigenvalues[-1]),0.)**.25
                        projection = ((x-ref.center) @ ref.whitening) @ v
                        valid = ok[3:] & ok[2:-1] & ok[1:-2] & ok[:-3]
                        indices = np.arange(3,len(r))[valid]
                        peak = int(indices[np.argmax(abs(projection))])
                        pending.append((d,dict(window_position=len(rows),controller_state=float(state),exact_score=exact,
                            exact_peak_time=pd.Timestamp(times[lo+peak]),coordinate_count=len(x))))
                    row['transient_supported_fraction']=min(float(supported.mean()),float(supported.sum()*dt/length.total_seconds()),1.)
                    if pending and row['transient_supported_fraction'] >= cfg.minimum_safe_fraction:
                        row['electrical_status']='supported'
                        row['transient_concentration']=max(info['exact_score'] for _,info in pending)
                        best = max(pending,key=lambda item:item[1]['exact_score'])[1]
                        row['exact_projection_peak_time']=best['exact_peak_time']
                        for d,info in pending:
                            operators.append(d);operator_info.append(info)
            rows.append(row);current += step
            if len(rows)%100 == 0:
                print(f'Encoded {len(rows)} real 24-hour windows; {len(operators)} identical-matrix contexts...',flush=True)
        indicators = pd.DataFrame(rows)
        matrices = np.stack(operators)
        solver = VQEDirectionSolver(layers=3,sweeps=6,restarts=2,seed=1729)
        directions = []
        print(f'Running independent VQE for all {len(matrices)} supported context operators...',flush=True)
        for first in range(0,len(matrices),256):
            directions.append(solver.solve_batch(matrices[first:first+256]))
            if first%2560 == 0:
                print(f'VQE {min(first+256,len(matrices))}/{len(matrices)} contexts',flush=True)
        # Note: the VQE score is the Rayleigh quotient of the identical operator
        # along the unit-norm variational direction, so it cannot exceed the
        # exact score (up to floating-point error).
        vectors = np.concatenate(directions)
        rayleigh = np.einsum('bi,bij,bj->b',vectors,matrices,vectors)
        vqe = np.maximum(rayleigh,0)**.25
        for info,approx in zip(operator_info,vqe):
            info['vqe_score']=float(approx)
        context = pd.DataFrame(operator_info)
        for position,group in context.groupby('window_position'):
            indicators.loc[position,'transient_concentration_vqe']=group.vqe_score.max()
        context.to_csv(output/'operator_context_scores.csv',index=False)
        np.savez_compressed(output/'identical_operators_and_vqe_directions.npz',operators=matrices,vqe_directions=vectors,window_positions=context.window_position.to_numpy(),controller_states=context.controller_state.to_numpy())
        indicators.to_parquet(output/'raw_indicators.parquet',index=False)
        provenance = dict(boundaries_utc=BOUNDARIES,input_audit=audit,electrical_config=asdict(cfg),
            vqe_resource_budget=dict(layers=3,sweeps=6,restarts=2,seed=1729,batch_size=256,simulator='NumPy statevector RY/CX; no eigenvector warm start or fallback'),
            behavioral_definition='new frozen raw-command seven-coordinate encoder; not replay of retained semantic baseline; command-bit-count mode proxy',
            observer_definition='state median/MAD and settling guard; not the atlas state-and-age observer',
            model_a_prediction_warning='retained sparse predictions; original Model A fit/holdout provenance not reconstructed',
            encoding_seconds=time.perf_counter()-tic,source_code_sha256={name:_hash(root/name) for name in ['src/run_model_d_ablation.py','src/Model_D/iabm_incidents/behavioral.py','src/Model_D/iabm_incidents/electrical.py','src/Model_D/iabm_incidents/ablation.py']})
        (output/'input_provenance.json').write_text(json.dumps(provenance,indent=2,default=str),encoding='utf-8')
    if 'vqe_projection_peak_time' not in indicators:
        indicators = localize_variational_projection(indicators,output,args.data_dir,provenance)
    reference_audit(root).to_csv(output/'reference_source_audit.csv',index=False)
    registry_path=root/'src/predictions/Model_D/candidate_incident_registry_refined.csv'
    incidents = pd.read_csv(registry_path)
    # The weak reference is fixed across levels; no self-generated outcomes.
    result = FourLevelAblationRunner().run(indicators=indicators,incidents=incidents,
        independent_incidents=False,study_end=pd.Timestamp(BOUNDARIES['end']))
    provenance['evaluation_source_code_sha256'] = {name:_hash(root/name) for name in ['src/run_model_d_ablation.py','src/Model_D/iabm_incidents/ablation.py']}
    import platform, pyarrow
    result.metadata['runtime_versions'] = dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,pyarrow=pyarrow.__version__)
    result.metadata['input_provenance']=provenance
    result.metadata['maintenance_search']='Found 2024--2026 retained reported-maintenance comparison, referenced Word sources absent; no recoverable 2022 interval-level maintenance truth.'
    result.export(output)
    plots=plot_results(result,output)
    if plots:
        from html import escape
        report=output/'ablation_report.html'
        links="<h2>Comparison figures</h2>"+"".join(f"<p><img src='{escape(path.name,quote=True)}' alt='{escape(path.stem)}' style='max-width:100%'></p>" for path in plots)
        report.write_text(report.read_text().replace('</html>',links+'</html>'),encoding='utf-8')
    print(result.summary.to_string(index=False),flush=True)
    print(f'Artifacts: {output.resolve()}\nElapsed: {time.perf_counter()-tic:.1f} seconds',flush=True)


if __name__ == '__main__':
    main()
