"""Methodological extensions to the focused Model D study (English artifacts).

Run from the repository root with ``PYTHONPATH=src/Model_D python
src/run_model_d_extensions.py --stage all``. Stages:

* ``model_a``: class-level and yearly Model A diagnostics on retained predictions.
* ``quantum``: magnitude/shape (kurtosis ratio, QJSD, trace distance) evidence,
  Kato–Temple certified VQE decisions, finite-shot/noise resource curves and
  shot-based noisy VQE on the frozen operators of every scale.
* ``telemetry``: second-order versus fourth-order statistics on the primary
  24-hour windows and the semi-synthetic fault-injection benchmark.
* ``fusion``: conformal Cauchy-combination fusion of evidence dimensions and
  scales, alternative transient coordinates and multivariate baselines.
* ``robustness``: certification of transient-only decisions, Kato-Temple
  tightness, shot/noise study at 24 h and 15 min, and conformal seed and
  mid-p sensitivity.

Primary and multiscale artifacts are read-only; outputs go to ``--output-dir``.

Purpose:
    Methodological extensions reported in the manuscript on top of the
    frozen primary ablation and the multiscale sensitivity. No primary
    indicator, threshold or operator is refitted; the stages only read the
    stored operators, indicators, calibrations and support masks (and, for
    ``telemetry``, the raw data re-aligned with the frozen observer).

Command line (from the repository root)::

    PYTHONPATH=src/Model_D python src/run_model_d_extensions.py --stage all
    PYTHONPATH=src/Model_D python src/run_model_d_extensions.py --stage fusion --output-dir reports/extensions

Inputs:
    * :data:`PRIMARY` (``reports/four_level_ablation/2022_focused``):
      ``raw_indicators.parquet``, ``ablation_metadata.json`` (``centers`` and
      ``scales`` of each coordinate), ``calibration.csv`` (``M2`` onset and
      recovery thresholds), ``support.csv`` (``common_support``),
      ``input_provenance.json`` (VQE budget, input hashes),
      ``identical_operators_and_vqe_directions.npz``,
      ``operator_context_scores.csv`` and the frozen observer/reference files
      used by :func:`run_model_d_multiscale.restore_electrical`.
    * :data:`MULTISCALE` (``reports/four_level_ablation/multiscale_sensitivity``):
      ``availability.csv`` (``cross_scale_support``) and, per scale,
      ``window_<m>min/`` with the same files as the primary folder.
    * :data:`REGISTRY` (weak reference registry), fusion stage.
    * ``data/estados_nonans.parquet`` (``source_month``, ``estado``,
      ``pred_estado``), model_a stage; raw telemetry under ``--data-dir``,
      telemetry stage.

Outputs (``--output-dir``, default ``reports/extensions``):
    * model_a: ``model_a_class_metrics.csv``, ``model_a_period_metrics.csv``.
    * quantum: ``certificates_<scale>.csv`` (per-context certificate
      columns), ``certificate_summary.csv``, ``information_<scale>.parquet``,
      and from :func:`shot_study`: ``shot_noise_study.csv``,
      ``noisy_vqe_study.csv``, ``decision_margins.csv``,
      ``swap_test_shots.csv``.
    * telemetry: ``order_statistics_primary.parquet``,
      ``injection_windows.parquet``, ``injection_summary.csv``.
    * fusion: ``conformal_primary_pvalues.csv``, ``fusion_summary.csv``,
      ``fusion_attribution.csv``.
    * robustness: ``certificate_decision_dependence.csv``,
      ``kato_temple_tightening.csv``, ``shot_noise_T4_decisions.csv``,
      ``shot_noise_T4_decisions_15min.csv``, ``conformal_seed_sensitivity.csv``,
      ``conformal_midp.csv``, ``conformal_episodes_<level>.csv``,
      ``consistency_checks.csv``.
    * Always: ``extensions_metadata.json`` (stage completion times, noise
      models, source hashes, elapsed time; merged with previous runs).

Script dependencies:
    ``run_model_d_ablation.py`` -> ``run_model_d_multiscale.py`` -> this
    script -> ``export_extension_figure_data.py``. Within this script the
    ``fusion`` stage uses ``information_primary.parquet`` (quantum stage) and
    ``order_statistics_primary.parquet`` (telemetry stage) when present and
    silently skips the corresponding alternative-coordinate rows otherwise,
    so with ``--stage all`` the order ``model_a``, ``quantum``,
    ``telemetry``, ``fusion``, ``robustness`` matters (``robustness`` reads
    the outputs of ``quantum`` and ``telemetry``).

.. mermaid::

    flowchart TD
        P[primary ablation artifacts] --> Q[stage_quantum + shot_study]
        M[multiscale artifacts] --> Q
        P --> T[stage_telemetry]
        RAW[raw telemetry] --> T
        E[estados_nonans.parquet] --> A[stage_model_a]
        Q -->|information_primary.parquet| F[stage_fusion + multiscale_fusion]
        T -->|order_statistics_primary.parquet| F
        P --> F
        Q -->|certificates_*.csv| R[stage_robustness]
        T -->|order_statistics_primary.parquet| R
        M --> F
        REG[candidate registry] --> F
        Q --> OUT[reports/extensions]
        T --> OUT
        F --> OUT
        A --> OUT
        OUT --> FIG[export_extension_figure_data.py]

Maintenance notes:
    All files of the primary and multiscale folders except ``.html`` are
    hashed before and after the run; a change raises :class:`RuntimeError`
    (the HTML reports are excluded because the upstream scripts append
    figures to them). Random seeds are fixed (shot estimators, noisy VQE,
    injection RNG ``20260902``, Isolation Forest ``random_state=0``). The
    retained ``extensions_metadata.json`` records about 140 s for its last
    run; the ``telemetry`` and ``quantum`` stages are the expensive ones.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from run_model_d_ablation import BOUNDARIES, load_aligned, _hash
from run_model_d_multiscale import restore_electrical
from iabm_incidents.ablation import (BEHAVIORAL_COLUMNS, COMMAND_COLUMN, EXACT_COLUMN, VQE_COLUMN,
    FourLevelAblationRunner, _normalize_incidents, _reference_metrics)
from iabm_incidents.config import ModelDConfig
from iabm_incidents.conformal import ConformalEvidenceFusion, ConformalFusionConfig
from iabm_incidents.electrical import VQEDirectionSolver, _safe_history
from iabm_incidents.injection import STATISTICS, Injection, fit_second_order, inject, window_statistics
from iabm_incidents.quantum import (NoiseModel, NoisyVQEDirectionSolver, ShotEstimator,
    deflated_second_direction, information_profile, kato_temple_bounds, swap_test_shots)

#: Primary (hourly, 24 h) ablation folder; read-only.
PRIMARY = Path('reports/four_level_ablation/2022_focused')
#: Multiscale sensitivity folder; read-only.
MULTISCALE = Path('reports/four_level_ablation/multiscale_sensitivity')
#: Weak reference registry produced by the candidate-registry batch.
REGISTRY = Path('src/predictions/Model_D/candidate_incident_registry_refined.csv')
#: Multiscale window lengths in minutes.
SCALES = (15, 60, 360, 1440)
#: The nine M2 evidence coordinates: seven behavioural, command-response, exact transient.
NINE = [*BEHAVIORAL_COLUMNS, COMMAND_COLUMN, EXACT_COLUMN]
#: Noise models of the resource study (ideal, mitigated and unmitigated NISQ-like).
NOISE = {'ideal': NoiseModel(), 'nisq_mitigated': NoiseModel(1e-3, 1e-2, 2e-2, True),
         'nisq_unmitigated': NoiseModel(1e-3, 1e-2, 2e-2, False)}


def scale_paths(minutes):
    """Primary hourly 24-h study or one multiscale arm.

    Args:
        minutes (str | int): ``'primary'`` or a window length in
            :data:`SCALES`.

    Returns:
        Path: :data:`PRIMARY` or ``MULTISCALE / 'window_<minutes>min'``.
    """
    return PRIMARY if minutes == 'primary' else MULTISCALE/f'window_{minutes}min'


def load_scale(minutes):
    """Raw indicators, frozen centers/scales, M2 thresholds and common support.

    Args:
        minutes (str | int): ``'primary'`` or a window length in
            :data:`SCALES`.

    Returns:
        tuple: ``(frame, meta, calibration, common)`` with ``frame`` the raw
        indicators (``start_time``/``end_time`` as naive UTC), ``meta`` the
        parsed ``ablation_metadata.json``, ``calibration`` the
        ``calibration.csv`` table indexed by ``level`` and ``common`` a
        boolean numpy mask of windows inside common support (primary:
        ``support.csv``; multiscale: ``cross_scale_support`` of
        ``availability.csv`` for this scale).

    Raises:
        FileNotFoundError: If one of the required files is missing.
    """
    folder = scale_paths(minutes)
    frame = pd.read_parquet(folder/'raw_indicators.parquet')
    for column in ('start_time', 'end_time'):
        frame[column] = pd.to_datetime(frame[column], utc=True, format='mixed').dt.tz_convert(None)
    meta = json.loads((folder/'ablation_metadata.json').read_text())
    calibration = pd.read_csv(folder/'calibration.csv').set_index('level')
    if minutes == 'primary':
        support = pd.read_csv(folder/'support.csv')
        common = frame.window_id.isin(support.loc[support.common_support, 'window_id'])
    else:
        availability = pd.read_csv(MULTISCALE/'availability.csv')
        part = availability[availability.window_minutes.eq(minutes)]
        common = frame.window_id.isin(part.loc[part.cross_scale_support, 'window_id'])
    return frame, meta, calibration, common.to_numpy()


def excess(frame, meta, columns):
    """Directional excess exactly as the ablation (absolute for consumption).

    Each coordinate is standardised with the frozen ablation center and
    scale; ``consumption_deviation`` contributes its absolute value and every
    other coordinate only its positive part. The VQE column reuses the exact
    transient coordinate's center/scale.

    Args:
        frame (pandas.DataFrame): Raw indicators.
        meta (dict): Ablation metadata with ``centers`` and ``scales``.
        columns (list[str]): Coordinates to transform.

    Returns:
        pandas.DataFrame: Non-negative excess per column, same index as
        ``frame`` (NaN where the indicator is missing).
    """
    out = pd.DataFrame(index=frame.index)
    for c in columns:
        key = EXACT_COLUMN if c == VQE_COLUMN else c
        delta = (frame[c]-meta['centers'][key])/meta['scales'][key]
        out[c] = delta.abs() if c == 'consumption_deviation' else delta.clip(lower=0)
    return out


def stage_model_a(output):
    """Per-class, macro and yearly agreement with controller-derived states.

    Compares the retained Model_A predictions (``pred_estado``) with the
    controller-derived states (``estado``) over the whole campaign. States
    with a share of at least 0.1 % are reported per class; states with at
    least 1 % are "dominant" and define the macro scores.

    Args:
        output (Path): Destination folder.

    Notes:
        Writes ``model_a_class_metrics.csv`` and ``model_a_period_metrics.csv``
        (``all`` plus one row per calendar year derived from
        ``source_month``) and prints both tables. Requires scikit-learn.
        These are agreement statistics on retained predictions, not a
        held-out evaluation of Model_A.
    """
    import pyarrow.dataset as ds
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
    d = ds.dataset('data/estados_nonans.parquet').to_table(columns=['source_month', 'estado', 'pred_estado']).to_pandas()
    y, p = d.estado.astype(int), d.pred_estado.astype(int)
    share = y.value_counts(normalize=True)
    dominant = sorted(share[share >= .01].index)
    labels = sorted(share[share >= .001].index)
    precision, recall, f1, support = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    table = pd.DataFrame(dict(state=labels, precision=precision, recall=recall, f1=f1, support=support,
                              support_share=share.loc[labels].to_numpy(), dominant=[s in dominant for s in labels]))
    table.to_csv(output/'model_a_class_metrics.csv', index=False)
    rows = []
    for name, part in [('all', d), *d.groupby(d.source_month.str[:4])]:
        yy, pp = part.estado.astype(int), part.pred_estado.astype(int)
        keep = yy.isin(dominant)
        rows.append(dict(period=name, samples=len(part), accuracy=accuracy_score(yy, pp),
            balanced_accuracy_all_states=balanced_accuracy_score(yy, pp),
            macro_f1_dominant=f1_score(yy[keep], pp[keep], labels=dominant, average='macro', zero_division=0),
            balanced_accuracy_dominant=float(np.mean(precision_recall_fscore_support(yy[keep], pp[keep], labels=dominant, zero_division=0)[1])),
            dominant_state_share=float(keep.mean()), states_observed=int(yy.nunique())))
    pd.DataFrame(rows).to_csv(output/'model_a_period_metrics.csv', index=False)
    print(table.round(4).to_string(index=False)); print(pd.DataFrame(rows).round(4).to_string(index=False))


def _context_frame(folder):
    """Load the stored context operators, VQE directions and context table.

    Args:
        folder (Path): Primary or per-scale folder.

    Returns:
        tuple: ``(operators, directions, context)`` with operators of shape
        ``(n, 8, 8)``, unit VQE directions ``(n, 8)`` and the
        ``operator_context_scores.csv`` dataframe (same row order).
    """
    stored = np.load(folder/'identical_operators_and_vqe_directions.npz')
    context = pd.read_csv(folder/'operator_context_scores.csv')
    return stored['operators'], stored['vqe_directions'], context


def stage_quantum(output, references):
    """Certificates, information profiles and resource curves on frozen operators.

    For every scale (primary and :data:`SCALES`): computes exact top and
    second eigenvalues, a deflated (VQD) second direction, information
    profiles (QJSD, kurtosis ratio, trace distance) against the frozen
    references, and Kato-Temple bounds of the VQE Rayleigh quotient under two
    protocols (``rigorous_frobenius_gap`` and ``vqd_gap_assisted``). A window
    decision is *certified* when the lower bound already exceeds the M2
    onset threshold (certified warning) or the upper bound stays below it
    (certified clear); other windows need the exact fallback.

    Args:
        output (Path): Destination folder.
        references (dict): Frozen transient references keyed by
            ``('asset-unknown', float_state)``.

    Notes:
        Writes ``certificates_<scale>.csv``, ``certificate_summary.csv`` and
        ``information_<scale>.parquet``, then runs :func:`shot_study`. The
        per-context CSV stores ``<protocol>_lower_score`` and
        ``<protocol>_upper_score`` for each protocol.
    """
    budget = json.loads((PRIMARY/'input_provenance.json').read_text())['vqe_resource_budget']
    solver = VQEDirectionSolver(**{k: budget[k] for k in ('layers', 'sweeps', 'restarts', 'seed')})
    certificate_rows, profile_frames = [], {}
    for minutes in ('primary', *SCALES):
        folder = scale_paths(minutes)
        frame, meta, calibration, common = load_scale(minutes)
        operators, directions, context = _context_frame(folder)
        exact = np.linalg.eigvalsh(operators)
        second = np.concatenate([deflated_second_direction(operators[i:i+512], directions[i:i+512], solver)
                                 for i in range(0, len(operators), 512)])
        context['exact_max'], context['exact_second'], context['vqd_second'] = exact[:, -1], exact[:, -2], second
        profiles = pd.DataFrame([information_profile(d, references[('asset-unknown', float(s))])
                                 for d, s in zip(operators, context.controller_state)])
        context = pd.concat([context, profiles], axis=1)
        protocols = {'rigorous_frobenius_gap': [kato_temple_bounds(d, v) for d, v in zip(operators, directions)],
                     'vqd_gap_assisted': [kato_temple_bounds(d, v, g) for d, v, g in zip(operators, directions, second)]}
        others = excess(frame, meta, NINE[:-1]).sum(axis=1)
        tau = calibration.loc['M2', 'onset_threshold']
        c, s = meta['centers'][EXACT_COLUMN], meta['scales'][EXACT_COLUMN]
        # Note: M2 score with the transient coordinate replaced by a bound.
        score = lambda t: others+((t-c)/s).clip(lower=0)
        exact_warn = score(frame[EXACT_COLUMN]) >= tau
        for protocol, records in protocols.items():
            bounds = pd.DataFrame(records)
            for column in ('rayleigh', 'residual_variance', 'upper', 'kato_temple_applicable'):
                context[f'{protocol}_{column}'] = bounds[column]
            # Note: empirical check that the upper bound covers the exact maximum.
            covered = context.exact_max <= bounds.upper*(1+1e-9)+1e-12
            context[f'{protocol}_lower_score'] = np.maximum(bounds.rayleigh, 0)**.25
            context[f'{protocol}_upper_score'] = np.maximum(bounds.upper, 0)**.25
            window = context.groupby('window_position').agg(lower=(f'{protocol}_lower_score', 'max'), upper=(f'{protocol}_upper_score', 'max'))
            low, high = score(window.lower.reindex(frame.index)), score(window.upper.reindex(frame.index))
            for phase in ('calibration', 'test'):
                m = common & frame.phase.eq(phase).to_numpy() & low.notna().to_numpy()
                certified_warn, certified_clear = low[m] >= tau, high[m] < tau
                certified = certified_warn | certified_clear
                certificate_rows.append(dict(scale=minutes, protocol=protocol, phase=phase, windows=int(m.sum()),
                    certified_fraction=float(certified.mean()), certified_warning=int(certified_warn.sum()),
                    certified_clear=int(certified_clear.sum()), exact_fallback_windows=int((~certified).sum()),
                    certified_decision_errors=int((certified_warn & ~exact_warn[m]).sum()+(certified_clear & exact_warn[m]).sum()),
                    context_bound_coverage=float(covered.mean()), kato_temple_applicable=float(bounds.kato_temple_applicable.mean()),
                    contexts=len(context)))
        window = context.groupby('window_position').agg(qjsd=('qjsd', 'max'), kurtosis_ratio=('kurtosis_ratio', 'max'),
                                                        trace_distance=('trace_distance', 'max'))
        profile_frames[minutes] = frame.join(window, how='left')
        context.to_csv(output/f'certificates_{minutes}.csv', index=False)
    pd.DataFrame(certificate_rows).to_csv(output/'certificate_summary.csv', index=False)
    print(pd.DataFrame(certificate_rows).round(4).to_string(index=False))
    for minutes, frame in profile_frames.items():
        frame[['window_id', 'phase', 'start_time', 'end_time', EXACT_COLUMN, 'qjsd', 'kurtosis_ratio', 'trace_distance']].to_parquet(output/f'information_{minutes}.parquet', index=False)
    shot_study(output, budget)


def shot_study(output, budget):
    """Finite-shot, depolarising and readout effects on the primary test decisions.

    On the primary test windows inside common support, the Rayleigh quotient
    of each stored VQE direction is re-estimated with a shot-based estimator
    (64 to 16384 shots per measurement setting, 10 repeats, plus an exact
    readout) under each noise model in :data:`NOISE`, and M2 decisions are
    compared with the exact-eigenvalue decisions. A shot-based noisy VQE is
    also re-optimised (three repeats, 1024 shots, mitigated noise).

    Args:
        output (Path): Destination folder.
        budget (dict): Primary VQE budget (``layers``, ``sweeps``,
            ``restarts``, ``seed``).

    Notes:
        Writes ``shot_noise_study.csv`` (``shots_per_setting`` is ``inf`` for
        exact readout), ``noisy_vqe_study.csv``, ``decision_margins.csv``
        (descriptive statistics of the distance of the exact M2 score to the
        threshold) and ``swap_test_shots.csv``.
    """
    frame, meta, calibration, common = load_scale('primary')
    operators, directions, context = _context_frame(PRIMARY)
    test = common & frame.phase.eq('test').to_numpy()
    keep = context.window_position.isin(np.flatnonzero(test)).to_numpy()
    d, v, positions = operators[keep], directions[keep], context.window_position.to_numpy()[keep]
    # Note: h = -D/||D|| is the normalised Hamiltonian whose expectation is
    # estimated; lam = -||D|| * <h> recovers the Rayleigh quotient of D.
    norm = np.linalg.norm(d, axis=(1, 2)); h = -d/norm[:, None, None]
    others = excess(frame, meta, NINE[:-1]).sum(axis=1).to_numpy()
    tau = calibration.loc['M2', 'onset_threshold']
    c, s = meta['centers'][EXACT_COLUMN], meta['scales'][EXACT_COLUMN]
    def decisions(lam):
        """Map context eigenvalue estimates to window-level M2 decisions.

        Args:
            lam (numpy.ndarray): One eigenvalue estimate per kept context.

        Returns:
            tuple: Boolean warning array per window and the per-window
            transient score series (max over contexts) indexed by position.
        """
        t = pd.Series(np.maximum(lam, 0)**.25).groupby(positions).max()
        return (others[t.index]+np.clip((t.to_numpy()-c)/s, 0, None)) >= tau, t
    exact, exact_t = decisions(np.linalg.eigvalsh(d)[:, -1])
    margin = np.abs(others[exact_t.index]+np.clip((exact_t.to_numpy()-c)/s, 0, None)-tau)
    settings = len(ShotEstimator(None).settings)
    rows = []
    for name, noise in NOISE.items():
        for shots in (64, 256, 1024, 4096, 16384, None):
            agreement, errors = [], []
            for repeat in range(1 if shots is None else 10):
                estimator = ShotEstimator(shots, noise, layers=budget['layers'], seed=1000*repeat+7)
                lam = -norm*estimator.expectation(v, h)
                warn, t = decisions(lam)
                agreement.append(float((warn == exact).mean()))
                errors.append(float(np.median(np.abs(t.to_numpy()-exact_t.to_numpy())/exact_t.to_numpy())))
            rows.append(dict(noise=name, shots_per_setting=shots if shots is not None else np.inf, settings=settings,
                shots_per_context=settings*shots if shots is not None else np.inf,
                mean_decision_agreement=float(np.mean(agreement)), min_decision_agreement=float(np.min(agreement)),
                median_relative_score_error=float(np.median(errors)), windows=len(exact), contexts=len(d)))
    noisy = []
    for repeat in range(3):
        estimator = ShotEstimator(1024, NOISE['nisq_mitigated'], layers=budget['layers'], seed=50+repeat)
        solver = NoisyVQEDirectionSolver(estimator, **{k: budget[k] for k in ('layers', 'sweeps', 'restarts')}, seed=budget['seed']+repeat)
        w = np.concatenate([solver.solve_batch(d[i:i+512]) for i in range(0, len(d), 512)])
        rayleigh = np.einsum('bi,bij,bj->b', w, d, w)
        final = -norm*ShotEstimator(4096, NOISE['nisq_mitigated'], layers=budget['layers'], seed=90+repeat).expectation(w, h)
        for label, lam in (('noisy_optimisation_exact_readout', rayleigh), ('noisy_optimisation_noisy_readout', final)):
            warn, t = decisions(lam)
            noisy.append(dict(repeat=repeat, protocol=label, decision_agreement=float((warn == exact).mean()),
                median_relative_score_deficit=float(np.median((exact_t.to_numpy()-t.to_numpy())/exact_t.to_numpy())),
                p95_relative_score_deficit=float(np.quantile((exact_t.to_numpy()-t.to_numpy())/exact_t.to_numpy(), .95))))
    pd.DataFrame(rows).to_csv(output/'shot_noise_study.csv', index=False)
    pd.DataFrame(noisy).to_csv(output/'noisy_vqe_study.csv', index=False)
    pd.DataFrame(dict(margin=margin, exact_warning=exact)).describe().to_csv(output/'decision_margins.csv')
    swap = pd.DataFrame([dict(precision=e, shots_95=swap_test_shots(.5, e)) for e in (.1, .05, .02, .01)])
    swap.to_csv(output/'swap_test_shots.csv', index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False)); print(pd.DataFrame(noisy).round(4).to_string(index=False))


def stage_telemetry(output, data_dir):
    """Second-order statistics on primary windows and the injection benchmark.

    Part 1 recomputes, for every supported primary window, the statistics in
    :data:`iabm_incidents.injection.STATISTICS` (including second- and
    fourth-order concentration) from the frozen-observer residuals and checks
    that the recomputed fourth-order value matches the stored one.

    Part 2 is a semi-synthetic fault-injection benchmark on one-hour
    windows of the calibration partition (``reference_end`` to
    ``calibration_end``) with at least 600 safe, commanded samples. Even
    days of the year define the clean null (threshold = 95th percentile of
    clean windows); odd days receive under-response, burst and dropout
    injections, evaluated by detection power, null false-alarm rate, AUC
    and paired increase fraction.

    Args:
        output (Path): Destination folder.
        data_dir (Path): Raw data folder.

    Raises:
        ValueError: If the raw telemetry hashes differ from the primary study.

    Notes:
        Writes ``order_statistics_primary.parquet``,
        ``injection_windows.parquet`` and ``injection_summary.csv``. The
        scenario label format
        ``<kind>|m=<magnitude>|rate=<rate>|len=<length>|phases=<count>`` is
        parsed by ``export_extension_figure_data.py``. Uses RNG seed
        ``20260902``.
    """
    tic = time.perf_counter()
    cfg, observer, model, provenance = restore_electrical(PRIMARY)
    telemetry, audit = load_aligned(data_dir)
    for name in ('analogicas_nonans.parquet', 'digitales.parquet'):
        if audit[name]['sha256'] != provenance['input_audit'][name]['sha256']:
            raise ValueError('Telemetry differs from the frozen primary study')
    residuals = observer.residuals(telemetry)
    reference = residuals[(residuals.timestamp >= pd.Timestamp(BOUNDARIES['observer_end'])) & (residuals.timestamp < pd.Timestamp(BOUNDARIES['reference_end']))]
    seconds = fit_second_order(reference, model.references, cfg)
    times = residuals.timestamp.to_numpy(dtype='datetime64[ns]')
    r_all = residuals[['residual_1', 'residual_2', 'residual_3']].to_numpy(float)
    states_all = residuals.controller_state.to_numpy()
    safe_all = residuals.safe.to_numpy(bool)
    print(f'Residuals ready in {time.perf_counter()-tic:.0f} s', flush=True)

    def statistics(start, end, r=None):
        """Compute window statistics on ``[start, end)``.

        Args:
            start: Window start (naive UTC, anything ``numpy.datetime64``
                accepts).
            end: Window end.
            r (numpy.ndarray | None): Optional replacement residuals for the
                window (injected faults); defaults to the observed residuals.

        Returns:
            tuple: ``(values, lo, hi, safe)`` with the statistics dict, the
            residual slice bounds and the safe mask.
        """
        lo, hi = np.searchsorted(times, [np.datetime64(start), np.datetime64(end)])
        part = residuals.iloc[lo:hi]
        safe = safe_all[lo:hi] & _safe_history(part, cfg)
        return window_statistics(r_all[lo:hi] if r is None else r, safe, states_all[lo:hi], model.references, seconds, cfg.minimum_samples), lo, hi, safe

    frame, meta, calibration, common = load_scale('primary')
    rows = []
    for position, row in frame.iterrows():
        if row.electrical_status != 'supported':
            continue
        values, *_ = statistics(row.start_time, row.end_time)
        rows.append(dict(window_position=position, **values))
    order = frame.join(pd.DataFrame(rows).set_index('window_position'), how='left')
    order.to_parquet(output/'order_statistics_primary.parquet', index=False)
    check = (order.fourth_order-order[EXACT_COLUMN]).abs().max()
    print(f'Primary order statistics: max |Tconc recomputed - stored| = {check:.2e}', flush=True)

    # Note: per-sample expected level / residual scale, used by the injector to
    # express under-response relative to the nominal response.
    offsets = np.full((len(residuals), 3), np.nan)
    for (asset, state), (center, scale) in observer.profiles.items():
        mask = states_all == state
        offsets[mask] = center/scale
    # Why: telemetry is not exactly 1 Hz (median period ~0.807 s), so event
    # durations are given in seconds and converted to samples, and the real
    # period is passed to the injector for the Poisson mean.
    dt = cfg.sample_period_seconds
    samples = lambda seconds: max(1, int(round(seconds/dt)))
    scenarios = [(Injection('under_response', d, phases=p), 0.) for d in (.01, .02, .05, .10) for p in ((0,), (0, 1, 2))]
    scenarios += [(Injection('bursts', a, rate_per_hour=q, length=samples(3.), phases=(0, 1, 2)), 3.) for a in (3., 6., 12.) for q in (6., 30.)]
    scenarios += [(Injection('dropouts', 1., rate_per_hour=q, length=samples(n), phases=(0, 1, 2)), float(n)) for n in (1, 3) for q in (2., 6., 20.)]
    hours = pd.date_range(BOUNDARIES['reference_end'], BOUNDARIES['calibration_end'], freq='1h')[:-1]
    records = []
    rng = np.random.default_rng(20260902)
    for start in hours:
        end = start+pd.Timedelta(hours=1)
        clean, lo, hi, safe = statistics(start, end)
        active = safe & (states_all[lo:hi] != 0) & np.isfinite(offsets[lo:hi]).all(1)
        # Faults act only during commanded operation; require ten active minutes.
        if not np.isfinite(clean['fourth_order']) or active.sum()*dt < 600:
            continue
        # Alternate calendar days balance the non-stationarity of the two calibration weeks.
        split = 'threshold' if start.dayofyear % 2 == 0 else 'evaluation'
        records.append(dict(start_time=start, split=split, scenario='clean', active_samples=int(active.sum()), **clean))
        if split == 'threshold':
            continue
        for scenario, duration_s in scenarios:
            r, events = inject(r_all[lo:hi], states_all[lo:hi], active, offsets[lo:hi], scenario, rng,
                               sample_period_seconds=dt, return_events=True)
            values, *_ = statistics(start, end, r)
            label = f'{scenario.kind}|m={scenario.magnitude:g}|rate={scenario.rate_per_hour:g}|len={duration_s:g}s|phases={len(scenario.phases)}'
            records.append(dict(start_time=start, split='evaluation', scenario=label, active_samples=int(active.sum()),
                                events=events, **values))
    bench = pd.DataFrame(records)
    bench.to_parquet(output/'injection_windows.parquet', index=False)
    null = bench[bench.scenario.eq('clean') & bench.split.eq('threshold')]
    clean_eval = bench[bench.scenario.eq('clean') & bench.split.eq('evaluation')]
    summary = []
    for scenario, part in bench[bench.scenario.ne('clean')].groupby('scenario'):
        for statistic in STATISTICS:
            tau = null[statistic].quantile(.95)
            x, y = part[statistic].to_numpy(float), clean_eval[statistic].to_numpy(float)
            x, y = x[np.isfinite(x)], y[np.isfinite(y)]
            # Note: Mann-Whitney AUC with ties counted as one half.
            auc = float((x[:, None] > y[None]).mean()+.5*(x[:, None] == y[None]).mean())
            paired = part.set_index('start_time')[statistic]-clean_eval.set_index('start_time')[statistic].reindex(part.start_time).to_numpy()
            # Note: power is window-level; it is also reported conditional on
            # windows that actually received at least one event.
            hit = part.events.gt(0).to_numpy() if not scenario.startswith('under_response') else np.ones(len(part), bool)
            flagged = part[statistic].to_numpy(float) >= tau
            summary.append(dict(scenario=scenario, statistic=statistic, windows=len(x), detection_power=float((x >= tau).mean()),
                null_false_alarm=float((y >= tau).mean()), auc=auc, paired_increase_fraction=float((paired > 0).mean()),
                windows_with_event=int(hit.sum()), mean_events=float(part.events.mean()),
                detection_power_given_event=float(flagged[hit].mean()) if hit.any() else np.nan))
    pd.DataFrame(summary).to_csv(output/'injection_summary.csv', index=False)
    print(pd.DataFrame(summary).pivot(index='scenario', columns='statistic', values='detection_power').round(3).to_string(), flush=True)
    print(f'Telemetry stage finished in {time.perf_counter()-tic:.0f} s', flush=True)


def evaluate_scores(test, score, onset, recovery, config, references):
    """Episodes and weak-reference metrics for an arbitrary held-out score.

    Args:
        test (pandas.DataFrame): Test windows (indicator rows).
        score (array-like): Score per test window.
        onset (float): Onset threshold; ``score >= onset`` is a warning.
        recovery (float): Recovery threshold; forced strictly below
            ``onset``.
        config (ModelDConfig): Base Model_D configuration (window step and
            length define the exposure and episode logic).
        references (pandas.DataFrame): Normalised weak reference events.

    Returns:
        tuple: ``(summary, test, episodes)`` where ``summary`` holds
        ``test_windows``, ``warning_windows``,
        ``warning_hours_per_100_observed_hours``, ``episode_count``,
        ``median_episode_hours`` and the weak-reference metrics; ``test`` is
        the scored copy and ``episodes`` the detected episodes.

    Notes:
        Exposure per window is one decision step, truncated at
        ``BOUNDARIES['end']``.
    """
    test = test.copy().reset_index(drop=True)
    test['deviation_score'], test['warning'] = np.asarray(score, float), np.asarray(score, float) >= onset
    recovery = min(recovery, np.nextafter(onset, -np.inf))
    cfg = replace(config, detection=replace(config.detection, onset_threshold=onset, recovery_threshold=recovery))
    episodes = FourLevelAblationRunner(model_d_config=cfg)._episodes(test, cfg)
    step = cfg.window.step
    exposure = ((test.end_time+step).clip(upper=pd.Timestamp(BOUNDARIES['end']))-test.end_time).dt.total_seconds()/3600
    metrics = _reference_metrics(test, episodes, references, False)
    return dict(test_windows=len(test), warning_windows=int(test.warning.sum()),
        warning_hours_per_100_observed_hours=float(100*exposure[test.warning].sum()/exposure.sum()),
        episode_count=len(episodes), median_episode_hours=float(episodes.duration_seconds.median()/3600) if len(episodes) else 0.,
        **metrics), test, episodes


def stage_fusion(output):
    """Conformal fusion, alternative transient coordinates and baselines.

    On the primary 24-hour windows compares: the equal-weight M2 sum with
    the stored calibration; conformal ACAT (Cauchy combination) fusion of the
    M0, M1 and M2 coordinates calibrated on the calibration windows; M2 with the
    transient coordinate replaced by QJSD, kurtosis ratio, trace distance or
    second-order concentration; and two multivariate baselines (Hotelling T2
    and Isolation Forest) fitted on reference windows with thresholds at
    the 95th/80th calibration percentiles. Then runs
    :func:`multiscale_fusion`.

    Args:
        output (Path): Destination folder (also the source of the optional
            quantum/telemetry stage outputs).

    Notes:
        Writes ``conformal_primary_pvalues.csv``, ``fusion_summary.csv`` and
        ``fusion_attribution.csv``. Requires scikit-learn.
    """
    from sklearn.ensemble import IsolationForest
    incidents = pd.read_csv(REGISTRY)
    references = _normalize_incidents(incidents)
    rows, attributions = [], []
    frame, meta, calibration, common = load_scale('primary')
    config = ModelDConfig()
    e = excess(frame, meta, NINE)
    cal, test = common & frame.phase.eq('calibration').to_numpy(), common & frame.phase.eq('test').to_numpy()
    ref = common & frame.phase.eq('reference').to_numpy()
    tau = calibration.loc['M2']
    summary, scored, _ = evaluate_scores(frame[test], e[test].sum(axis=1), tau.onset_threshold, tau.recovery_threshold, config, references)
    rows.append(dict(scale=1440, cadence='primary', method='equal_weight_sum_M2', **summary))
    for name, columns in (('conformal_ACAT_M0', list(BEHAVIORAL_COLUMNS)), ('conformal_ACAT_M1', NINE[:-1]), ('conformal_ACAT_M2', NINE)):
        fusion = ConformalEvidenceFusion(columns).fit(e[cal])
        p = fusion.transform(e[test])
        onset, recovery = fusion.thresholds
        summary, scored, episodes = evaluate_scores(frame[test], p.fused_score, onset, recovery, config, references)
        loo = fusion.transform(e[cal]).p_fused
        rows.append(dict(scale=1440, cadence='primary', method=name, calibration_warning_fraction=float((loo <= fusion.config.onset_alpha).mean()), **summary))
        if name.endswith('M2'):
            p.assign(window_id=frame.window_id[test].to_numpy(), end_time=frame.end_time[test].to_numpy(), warning=scored.warning.to_numpy()).to_csv(output/'conformal_primary_pvalues.csv', index=False)
            dominant = p.loc[scored.warning.to_numpy(), 'dominant_evidence'].value_counts(normalize=True)
            attributions.extend(dict(scale=1440, evidence=k, warning_share=v) for k, v in dominant.items())
    for variant in ('qjsd', 'kurtosis_ratio', 'trace_distance', 'second_order'):
        source = output/('order_statistics_primary.parquet' if variant == 'second_order' else 'information_primary.parquet')
        if not source.exists():
            continue
        alt = frame.copy(); values = pd.read_parquet(source)[variant].to_numpy()
        alt[EXACT_COLUMN] = alt[VQE_COLUMN] = values
        result = FourLevelAblationRunner().run(indicators=alt, incidents=incidents, independent_incidents=False, study_end=pd.Timestamp(BOUNDARIES['end']))
        row = result.summary.set_index('level').loc['M2']
        rows.append(dict(scale=1440, cadence='primary', method=f'equal_weight_sum_M2_{variant}', **{k: row[k] for k in ('test_windows', 'warning_windows', 'warning_hours_per_100_observed_hours', 'episode_count', 'median_episode_hours', 'weak_reference_reference_events', 'weak_reference_event_overlap_fraction', 'weak_reference_fragmentation_mean_episodes_per_hit', 'weak_reference_median_absolute_onset_decision_error_hours', 'weak_reference_family_agreement')}))
    raw = frame[NINE].copy()
    center, spread = raw[ref].median(), (raw[ref]-raw[ref].median()).abs().median()*1.4826+1e-9
    zr = ((raw-center)/spread).to_numpy()
    cov = np.cov(zr[ref].T)+1e-6*np.eye(len(NINE)); inverse = np.linalg.inv(cov)
    t2 = np.einsum('bi,ij,bj->b', zr, inverse, zr)
    forest = IsolationForest(n_estimators=500, random_state=0).fit(zr[ref])
    # Unsupported windows stay outside common support; zero-fill only to vectorise scoring.
    for method, statistic in (('hotelling_T2', t2), ('isolation_forest', -forest.score_samples(np.nan_to_num(zr)))):
        onset, recovery = np.quantile(statistic[cal], .95), np.quantile(statistic[cal], .80)
        summary, *_ = evaluate_scores(frame[test], statistic[test], onset, recovery, config, references)
        rows.append(dict(scale=1440, cadence='primary', method=method, **summary))
    multiscale_fusion(rows, attributions, references, incidents)
    table = pd.DataFrame(rows)
    table.to_csv(output/'fusion_summary.csv', index=False)
    pd.DataFrame(attributions).to_csv(output/'fusion_attribution.csv', index=False)
    print(table[['scale', 'method', 'test_windows', 'warning_windows', 'warning_hours_per_100_observed_hours', 'episode_count',
                 'weak_reference_event_overlap_fraction', 'weak_reference_fragmentation_mean_episodes_per_hit']].round(4).to_string(index=False))
    print(pd.DataFrame(attributions).round(3).to_string(index=False))


def multiscale_fusion(rows, attributions, references, incidents):
    """Per-scale and scale-adaptive ACAT on the common 15-minute cohort.

    Args:
        rows (list[dict]): Summary rows; extended in place.
        attributions (list[dict]): Warning attribution rows; extended in
            place.
        references (pandas.DataFrame): Normalised weak reference events.
        incidents (pandas.DataFrame): Raw registry (unused here; kept for
            signature symmetry).

    Notes:
        Calibration/test masks are the intersection of the common support of
        every scale, with the phase taken from the 15-minute frame. The
        method requires that all scales share the same endpoint rows in the
        same order. The adaptive fusion reports ``calibration_warning_fraction``
        at a hard-coded 0.05 level.
    """
    frames, evidence, cal_mask, test_mask = {}, {}, None, None
    for minutes in SCALES:
        frame, meta, calibration, common = load_scale(minutes)
        frames[minutes] = (frame, calibration, common)
        evidence[minutes] = excess(frame, meta, NINE).add_suffix(f'@{minutes}')
        cal_mask = common & frame.phase.eq('calibration').to_numpy() if cal_mask is None else cal_mask & common
        test_mask = common & frame.phase.eq('test').to_numpy() if test_mask is None else test_mask & common
    # Invariant: every scale has identical endpoint rows (shared 15-min grid),
    # so boolean masks from one scale index all of them.
    base = frames[15][0]
    config = replace(ModelDConfig(), window=replace(ModelDConfig().window, step=pd.Timedelta(minutes=15)))
    # Same retrospective cohort as the multiscale runner: events observed in the shortest supported test windows.
    observed = base[test_mask]
    times = references.event_time.to_numpy(dtype='datetime64[ns]')
    ends, starts = observed.end_time.to_numpy(dtype='datetime64[ns]'), observed.start_time.to_numpy(dtype='datetime64[ns]')
    pos = np.searchsorted(ends, times, side='right')
    inside = (pos < len(ends)) & (starts[np.minimum(pos, len(starts)-1)] <= times)
    cohort = references[inside]
    allcols = pd.concat(evidence.values(), axis=1)
    for minutes in SCALES:
        frame, calibration, common = frames[minutes]
        e = evidence[minutes]
        tau = calibration.loc['M2']
        summary, *_ = evaluate_scores(frame[test_mask].assign(start_time=frame.start_time[test_mask]), e[test_mask].sum(axis=1),
                                      tau.onset_threshold, tau.recovery_threshold, replace(config, window=replace(config.window, length=pd.Timedelta(minutes=minutes))), cohort)
        rows.append(dict(scale=minutes, cadence='15min', method='equal_weight_sum_M2', **summary))
        fusion = ConformalEvidenceFusion(list(e.columns)).fit(e[cal_mask])
        p = fusion.transform(e[test_mask])
        summary, *_ = evaluate_scores(frame[test_mask], p.fused_score, *fusion.thresholds, replace(config, window=replace(config.window, length=pd.Timedelta(minutes=minutes))), cohort)
        rows.append(dict(scale=minutes, cadence='15min', method='conformal_ACAT_M2', **summary))
    fusion = ConformalEvidenceFusion(list(allcols.columns)).fit(allcols[cal_mask])
    p = fusion.transform(allcols[test_mask])
    widest = frames[1440][0]
    summary, scored, _ = evaluate_scores(widest[test_mask], p.fused_score, *fusion.thresholds, replace(config, window=replace(config.window, length=pd.Timedelta(minutes=1440))), cohort)
    loo = fusion.transform(allcols[cal_mask]).p_fused
    rows.append(dict(scale='adaptive', cadence='15min', method='conformal_ACAT_multiscale', calibration_warning_fraction=float((loo <= fusion.config.onset_alpha).mean()), **summary))
    dominant = p.loc[scored.warning.to_numpy(), 'dominant_evidence'].dropna()
    for k, v in dominant.str.split('@').str[1].value_counts(normalize=True).items():
        attributions.append(dict(scale='adaptive', evidence=f'scale={k}', warning_share=v))
    for k, v in dominant.str.split('@').str[0].value_counts(normalize=True).items():
        attributions.append(dict(scale='adaptive', evidence=k, warning_share=v))


def stage_robustness(output):
    """Decision-dependence, Kato-Temple tightness, shot and seed robustness.

    The composite M2 decision is often fixed by the eight non-transient
    coordinates, so certifying it says little about the transient operator.
    This stage therefore repeats certification on a *transient-only* decision
    (``T4 >= q95`` of the calibration ``T4``), counts the windows whose
    composite decision actually depends on the transient coordinate,
    measures how much the rigorous Kato-Temple bound tightens the elementary
    ``[rho, ||D||_F]`` bracket, reruns the finite-shot study on transient-only
    decisions at 24 h and 15 min, and quantifies the seed sensitivity of the
    smoothed conformal p-values (50 seeds; plus a non-randomised mid-p
    variant) for the M0, M1 and M2 fusions.

    Args:
        output (Path): Destination folder (``reports/extensions``).

    Notes:
        Writes ``certificate_decision_dependence.csv``,
        ``kato_temple_tightening.csv``, ``shot_noise_T4_decisions.csv``,
        ``shot_noise_T4_decisions_15min.csv``,
        ``conformal_seed_sensitivity.csv``, ``conformal_midp.csv``,
        ``conformal_episodes_<level>.csv`` and ``consistency_checks.csv``.
        Requires the outputs of the ``quantum`` and ``telemetry`` stages.
    """
    from scipy.stats import spearmanr
    from iabm_incidents import conformal as conformal_module
    rows, tightening = [], []
    for minutes in ('primary', *SCALES):
        frame, meta, calibration, common = load_scale(minutes)
        operators, directions, context = _context_frame(scale_paths(minutes))
        certificates = pd.read_csv(output/f'certificates_{minutes}.csv')
        rho = np.einsum('bi,bij,bj->b', directions, operators, directions)
        frobenius = np.linalg.norm(operators, axis=(1, 2))
        exact_max = np.linalg.eigvalsh(operators)[:, -1]
        positions = context.window_position.to_numpy()
        # Note: window score = max over controller-state contexts of [.]_+^{1/4}.
        window = lambda values: pd.Series(np.maximum(values, 0)**.25).groupby(positions).max().reindex(frame.index)
        lower = window(rho)
        uppers = {'cap_only': window(frobenius),
                  'rigorous_kato_temple': window(certificates.rigorous_frobenius_gap_upper.to_numpy()),
                  'vqd_assisted': window(certificates.vqd_gap_assisted_upper.to_numpy())}
        calibration_mask = common & frame.phase.eq('calibration').to_numpy()
        test = common & frame.phase.eq('test').to_numpy() & lower.notna().to_numpy()
        others = excess(frame, meta, NINE[:-1]).sum(axis=1)
        tau = calibration.loc['M2', 'onset_threshold']
        c, s = meta['centers'][EXACT_COLUMN], meta['scales'][EXACT_COLUMN]
        composite = lambda t: others+((t-c)/s).clip(lower=0)
        transient_tau = float(frame[EXACT_COLUMN][calibration_mask].quantile(.95))
        # Invariant: a window is pivotal when removing T4 changes the M2 decision.
        pivotal = int(((others[test] >= tau) != (composite(frame[EXACT_COLUMN])[test] >= tau)).sum())
        for rule, score, threshold in (('composite_M2', composite, tau), ('transient_only', lambda t: t, transient_tau)):
            exact = score(frame[EXACT_COLUMN])[test] >= threshold
            for bracket, upper in uppers.items():
                warn, clear = score(lower)[test] >= threshold, score(upper)[test] < threshold
                rows.append(dict(scale=minutes, decision=rule, bracket=bracket, windows=int(test.sum()),
                    exact_warnings=int(exact.sum()), threshold=threshold, certified_fraction=float((warn | clear).mean()),
                    exact_fallbacks=int((~(warn | clear)).sum()), certified_errors=int((warn & ~exact).sum()+(clear & exact).sum()),
                    transient_pivotal_windows=pivotal if rule == 'composite_M2' else int(test.sum())))
        applicable = certificates.rigorous_frobenius_gap_kato_temple_applicable.to_numpy(bool)
        upper = certificates.rigorous_frobenius_gap_upper.to_numpy()
        relative = lambda bound: (bound-exact_max)/np.abs(exact_max).clip(min=1e-12)
        tightening.append(dict(scale=minutes, contexts=len(rho), negative_rayleigh_fraction=float((rho < 0).mean()),
            kato_temple_applicable=float(applicable.mean()),
            median_relative_excess_cap=float(np.median(relative(frobenius)[applicable])) if applicable.any() else np.nan,
            median_relative_excess_kato_temple=float(np.median(relative(upper)[applicable])) if applicable.any() else np.nan))
    pd.DataFrame(rows).to_csv(output/'certificate_decision_dependence.csv', index=False)
    pd.DataFrame(tightening).to_csv(output/'kato_temple_tightening.csv', index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False)); print(pd.DataFrame(tightening).round(4).to_string(index=False))
    for minutes, name, shot_levels in (('primary', 'shot_noise_T4_decisions.csv', (16, 64, 256, 1024, 4096)),
                                       (15, 'shot_noise_T4_decisions_15min.csv', (16, 64, 256, 1024, 4096))):
        frame, meta, calibration, common = load_scale(minutes)
        operators, directions, context = _context_frame(scale_paths(minutes))
        test = common & frame.phase.eq('test').to_numpy()
        threshold = float(frame[EXACT_COLUMN][common & frame.phase.eq('calibration').to_numpy()].quantile(.95))
        keep = context.window_position.isin(np.flatnonzero(test)).to_numpy()
        d, v, positions = operators[keep], directions[keep], context.window_position.to_numpy()[keep]
        norm = np.linalg.norm(d, axis=(1, 2)); h = -d/norm[:, None, None]
        aggregate = lambda lam: pd.Series(np.maximum(lam, 0)**.25).groupby(positions).max()
        exact = aggregate(np.linalg.eigvalsh(d)[:, -1])
        reference = exact >= threshold
        shots_rows = [dict(noise='exact_expectation', shots_per_setting=np.inf,
                           mean_agreement=float(((aggregate(np.einsum('bi,bij,bj->b', v, d, v)) >= threshold) == reference).mean()))]
        for noise_name, noise in NOISE.items():
            for shots in shot_levels:
                agreement = [float(((aggregate(-norm*ShotEstimator(shots, noise, seed=7*r+1).expectation(v, h)) >= threshold) == reference).mean())
                             for r in range(20)]
                shots_rows.append(dict(noise=noise_name, shots_per_setting=shots, mean_agreement=float(np.mean(agreement)),
                                       min_agreement=float(np.min(agreement))))
        pd.DataFrame(shots_rows).assign(scale=minutes, decision='transient_only', windows=len(reference),
            exact_warnings=int(reference.sum()), near_threshold_windows_5pct=int((np.abs(exact-threshold)/threshold < .05).sum())
        ).to_csv(output/name, index=False)
    references = _normalize_incidents(pd.read_csv(REGISTRY))
    frame, meta, calibration, common = load_scale('primary')
    e = excess(frame, meta, NINE)
    calibration_mask, test = common & frame.phase.eq('calibration').to_numpy(), common & frame.phase.eq('test').to_numpy()
    levels = {'M0': list(BEHAVIORAL_COLUMNS), 'M1': NINE[:-1], 'M2': NINE}
    seeds, midp = [], []

    class _MidP:
        """Stand-in generator returning U = 0.5 (non-randomised mid-p values)."""

        def uniform(self, size=None):
            """Return an array of 0.5 with the requested shape."""
            return np.full(size, .5)

    for level, columns in levels.items():
        for seed in range(50):
            fusion = ConformalEvidenceFusion(columns, ConformalFusionConfig(seed=seed)).fit(e[calibration_mask])
            p = fusion.transform(e[test])
            summary, scored, episodes = evaluate_scores(frame[test], p.fused_score, *fusion.thresholds, ModelDConfig(), references)
            dominant = p.loc[scored.warning.to_numpy(), 'dominant_evidence'].value_counts(normalize=True)
            candidate = scored[(scored.end_time >= '2022-08-23') & (scored.end_time < '2022-08-26')].warning.mean()
            seeds.append(dict(level=level, seed=seed, warning_windows=summary['warning_windows'], episodes=summary['episode_count'],
                overlap=summary['weak_reference_event_overlap_fraction'],
                command_share=float(dominant.get(COMMAND_COLUMN, 0)), transient_share=float(dominant.get(EXACT_COLUMN, 0)),
                candidate_23_25aug_warning_fraction=float(candidate)))
        fusion = ConformalEvidenceFusion(columns).fit(e[calibration_mask])
        p = fusion.transform(e[test])
        summary, scored, episodes = evaluate_scores(frame[test], p.fused_score, *fusion.thresholds, ModelDConfig(), references)
        episodes.assign(level=level).to_csv(output/f'conformal_episodes_{level}.csv', index=False)
        pvalues = np.column_stack([conformal_module.smoothed_conformal_pvalues(fusion.calibration[c], e[test][c].to_numpy(float), _MidP())
                                   for c in columns])
        fused = -np.log10(np.clip(conformal_module.cauchy_combination(pvalues), 1e-300, 1))
        summary, *_ = evaluate_scores(frame[test], fused, *fusion.thresholds, ModelDConfig(), references)
        midp.append(dict(level=level, warning_windows=summary['warning_windows'], episodes=summary['episode_count'],
                         overlap=summary['weak_reference_event_overlap_fraction']))
    pd.DataFrame(seeds).to_csv(output/'conformal_seed_sensitivity.csv', index=False)
    pd.DataFrame(midp).to_csv(output/'conformal_midp.csv', index=False)
    print(pd.DataFrame(seeds).groupby('level').agg(['min', 'median', 'max']).drop(columns='seed').round(3).to_string())
    print(pd.DataFrame(midp).to_string(index=False))
    order = pd.read_parquet(output/'order_statistics_primary.parquet')
    held_out = order.phase.eq('test')
    pd.DataFrame([dict(
        spearman_T4_kurtosis_all=float(spearmanr(order.fourth_order, order.kurtosis_ratio, nan_policy='omit')[0]),
        spearman_T4_kurtosis_test=float(spearmanr(order.fourth_order[held_out], order.kurtosis_ratio[held_out], nan_policy='omit')[0]),
        max_abs_T4_recomputation_error=float((order.fourth_order-order[EXACT_COLUMN]).abs().max()))]
    ).to_csv(output/'consistency_checks.csv', index=False)


def main():
    """Run the selected extension stages and verify read-only inputs.

    Raises:
        RuntimeError: If any non-HTML file of the primary or multiscale
            folders changed during the run.
        ValueError: Propagated from :func:`stage_telemetry` on hash mismatch.

    Notes:
        The module docstring is also the ``--help`` text (raw formatting).
        Options: ``--stage`` (``model_a``, ``quantum``, ``telemetry``,
        ``fusion``, ``robustness`` or ``all``), ``--output-dir`` and ``--data-dir`` (Paths).
        Updates ``extensions_metadata.json`` in the output folder.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=['model_a', 'quantum', 'telemetry', 'fusion', 'robustness', 'all'], default='all')
    parser.add_argument('--output-dir', type=Path, default=Path('reports/extensions'))
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    args = parser.parse_args(); output = args.output_dir; output.mkdir(parents=True, exist_ok=True)
    before = {str(p): _hash(p) for p in [*PRIMARY.rglob('*'), *MULTISCALE.rglob('*')] if p.is_file() and p.suffix != '.html'}
    tic = time.perf_counter()
    _, _, model, _ = restore_electrical(PRIMARY)
    stages = ['model_a', 'quantum', 'telemetry', 'fusion', 'robustness'] if args.stage == 'all' else [args.stage]
    for stage in stages:
        print(f'== {stage} ==', flush=True)
        if stage == 'model_a':
            stage_model_a(output)
        elif stage == 'quantum':
            stage_quantum(output, model.references)
        elif stage == 'telemetry':
            stage_telemetry(output, args.data_dir)
        elif stage == 'fusion':
            stage_fusion(output)
        else:
            stage_robustness(output)
    after = {str(p): _hash(p) for p in [*PRIMARY.rglob('*'), *MULTISCALE.rglob('*')] if p.is_file() and p.suffix != '.html'}
    if before != after:
        raise RuntimeError('Primary or multiscale artifacts changed')
    manifest = output/'extensions_metadata.json'
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}
    previous.update({f'{stage}_completed': pd.Timestamp.now(tz='UTC').isoformat() for stage in stages})
    previous.update(read_only_inputs_unchanged=True, noise_models={k: v.__dict__ for k, v in NOISE.items()},
        source_code_sha256={str(p): _hash(p) for p in [Path(__file__), *Path('src/Model_D/iabm_incidents').glob('*.py')]},
        elapsed_seconds_last_run=time.perf_counter()-tic)
    manifest.write_text(json.dumps(previous, indent=2, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
