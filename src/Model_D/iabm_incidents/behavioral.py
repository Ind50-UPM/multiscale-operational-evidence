r"""Chronologically fitted seven-coordinate behavioral encoder for raw telemetry.

This supplies a complete M0 when retained Model D tables contain only semantic
scores. It is a new matched raw-command implementation, not a replay of the
historical semantic baseline. All references are frozen before evaluated windows.

Role in the pipeline:
    Produces the seven behavioural indicators that form level M0 of the
    matched ablation (:data:`iabm_incidents.ablation.BEHAVIORAL_COLUMNS`)
    directly from per-sample controller states, command-response residuals
    and retained Model_A predictions, so that M0..M3 are evaluated on the
    same raw-telemetry windows.

Main public objects:
    * :class:`BehavioralWindowEncoder` -- ``fit`` on nominal samples, then
      ``encode`` one already sliced window at a time.

Data contract:
    Per-sample frames, sorted by time, with columns named by
    :class:`~iabm_incidents.electrical.ElectricalConfig`:

    * ``config.state_column`` (default ``controller_state``): six-bit
      controller command state, an integer in ``[0, 64)`` stored as a number.
    * ``config.timestamp_column`` (default ``timestamp``): ``datetime64``
      timestamps; consecutive differences larger than
      ``maximum_gap_seconds`` are treated as gaps.
    * ``encode`` additionally needs ``residual_1..3`` (signed standardized
      phase residuals from
      :class:`~iabm_incidents.electrical.CommandResponseObserver`), a boolean
      ``safe`` column and ``pred_estado`` (retained Model_A state prediction,
      NaN where unavailable).

    Samples are assumed to be regularly spaced by ``sample_period_seconds``;
    run durations are computed as ``run_length * sample_period_seconds``.

Phases:
    ``fit`` must only see the nominal reference partition; the fitted
    references (transition distribution, dwell medians, run rates, mode
    distribution, 95th-percentile occupancy) are then frozen and applied to
    calibration and test windows by ``encode``.

The seven coordinates (for one window):
    With :math:`\mathrm{JS}` the base-2 Jensen--Shannon divergence,

    .. math::

       D_{seq} &= \mathrm{JS}(\hat P_{trans}, P^{0}_{trans}), \\
       D_{dur} &= \text{mean over runs of } |\operatorname{med}(d_s) - m^0_s| / \sigma^0_s, \\
       R_{exc} &= \max_{s: \lambda^0_s > 0} \max\big(0, (N_s - \lambda^0_s T)/(\lambda^0_s T)\big), \\
       P_{exc} &= \max_s \max\big(0, (o_s - q^{0}_{95,s}) / \max(q^0_{95,s}, 1/n)\big), \\
       D_{cons} &= \text{mean of safe residuals } r_{1..3}, \\
       E_{state} &= \Pr(\hat u^{A} \ne u \mid \hat u^{A}\ \text{available}), \\
       D_{mode} &= \mathrm{JS}(\hat P_{popcount(u)}, P^0_{popcount(u)}),

    where :math:`d_s` are run durations of state :math:`s`, :math:`m^0_s,
    \sigma^0_s` the nominal dwell median and :math:`\max(1.4826\,
    \mathrm{MAD}, \Delta t)`, :math:`N_s` the run count, :math:`\lambda^0_s`
    the nominal run rate per observed second, :math:`T` the observed time,
    :math:`o_s` the occupancy fraction and :math:`n` the number of samples.

Maintenance notes:
    Known issue: under pandas copy-on-write, the gap mask obtained from
    ``Series.to_numpy()`` in :meth:`BehavioralWindowEncoder.encode` can be a
    read-only view, so the in-place ``gaps[1:] |= ...`` update raises
    (observed in ``test_behavioral_partial_nominal_support_and_gap_mask``).
    Transition pairs are encoded as ``64 * from + to`` (4096 bins), and mode
    is the number of active control bits (popcount, 7 bins).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .electrical import ElectricalConfig


def _runs(states: np.ndarray, gap: np.ndarray, dt: float):
    """Compress a state sequence into runs that never bridge timestamp gaps.

    A new run starts at index 0, at every state change and at every sample
    flagged as a gap.

    Args:
        states: 1-D integer array of controller states.
        gap: Boolean array of the same length; ``True`` marks a sample that
            follows a timestamp gap (or a removed sample).
        dt: Sample period in seconds.

    Returns:
        Tuple ``(run_states, run_durations, pair_valid)`` where
        ``run_states`` is the state of each run, ``run_durations`` its
        length times ``dt`` (seconds), and ``pair_valid`` (length
        ``n_runs - 1``) is ``True`` when the transition into the next run is
        not caused by a gap and may therefore be counted as a transition.
    """
    change = np.r_[True, states[1:] != states[:-1]] | gap
    starts = np.flatnonzero(change)
    counts = np.diff(np.r_[starts, len(states)])
    return states[starts], counts*dt, ~gap[starts[1:]]


def _distribution(values: np.ndarray, size: int):
    """Return the empirical distribution of non-negative integer codes.

    Args:
        values: 1-D array of integer codes in ``[0, size)`` (larger codes
            extend the output beyond ``size``).
        size: Minimum number of bins.

    Returns:
        Float array of relative frequencies; all zeros for empty input
        (the denominator is floored at 1).
    """
    counts = np.bincount(values, minlength=size).astype(float)
    return counts/max(counts.sum(), 1.)


def _js(p, q):
    r"""Base-2 Jensen--Shannon divergence between two discrete distributions.

    .. math::

       \mathrm{JS}(p, q) = \tfrac12 \mathrm{KL}(p \,\|\, m) + \tfrac12 \mathrm{KL}(q \,\|\, m),
       \qquad m = (p + q)/2.

    Args:
        p: Probability vector.
        q: Probability vector of the same length.

    Returns:
        Divergence in bits, in ``[0, 1]`` for normalized inputs. Zero-mass
        bins are skipped (``0 log 0 = 0``).
    """
    mid = (p+q)/2
    def term(a):
        """Return the KL contribution of ``a`` against the midpoint ``mid``."""
        use = a > 0
        return np.sum(a[use]*np.log2(a[use]/mid[use]))
    return float((term(p)+term(q))/2)


class BehavioralWindowEncoder:
    """Fit command-run transition, duration, recurrence, persistence/mode references.

    Dseq is Jensen--Shannon divergence of compressed command-run transitions;
    Ddur is frequency-weighted per-state median dwell drift/MAD; Rexc is the
    maximum positive run-count excess against frozen observed-time rates;
    Pexc is maximum occupancy excess against nominal 95th percentiles; Dcons
    is mean signed state-standardized electrical residual; Estate is actual
    command versus retained Model A disagreement on available matched records;
    Dmode is Jensen--Shannon divergence of active-control-bit-count distribution.

    Estate retains its prediction-coverage fraction: unavailable predictions
    are not counted as correct. Mode is an explicit command-activity proxy,
    not a reconstruction of Model C's complete semantic mode ontology.

    Attributes:
        config: Telemetry contract and support policy.
        fitted: ``True`` once :meth:`fit` completed successfully.
        transitions: Nominal distribution over the 4096 transition codes
            ``64 * from + to`` (set by :meth:`fit`).
        dwell: Mapping ``state -> (median_seconds, scale_seconds)`` with
            ``scale = max(1.4826 * MAD, sample_period_seconds)``.
        run_rates: Length-64 array of nominal runs per observed second.
        modes: Nominal distribution of active-bit counts (7 bins).
        occupancy95: Length-64 array of the per-state 95th percentile of
            window occupancy fractions over nominal rolling windows.

    See Also:
        :mod:`iabm_incidents.ablation` for how the coordinates are scaled,
        thresholded and compared across levels.
    """

    def __init__(self, config: ElectricalConfig = ElectricalConfig()):
        """Create an unfitted encoder.

        Args:
            config: Telemetry column names, sampling period, gap and support
                thresholds.
        """
        self.config = config
        self.fitted = False

    def fit(self, nominal: pd.DataFrame, window_length: pd.Timedelta = pd.Timedelta(hours=24),
            step: pd.Timedelta = pd.Timedelta(hours=1)) -> BehavioralWindowEncoder:
        """Fit only the supplied nominal training samples, excluding timestamp gaps.

        Args:
            nominal: Time-ordered nominal reference samples of a single
                contiguous partition (see the module data contract). Every
                state must be a known six-bit value.
            window_length: Length of the rolling windows used for the
                occupancy reference; should match the evaluation windows.
            step: Stride of those rolling windows; window starts are aligned
                to ``ceil(first_timestamp, step)``.

        Returns:
            ``self``, fitted.

        Raises:
            ValueError: If any state is non-finite, outside ``[0, 64)`` or
                non-integer, or if the partition is too short to contain a
                single full rolling window.

        Notes:
            Transitions across timestamp gaps are excluded, but dwell runs
            are still split at gaps. Run rates are normalized by the total
            sample time ``len(u) * sample_period_seconds``. Occupancy windows
            are half-open ``[start, start + window_length)`` located by
            :func:`numpy.searchsorted`, so ``nominal`` must be sorted by
            timestamp. The method does not sort or group by asset.
        """
        u = nominal[self.config.state_column].to_numpy(float)
        if not np.isfinite(u).all() or ((u < 0) | (u >= 64) | (u != np.round(u))).any():
            raise ValueError('Behavioral encoder requires known six-bit controller states')
        u = u.astype(int)
        gaps = nominal[self.config.timestamp_column].diff().dt.total_seconds().gt(self.config.maximum_gap_seconds).to_numpy()
        states, durations, pair_valid = _runs(u, gaps, self.config.sample_period_seconds)
        pairs = (states[:-1]*64+states[1:])[pair_valid]
        self.transitions = _distribution(pairs,4096)
        self.dwell = {}
        for state in np.unique(states):
            values = durations[states == state]
            median = float(np.median(values)); mad = float(1.4826*np.median(abs(values-median)))
            self.dwell[int(state)] = median, max(mad,self.config.sample_period_seconds)
        self.run_rates = np.bincount(states,minlength=64)/(len(u)*self.config.sample_period_seconds)
        modes = np.array([int(state).bit_count() for state in u])
        self.modes = _distribution(modes,7)
        occupancies = []
        times = nominal[self.config.timestamp_column].to_numpy(dtype='datetime64[ns]')
        current = pd.Timestamp(times[0]).ceil(step)
        while current+window_length <= pd.Timestamp(times[-1]):
            lo, hi = np.searchsorted(times,[current.to_datetime64(), (current+window_length).to_datetime64()])
            if hi > lo:
                occupancies.append(_distribution(u[lo:hi],64))
            current += step
        if not occupancies:
            raise ValueError('Nominal partition too short for behavioral occupancy reference')
        self.occupancy95 = np.quantile(occupancies,.95,axis=0)
        self.fitted = True
        return self

    def encode(self, samples: pd.DataFrame, minimum_predictions: int = 32) -> dict:
        """Encode one already sliced window; NaN marks missing model support.

        Args:
            samples: Time-ordered samples of exactly one evaluation window
                (see the module data contract for required columns).
            minimum_predictions: Minimum number of available Model_A
                predictions required to report ``state_error_rate``.

        Returns:
            Dictionary with the seven coordinates ``sequence_divergence``,
            ``duration_drift``, ``recurrence_excess``, ``persistence_excess``,
            ``consumption_deviation``, ``state_error_rate`` and
            ``mode_divergence``, plus the audit fields
            ``controller_known_fraction``,
            ``behavioral_nominal_state_fraction``,
            ``state_prediction_fraction`` and ``state_prediction_samples``.
            When support is insufficient, only the seven coordinates are
            returned, all NaN.

        Raises:
            ValueError: If the encoder has not been fitted.

        Notes:
            Support gates, in order: the fraction of known six-bit states
            must reach ``minimum_safe_fraction`` and their count
            ``minimum_samples``; then the fraction of samples whose state
            has a nominal dwell reference must reach
            ``minimum_safe_fraction``. Unknown-state samples are removed and
            their positions are treated as gaps so runs and transitions do
            not bridge them. ``consumption_deviation`` averages all three
            residual phases over safe, finite samples (signed, so drops are
            negative). ``state_error_rate`` is NaN when fewer than
            ``minimum_predictions`` predictions exist; missing predictions
            are never counted as correct.
        """
        if not self.fitted:
            raise ValueError('Fit the nominal behavioral encoder first')
        cfg = self.config
        u = samples[cfg.state_column].to_numpy(float)
        finite = np.isfinite(u) & (u >= 0) & (u < 64) & (u == np.round(u))
        known_fraction = float(finite.mean()) if len(u) else 0.
        original_positions = np.flatnonzero(finite)
        if known_fraction < cfg.minimum_safe_fraction or finite.sum() < cfg.minimum_samples:
            return {c:np.nan for c in ('sequence_divergence','duration_drift','recurrence_excess','persistence_excess','consumption_deviation','state_error_rate','mode_divergence')}
        if not finite.all():
            samples = samples.loc[finite]
            u = u[finite]
        u = u.astype(int)
        gaps = samples[cfg.timestamp_column].diff().dt.total_seconds().gt(cfg.maximum_gap_seconds).to_numpy()
        # Why: removed unknown-state samples must split runs like timestamp gaps.
        # Note: under pandas copy-on-write ``gaps`` may be read-only, making
        # this in-place update fail (known failing test; not fixed here).
        gaps[1:] |= np.diff(original_positions) > 1
        states, durations, pair_valid = _runs(u,gaps,cfg.sample_period_seconds)
        pairdist = _distribution((states[:-1]*64+states[1:])[pair_valid],4096)
        nominal_state_fraction = float(np.isin(u, list(self.dwell)).mean())
        if nominal_state_fraction < cfg.minimum_safe_fraction:
            return {c:np.nan for c in ('sequence_divergence','duration_drift','recurrence_excess','persistence_excess','consumption_deviation','state_error_rate','mode_divergence')}
        dwell = []
        for state in np.unique(states):
            if state not in self.dwell:
                continue
            median, scale = self.dwell[state]
            dwell.extend([abs(np.median(durations[states == state])-median)/scale]*int((states == state).sum()))
        counts = np.bincount(states,minlength=64)
        # Note: expected run counts scale nominal per-second rates by the
        # window's retained sample time; states never seen nominally are skipped.
        expected = self.run_rates*len(u)*cfg.sample_period_seconds
        occupancy = _distribution(u,64)
        active = expected > 0
        recurrence = float(np.maximum((counts[active]-expected[active])/expected[active],0).max())
        persistence = float(np.maximum((occupancy-self.occupancy95)/np.maximum(self.occupancy95,1/len(u)),0).max())
        residual = samples[['residual_1','residual_2','residual_3']].to_numpy(float)
        safe = samples.safe.to_numpy(bool) & np.isfinite(residual).all(1)
        pred = samples.pred_estado.to_numpy(float)
        predicted = np.isfinite(pred)
        error = float(np.mean(pred[predicted] != u[predicted])) if predicted.sum() >= minimum_predictions else np.nan
        return dict(sequence_divergence=_js(pairdist,self.transitions), duration_drift=float(np.mean(dwell)),
            recurrence_excess=recurrence,persistence_excess=persistence,
            consumption_deviation=float(residual[safe].mean()) if safe.any() else np.nan,
            state_error_rate=error, mode_divergence=_js(_distribution(np.array([int(state).bit_count() for state in u]),7),self.modes),
            controller_known_fraction=known_fraction,
            behavioral_nominal_state_fraction=nominal_state_fraction,
            state_prediction_fraction=float(predicted.mean()), state_prediction_samples=int(predicted.sum()))
