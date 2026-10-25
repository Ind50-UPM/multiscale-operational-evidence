"""Semi-synthetic fault injection into real residuals and order-wise electrical statistics.

Faults are injected into genuine nominal telemetry through the frozen
command observer, r = (y-c_s)/s_s, so the null distribution, switching,
gaps and serial dependence remain those of the real asset. Each injected
window has a known label, which the weakly documented incident inventory
cannot provide. Statistics are evaluated per controller-state context and
aggregated by the maximum, exactly as Tconc.

Role in the pipeline
--------------------
This module supports the controlled (semi-synthetic) validation of the
electrical evidence of :mod:`iabm_incidents.electrical`. It compares
order-wise statistics -- first order (mean shift), second order (covariance
deviation), fourth order (:math:`T_\\mathrm{conc}`) and information-geometric
shape measures from :mod:`iabm_incidents.quantum` -- on the same frozen
whitening, so differences in detection power are attributable to the moment
order and not to preprocessing.

Fault model
-----------
With observer centre :math:`c_s` and scale :math:`s_s`, a multiplicative loss
:math:`y' = (1-\\delta) y` of delivered power becomes, in residual units,

.. math::

    r' = (1-\\delta)\\, r - \\delta\\, c_s / s_s .

* ``under_response``: :math:`\\delta` = ``magnitude`` on all active samples of
  the selected ``phases``;
* ``dropouts``: the same map with :math:`\\delta` = ``magnitude`` on short
  events (``1`` collapses the phase to :math:`y' = 0`, i.e.
  :math:`r' = -c_s/s_s`);
* ``bursts``: additive :math:`\\pm a` (``magnitude``) in residual-scale units.

Event counts are Poisson with mean ``rate_per_hour * hours``; each event
affects ``length`` consecutive samples of one randomly chosen phase, restricted
to active samples.

.. mermaid::

    flowchart LR
        N["nominal residuals r (real asset)"] --> I["inject<br/>under_response / bursts / dropouts"]
        I --> W["window_statistics<br/>per state context, max aggregate"]
        R["TransientReference (F0, W)"] --> W
        S["SecondOrderReference (C0)"] --> W
        W --> O["mean_residual_norm, mean_shift,<br/>second_order, fourth_order,<br/>kurtosis_ratio, qjsd"]

Main public objects
-------------------
* :data:`STATISTICS` -- names of the exported statistics.
* :class:`SecondOrderReference` and :func:`fit_second_order`.
* :func:`context_statistics` and :func:`window_statistics`.
* :data:`SHAPE_COLUMNS` and :func:`window_shape_signature` (Pauli signature of
  the dominant shape change, for fault-type identification).
* :class:`Injection` and :func:`inject`.

Data contracts
--------------
Residual arrays have shape ``(N, 3)`` in residual-scale units, in time order;
``safe``, ``states`` and ``active`` are length-``N`` arrays; ``offsets`` has
shape ``(N, 3)`` and holds :math:`c_s/s_s` per sample and phase. Reference
dictionaries are keyed by ``(asset_id, state)`` as in
:attr:`~iabm_incidents.electrical.TransientConcentration.references`.

Maintenance notes
-----------------
* Settling/gap guards are *not* re-applied here; ``safe`` must already
  include them (e.g. the ``safe`` column of
  :meth:`~iabm_incidents.electrical.CommandResponseObserver.residuals`).
* :func:`fit_second_order` must be fitted on the same nominal partition as the
  transient references to keep the second- and fourth-order comparison fair.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .electrical import TransientReference, causal_delay, fourth_scatter
from .quantum import REAL_PAULI_LABELS, information_profile, pauli_vector, shape_change

STATISTICS = ("mean_residual_norm", "mean_shift", "second_order", "fourth_order", "kurtosis_ratio", "qjsd")
#: Column names of the 35 non-identity Pauli coefficients of the shape change
#: (the identity coefficient of a trace-zero matrix is always zero).
SHAPE_COLUMNS = tuple(f"pauli_{label}" for label in REAL_PAULI_LABELS if label != "III")


@dataclass(frozen=True)
class SecondOrderReference:
    """Frozen whitened second moment C0 of one state context.

    Attributes:
        moment: :math:`C_0 = \\frac1n\\sum z z^\\top` of the whitened nominal
            embedding, shape ``(8, 8)``.
        scale: Normalizer :math:`\\max(\\operatorname{tr}C_0/8, 10^{-8})`.
    """

    moment: np.ndarray
    scale: float


def fit_second_order(residuals: pd.DataFrame, references: dict, config) -> dict:
    """C0 = mean zzᵀ on the same reference partition and whitening as F0.

    Args:
        residuals: Nominal residual frame with ``residual_1..3``, ``safe`` and
            ``config.state_column``, in time order.
        references: Fitted transient references keyed by
            ``(asset_id, state)``.
        config: :class:`~iabm_incidents.electrical.ElectricalConfig` (only
            ``state_column`` and ``minimum_samples`` are used).

    Returns:
        Mapping ``(asset_id, state) -> SecondOrderReference`` for contexts with
        at least ``minimum_samples`` embedding rows.

    Notes:
        Rows are selected by state only; the asset of the key is not used to
        filter ``residuals``, so callers should pass a single-asset frame.
    """
    r = residuals[["residual_1", "residual_2", "residual_3"]].to_numpy(float)
    states = residuals[config.state_column].to_numpy()
    safe = residuals.safe.to_numpy(bool)
    out = {}
    for (asset, state), reference in references.items():
        x = causal_delay(r, safe & (states == state))
        if len(x) < config.minimum_samples:
            continue
        z = (x-reference.center) @ reference.whitening
        c0 = z.T @ z/len(z)
        out[(asset, state)] = SecondOrderReference(c0, max(np.trace(c0)/8, 1e-8))
    return out


def context_statistics(x: np.ndarray, reference: TransientReference, second: SecondOrderReference) -> dict[str, float]:
    """Order-wise statistics on one context's causal coordinates.

    ``second_order`` is [λ_max((C-C0)/s)]₊^{1/2} and ``fourth_order`` the
    paper's Tconc [λ_max((F-F0)/s0)]₊^{1/4}; both use the identical whitening.

    Args:
        x: Causal embedding rows of one context, shape ``(n, 8)``, ``n >= 1``.
        reference: Frozen transient reference of the context.
        second: Frozen second-order reference of the context.

    Returns:
        Dictionary with ``mean_shift``
        (:math:`\\sqrt8\\,\\lVert\\bar z\\rVert`), ``second_order``,
        ``fourth_order``, ``kurtosis_ratio`` (:math:`\\operatorname{tr}F /
        \\operatorname{tr}F_0`) and ``qjsd`` (quantum Jensen--Shannon
        divergence between trace-normalized :math:`F` and :math:`F_0`).

    See Also:
        :func:`iabm_incidents.quantum.information_profile`.
    """
    z = (x-reference.center) @ reference.whitening
    c = z.T @ z/len(z)
    d2 = (c-second.moment)/second.scale
    d4 = (fourth_scatter(z)-reference.scatter)/reference.scale
    d4 = (d4+d4.T)/2
    profile = information_profile(d4, reference)
    return dict(mean_shift=float(np.linalg.norm(z.mean(0))*np.sqrt(8)),
                second_order=max(float(np.linalg.eigvalsh((d2+d2.T)/2)[-1]), 0.)**.5,
                fourth_order=max(float(np.linalg.eigvalsh(d4)[-1]), 0.)**.25,
                kurtosis_ratio=profile["kurtosis_ratio"], qjsd=profile["qjsd"])


def window_statistics(r: np.ndarray, safe: np.ndarray, states: np.ndarray, references: dict,
                      seconds: dict, minimum_samples: int, asset: str = "asset-unknown") -> dict[str, float]:
    """Max over supported state contexts; NaN when no context is supported.

    Args:
        r: Residuals of one window, shape ``(N, 3)``, time ordered.
        safe: Safety mask of shape ``(N,)``; non-finite rows are additionally
            excluded.
        states: Controller state per sample, shape ``(N,)``.
        references: Transient references keyed by ``(asset_id, state)``.
        seconds: Second-order references with the same keys.
        minimum_samples: Minimum number of usable samples (for
            ``mean_residual_norm``) and of embedding rows per context.
        asset: Asset identifier used to build the reference keys.

    Returns:
        Dictionary over :data:`STATISTICS`; each context statistic is the
        maximum over supported contexts, and ``mean_residual_norm`` is the mean
        Euclidean norm of the usable residuals.
    """
    out = {name: np.nan for name in STATISTICS}
    ok = safe & np.isfinite(r).all(1)
    if ok.sum() >= minimum_samples:
        out["mean_residual_norm"] = float(np.linalg.norm(r[ok], axis=1).mean())
    for state in pd.unique(states):
        reference, second = references.get((asset, state)), seconds.get((asset, state))
        x = causal_delay(r, ok & (states == state))
        if reference is None or second is None or len(x) < minimum_samples:
            continue
        for name, value in context_statistics(x, reference, second).items():
            out[name] = value if not np.isfinite(out[name]) else max(out[name], value)
    return out


def window_shape_signature(r: np.ndarray, safe: np.ndarray, states: np.ndarray, references: dict,
                           minimum_samples: int, asset: str = "asset-unknown") -> dict[str, float]:
    r"""Pauli signature of the shape change of the dominant transient context.

    For every supported controller-state context the fourth-order scatter
    :math:`F_{w,c}` and its score :math:`T^{(4)}_{w,c}` are computed as in
    :func:`context_statistics`; the context with the largest score is
    retained, and the 35 non-identity Pauli coefficients of
    :math:`\Delta\rho=\rho(F_{w,c})-\rho(F_{0,c})` are returned. Where
    :math:`T^{(4)}_w` says *how much* the transient organisation changed, the
    signature says *in which direction* it changed, which is what separates,
    for instance, single-phase from three-phase under-response.

    Args:
        r: Residuals of one window, shape ``(N, 3)``, time ordered.
        safe: Safety mask of shape ``(N,)``; non-finite rows are excluded.
        states: Controller state per sample, shape ``(N,)``.
        references: Transient references keyed by ``(asset_id, state)``.
        minimum_samples: Minimum number of embedding rows per context.
        asset: Asset identifier used to build the reference keys.

    Returns:
        dict[str, float]: ``shape_context_state`` (state of the retained
        context), ``shape_fourth_order`` (its :math:`T^{(4)}`) and the
        :data:`SHAPE_COLUMNS`; all NaN when no context is supported.

    Notes:
        The coefficients are a linear, invertible reparametrisation of the
        entries of :math:`\Delta\rho`; any identification gain comes from
        using the full operator rather than a scalar summary, while the Pauli
        basis gives each coefficient a three-qubit measurement and a reading in
        terms of the embedding bits.
    """
    out = {"shape_context_state": np.nan, "shape_fourth_order": np.nan, **{c: np.nan for c in SHAPE_COLUMNS}}
    ok = safe & np.isfinite(r).all(1)
    best = None
    for state in pd.unique(states):
        reference = references.get((asset, state))
        x = causal_delay(r, ok & (states == state))
        if reference is None or len(x) < minimum_samples:
            continue
        z = (x-reference.center) @ reference.whitening
        f = fourth_scatter(z)
        d4 = (f-reference.scatter)/reference.scale
        score = max(float(np.linalg.eigvalsh((d4+d4.T)/2)[-1]), 0.)**.25
        # Note: ties keep the first context, matching max() in window_statistics.
        if best is None or score > best[0]:
            best = (score, state, (f+f.T)/2, reference)
    if best is None:
        return out
    delta = shape_change(best[2], best[3])
    coefficients = pauli_vector((delta+delta.T)/2)
    out.update(shape_context_state=float(best[1]), shape_fourth_order=best[0])
    out.update({c: float(v) for c, v in zip(SHAPE_COLUMNS, coefficients[1:])})
    return out


@dataclass(frozen=True)
class Injection:
    """One prespecified fault scenario applied to samples in active states.

    ``kind`` is ``under_response`` (multiplicative loss δ of delivered power on
    ``phases``), ``bursts`` (sign-random transients of ``amplitude`` residual
    scales and ``length`` samples at ``rate_per_hour``) or ``dropouts``
    (short loss of a fraction ``magnitude`` of one phase's power, e.g.
    contactor chatter; 1 means collapse to zero).

    Attributes:
        kind: ``"under_response"``, ``"bursts"`` or ``"dropouts"``.
        magnitude: :math:`\\delta` (fraction, typically in ``[0, 1]``) for
            ``under_response``/``dropouts``; burst amplitude in residual-scale
            units for ``bursts``. Must be nonnegative.
        rate_per_hour: Poisson event rate for ``bursts``/``dropouts``.
        length: Samples per event, ``>= 1``.
        phases: Zero-based phase indices affected (all of them for
            ``under_response``; one is drawn per event otherwise).

    Raises:
        ValueError: On unknown ``kind``, negative ``magnitude`` or
            ``rate_per_hour``, or ``length < 1``.
    """

    kind: str
    magnitude: float
    rate_per_hour: float = 0.
    length: int = 1
    phases: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        """Validate the scenario.

        Raises:
            ValueError: On unknown ``kind``, negative ``magnitude`` or
                ``rate_per_hour``, or ``length < 1``.
        """
        if self.kind not in ("under_response", "bursts", "dropouts"):
            raise ValueError("Unknown injection kind")
        if self.magnitude < 0 or self.rate_per_hour < 0 or self.length < 1:
            raise ValueError("Nonnegative magnitudes and positive burst length required")


def inject(r: np.ndarray, states: np.ndarray, active: np.ndarray, offsets: np.ndarray,
           injection: Injection, rng: np.random.Generator, sample_period_seconds: float = 1.,
           return_events: bool = False):
    """Return injected residuals; ``offsets`` holds c_s/s_s per sample and phase.

    For y' = (1-δ)y the residual becomes (1-δ)r-δc/s; a dropout sets y'=0,
    i.e. r'=-c/s. Bursts add ±a in residual-scale units. Idle samples are untouched.

    Args:
        r: Nominal residuals, shape ``(N, 3)``; not modified.
        states: Controller state per sample (unused by the current fault
            kinds; kept for interface stability).
        active: Boolean mask of shape ``(N,)`` of samples eligible for
            injection.
        offsets: :math:`c_s/s_s` per sample and phase, shape ``(N, 3)``.
        injection: Fault scenario.
        rng: Random generator driving event count, starts, phases and signs.
        sample_period_seconds: Sampling period used to convert ``N`` samples
            into hours for the Poisson mean. Pass the real telemetry period
            (0.807 s for the case study); the default of one second is only
            correct for exactly 1-Hz data.
        return_events: When ``True`` also return the number of events actually
            applied (always 0 for ``under_response``), so that window-level
            detection power can be conditioned on windows that received a fault.

    Returns:
        numpy.ndarray | tuple[numpy.ndarray, int]: Injected copy of ``r`` with
        shape ``(N, 3)``, or ``(injected, n_events)`` when ``return_events``.

    Notes:
        The Poisson mean uses the full record duration ``N * period``, not only
        the active time. Event starts are drawn without replacement among
        active samples and capped at their number; spans are truncated at the
        record end and filtered to active samples. Overlapping bursts add up.
    """
    out = np.array(r, float, copy=True)
    phases = list(injection.phases)
    if injection.kind == "under_response":
        delta = injection.magnitude
        out[np.ix_(active, phases)] = (1-delta)*out[np.ix_(active, phases)]-delta*offsets[np.ix_(active, phases)]
        return (out, 0) if return_events else out
    candidates = np.flatnonzero(active)
    if not len(candidates):
        return (out, 0) if return_events else out
    hours = len(r)*sample_period_seconds/3600
    count = rng.poisson(injection.rate_per_hour*hours)
    # Note: start draws are without replacement, so at most one event starts
    # per sample, although spans of different events may overlap.
    applied = min(count, len(candidates))
    for start in rng.choice(candidates, size=applied, replace=False):
        span = np.arange(start, min(start+injection.length, len(r)))
        span = span[active[span]]
        phase = rng.choice(phases)
        if injection.kind == "bursts":
            out[span, phase] += rng.choice([-1., 1.])*injection.magnitude
        else:
            m = injection.magnitude
            out[span, phase] = (1-m)*out[span, phase]-m*offsets[span, phase]
    return (out, int(applied)) if return_events else out
