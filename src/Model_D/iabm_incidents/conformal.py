r"""Conformal evidence p-values and Cauchy-combination fusion across dimensions/scales.

Each directional-excess coordinate is converted into a smoothed conformal
p-value against the frozen calibration partition, so heterogeneous evidence
(recurrence counts, divergences, residual norms, Tconc) is placed on one
calibrated scale without hand-tuned weights. The Cauchy combination (ACAT) of
Liu & Xie (2020) fuses arbitrarily dependent p-values with an approximately
exact level in the upper tail, and the minimum-p coordinate gives an
attribution of which evidence dimension (and scale) drove each warning.
Validity is marginal and assumes calibration exchangeability; overlapping
windows weaken this, so calibration warning rates are reported, not assumed.

Role in the pipeline:
    An alternative to the weighted linear score of
    :func:`~iabm_incidents.detection.compute_deviation_score` and the
    quantile thresholds of :mod:`iabm_incidents.ablation`. Larger evidence
    values must mean "more anomalous" (directional excesses), because the
    p-value counts calibration scores that are *greater than or equal to*
    the evaluated score.

Main public objects:
    * :func:`smoothed_conformal_pvalues` -- per-coordinate randomized p-values.
    * :func:`cauchy_combination` -- row-wise weighted ACAT fusion.
    * :class:`ConformalFusionConfig` -- prespecified onset/recovery levels.
    * :class:`ConformalEvidenceFusion` -- fit/transform wrapper.

Data contract:
    ``fit`` receives the calibration-phase evidence table and ``transform``
    any (typically test-phase) table; both must contain every column listed
    in ``columns`` as float-coercible values. Non-finite evaluated values
    yield NaN p-values and are dropped from the fusion. The calibration table
    must come from a chronologically earlier, disjoint partition (it is never
    refitted inside ``transform``).

Equations:
    For a calibration sample :math:`c_1,\dots,c_n` (finite values only) and
    an evaluated score :math:`s`,

    .. math::

       p(s) = \frac{\#\{i : c_i > s\} + U\,(\#\{i : c_i = s\} + 1)}{n + 1},
       \qquad U \sim \mathcal{U}(0, 1).

    The weighted Cauchy combination of :math:`K` p-values is

    .. math::

       T = \frac{\sum_k w_k \tan\big((0.5 - p_k)\pi\big)}{\sum_k w_k},
       \qquad p_{\mathrm{fused}} = \frac{1}{2} - \frac{\arctan T}{\pi},

    and the fused evidence score is :math:`-\log_{10} p_{\mathrm{fused}}`.

.. mermaid::

   flowchart LR
       C[calibration evidence] -->|fit| F[ConformalEvidenceFusion]
       E[evaluated evidence] -->|transform| F
       F --> P["p_coordinate (smoothed conformal)"]
       P --> A[cauchy_combination]
       A --> S["p_fused, fused_score = -log10 p_fused"]
       P --> D["dominant_evidence = argmin p"]

Maintenance notes:
    Randomization is reproducible: ``transform`` re-seeds a fresh
    :class:`numpy.random.Generator` from ``config.seed`` on every call, so
    repeated calls on the same input return identical p-values, and the
    random stream depends on the column order. Numerical guards (clipping to
    ``[1e-15, 1 - 1e-15]``, the ``1/(p\pi)`` asymptote for tiny p and the
    ``1/(T\pi)`` tail for huge T) must be preserved when refactoring.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def smoothed_conformal_pvalues(calibration: np.ndarray, scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    r"""Compute smoothed (randomized) conformal p-values against calibration scores.

    .. math::

       p(s) = \frac{\#\{c > s\} + U\,(\#\{c = s\} + 1)}{n + 1}

    with an independent :math:`U \sim \mathcal{U}(0,1)` per evaluated score.
    Under exchangeability of the evaluated score with the calibration
    sample, :math:`p(s)` is exactly uniform on ``(0, 1)``.

    Args:
        calibration: 1-D array of calibration scores; non-finite entries are
            discarded before ranking.
        scores: Array of evaluated scores (any shape). Higher means more
            anomalous.
        rng: Random generator used for the tie-breaking uniforms; consumes
            ``scores.size`` draws.

    Returns:
        Float array with the same shape as ``scores``; entries for
        non-finite scores are NaN.

    Raises:
        ValueError: If ``calibration`` contains no finite value.
    """
    c = np.sort(np.asarray(calibration, float)[np.isfinite(calibration)])
    s = np.asarray(scores, float)
    if not len(c):
        raise ValueError("Finite calibration scores required")
    greater = len(c)-np.searchsorted(c, s, side="right")
    equal = np.searchsorted(c, s, side="right")-np.searchsorted(c, s, side="left")
    p = (greater+rng.uniform(size=s.shape)*(equal+1))/(len(c)+1)
    return np.where(np.isfinite(s), p, np.nan)


def cauchy_combination(pvalues: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    r"""Fuse p-values row-wise with the weighted Cauchy combination test (ACAT).

    .. math::

       T_b = \frac{\sum_k w_k\,\mathbb{1}[p_{bk}\ \text{finite}]\,
                   \tan\big((0.5 - p_{bk})\pi\big)}
                  {\sum_k w_k\,\mathbb{1}[p_{bk}\ \text{finite}]},
       \qquad p_b = \frac{1}{2} - \frac{\arctan T_b}{\pi}.

    NaN coordinates are dropped and the remaining weights renormalised.

    Args:
        pvalues: ``B x K`` array of p-values (rows are windows, columns are
            evidence coordinates); NaN/inf entries mean "unavailable".
        weights: Optional length-``K`` non-negative weights; defaults to
            equal weights.

    Returns:
        Length-``B`` array of fused p-values clipped to ``[0, 1]``; NaN for
        rows with zero total available weight.

    Raises:
        ValueError: If ``pvalues`` is not 2-D, ``weights`` has the wrong
            shape, or any weight is negative.

    Notes:
        P-values are clipped to ``[1e-15, 1 - 1e-15]``. For clipped
        :math:`p < 10^{-12}` the tangent is replaced by its asymptote
        :math:`1/(p\pi)`, and for :math:`T > 10^{12}` the fused value uses
        :math:`1/(T\pi)`, avoiding catastrophic cancellation in both tails.
        A 1-D ``pvalues`` with ``weights=None`` fails with ``IndexError``
        before the shape check is reached.
    """
    p = np.asarray(pvalues, float)
    w = np.ones(p.shape[1]) if weights is None else np.asarray(weights, float)
    if p.ndim != 2 or w.shape != (p.shape[1],) or (w < 0).any():
        raise ValueError("B x K p-values and K nonnegative weights required")
    ok = np.isfinite(p)
    weight = np.where(ok, w, 0.)
    total = weight.sum(1)
    clipped = np.clip(np.where(ok, p, .5), 1e-15, 1-1e-15)
    # tan((0.5-p)π) is computed as 1/(pπ) for tiny p to avoid cancellation.
    t = np.where(clipped < 1e-12, 1/(clipped*np.pi), np.tan((.5-clipped)*np.pi))
    statistic = (weight*t).sum(1)/np.where(total > 0, total, np.nan)
    combined = np.where(statistic > 1e12, 1/(statistic*np.pi), .5-np.arctan(statistic)/np.pi)
    return np.where(total > 0, np.clip(combined, 0, 1), np.nan)


@dataclass(frozen=True)
class ConformalFusionConfig:
    """Prespecified conformal levels for onset and recovery decisions.

    The levels mirror the 95th/80th-percentile onset/recovery policy of
    :class:`~iabm_incidents.ablation.AblationConfig`: a window whose fused
    p-value is at most ``onset_alpha`` corresponds to an onset-level
    exceedance, and recovery uses the looser ``recovery_alpha``.

    Attributes:
        onset_alpha: Onset significance level in ``(0, recovery_alpha)``.
        recovery_alpha: Recovery significance level in ``(onset_alpha, 1)``.
        seed: Seed of the generator used for p-value tie randomization.

    Raises:
        ValueError: If ``0 < onset_alpha < recovery_alpha < 1`` does not hold.
    """

    onset_alpha: float = .05
    recovery_alpha: float = .20
    seed: int = 20260901

    def __post_init__(self) -> None:
        """Validate the ordering of the significance levels.

        Raises:
            ValueError: If ``0 < onset_alpha < recovery_alpha < 1`` fails.
        """
        if not 0 < self.onset_alpha < self.recovery_alpha < 1:
            raise ValueError("Require 0 < onset alpha < recovery alpha < 1")


class ConformalEvidenceFusion:
    """Fit on calibration evidence, then emit per-coordinate and fused p-values.

    Examples:
        >>> import pandas as pd
        >>> cal = pd.DataFrame({"a": [0.1, 0.2, 0.3, 0.4], "b": [1.0, 2.0, 3.0, 4.0]})
        >>> fusion = ConformalEvidenceFusion(["a", "b"]).fit(cal)
        >>> out = fusion.transform(pd.DataFrame({"a": [0.5], "b": [5.0]}))
        >>> sorted(out.columns)
        ['dominant_evidence', 'fused_score', 'p_a', 'p_b', 'p_fused']

    Attributes:
        columns: Ordered evidence coordinate names.
        config: Significance levels and randomization seed.
        calibration: Mapping from coordinate name to the stored calibration
            array (empty until :meth:`fit` is called).
    """

    def __init__(self, columns: list[str], config: ConformalFusionConfig = ConformalFusionConfig()) -> None:
        """Create an unfitted fusion object.

        Args:
            columns: Non-empty list of evidence column names to calibrate.
            config: Significance levels and seed.

        Raises:
            ValueError: If ``columns`` is empty.
        """
        if not columns:
            raise ValueError("At least one evidence coordinate required")
        self.columns, self.config = list(columns), config
        self.calibration: dict[str, np.ndarray] = {}

    def fit(self, calibration: pd.DataFrame) -> ConformalEvidenceFusion:
        """Store the calibration evidence of every coordinate.

        Args:
            calibration: Calibration-phase evidence table containing every
                column in :attr:`columns`. Non-finite values are kept here
                and discarded at p-value time.

        Returns:
            ``self``, to allow chaining.

        Raises:
            KeyError: If a configured column is missing.

        Notes:
            Only the calibration partition should be passed; using test rows
            here would leak evaluation data into the reference distribution.
        """
        self.calibration = {c: calibration[c].to_numpy(float) for c in self.columns}
        return self

    def transform(self, evidence: pd.DataFrame) -> pd.DataFrame:
        """Convert evidence into conformal p-values, fused p-value and attribution.

        Args:
            evidence: Table containing every column in :attr:`columns`.

        Returns:
            DataFrame indexed like ``evidence`` with columns
            ``p_<coordinate>`` (one per coordinate, in order), ``p_fused``
            (equal-weight Cauchy combination), ``fused_score``
            (:math:`-\\log_{10}` of ``p_fused`` clipped to ``[1e-300, 1]``) and
            ``dominant_evidence`` (name of the coordinate with the smallest
            finite p-value; ``None`` when no coordinate is finite).

        Raises:
            ValueError: If :meth:`fit` has not been called.
            KeyError: If a configured column is missing from ``evidence``.

        Notes:
            The generator is re-seeded from ``config.seed`` on every call,
            so results are deterministic per call. Ties in the minimum
            p-value resolve to the first coordinate in :attr:`columns`.
        """
        if not self.calibration:
            raise ValueError("Fit the conformal fusion on calibration evidence first")
        rng = np.random.default_rng(self.config.seed)
        out = pd.DataFrame(index=evidence.index)
        for c in self.columns:
            out[f"p_{c}"] = smoothed_conformal_pvalues(self.calibration[c], evidence[c].to_numpy(float), rng)
        p = out[[f"p_{c}" for c in self.columns]].to_numpy(float)
        out["p_fused"] = cauchy_combination(p)
        out["fused_score"] = -np.log10(np.clip(out.p_fused, 1e-300, 1))
        filled = np.where(np.isfinite(p), p, np.inf)
        out["dominant_evidence"] = np.array(self.columns, dtype=object)[filled.argmin(1)]
        out.loc[~np.isfinite(p).any(1), "dominant_evidence"] = None
        return out

    @property
    def thresholds(self) -> tuple[float, float]:
        """Onset/recovery thresholds on the ``-log10`` fused-p scale.

        Returns:
            Tuple ``(-log10(onset_alpha), -log10(recovery_alpha))``; with the
            defaults approximately ``(1.301, 0.699)``. Compare these with
            ``fused_score`` from :meth:`transform`.
        """
        return -np.log10(self.config.onset_alpha), -np.log10(self.config.recovery_alpha)
