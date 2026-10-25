"""Frozen command-response residuals and solver-independent fourth-order evidence.

Role in the pipeline
--------------------
This module produces the electrical evidence channel of Model D. Raw
three-phase telemetry (for example the ``RP1``/``RP2``/``RP3`` power readings)
is first explained by the *commanded* controller state through a frozen,
per-asset, per-state observer. The remaining signed residuals are embedded in a
causal eight-coordinate representation, whitened against a frozen nominal
reference and summarized by a fourth-order (kurtosis-type) scatter operator.
The largest eigenvalue of the normalized deviation operator defines the
transient-concentration score :math:`T_\\mathrm{conc}`.

The eight-coordinate causal representation and normalized scatter follow the
QIP branch of CReST_Bode_and_Ablation_Atlas. Fit on declared nominal
partitions; neither observer nor reference is fitted implicitly by the incident
pipeline.

Main public objects
-------------------
* :class:`ElectricalConfig` -- telemetry contract and support policy.
* :class:`CommandResponseObserver` -- robust same-state observer producing
  :math:`r = (y - c_s)/s_s`.
* :func:`causal_delay` -- causal eight-coordinate embedding.
* :func:`fourth_scatter` -- degree-four scatter matrix (FOBI kurtosis matrix).
* :class:`DominantDirectionSolver` -- protocol for optional direction solvers.
* :class:`VQEDirectionSolver` -- three-qubit RY/CX variational eigensolver
  (classical statevector simulation) optimized by Rotosolve.
* :class:`TransientReference` -- frozen per-context whitening and nominal
  scatter.
* :class:`TransientConcentration` -- fits references and scores windows.

Algorithm
---------
For each controller state :math:`s` of each asset the observer stores a robust
centre :math:`c_s` (median) and scale :math:`s_s` (MAD-based, floored) per
phase, so that

.. math::

    r_t = (y_t - c_{s_t}) / s_{s_t} \\in \\mathbb{R}^3 .

The causal embedding at time :math:`t` is

.. math::

    x_t = \\bigl(r_t,\\; r_{t-1},\\; \\tfrac{1}{\\sqrt3}\\mathbf 1^\\top r_{t-2},
          \\; \\tfrac{1}{\\sqrt3}\\mathbf 1^\\top r_{t-3}\\bigr) \\in \\mathbb{R}^8 ,

i.e. the three phases at lag 0 and lag 1 plus the common mode at lags 2 and 3.
With a shrinkage whitening :math:`W` and centre :math:`m` estimated on nominal
data, :math:`z = (x - m) W` and

.. math::

    F = \\mathbb{E}\\Bigl[\\frac{\\lVert z\\rVert^2}{p}\\, z z^\\top\\Bigr],
    \\qquad p = 8 .

For exactly white Gaussian :math:`z` the null value is
:math:`\\frac{p+2}{p} I`; in practice the empirical nominal :math:`F_0` is used.
The deviation operator and score are

.. math::

    D = \\frac{F - F_0}{s_0}, \\quad s_0 = \\max(\\operatorname{tr}F_0/8, 10^{-8}),
    \\qquad T_\\mathrm{conc} = [\\lambda_{\\max}(D)]_+^{1/4} .

The exact :math:`\\lambda_{\\max}` is always computed by ``numpy.linalg.eigvalsh``.
An optional :class:`DominantDirectionSolver` (e.g. :class:`VQEDirectionSolver`)
returns a unit direction :math:`v` whose Rayleigh quotient
:math:`v^\\top D v` is exported *separately* as ``transient_concentration_vqe``;
it never replaces the exact score.

.. mermaid::

    flowchart LR
        Y["telemetry y (RP1..RP3, controller_state)"] --> O["CommandResponseObserver<br/>r=(y-c_s)/s_s, safe mask"]
        O --> E["causal_delay<br/>8 coords (lag0, lag1, CM lag2, CM lag3)"]
        E --> W["TransientReference<br/>centre, shrinkage whitening W, F0, s0"]
        W --> D["TransientConcentration.operator<br/>D=(F-F0)/s0"]
        D --> X["exact eigvalsh -> Tconc"]
        D --> V["optional VQEDirectionSolver<br/>Rayleigh v'Dv -> Tconc_vqe"]

.. mermaid::

    classDiagram
        class ElectricalConfig {
          +channels
          +state_column
          +timestamp_column
          +minimum_samples
          +minimum_safe_fraction
          +settling_samples
          +maximum_gap_seconds
          +sample_period_seconds
          +shrinkage
          +inverse_floor
        }
        class CommandResponseObserver {
          +profiles
          +fit(nominal)
          +residuals(samples)
        }
        class DominantDirectionSolver {
          <<Protocol>>
          +solve(operator)
        }
        class VQEDirectionSolver {
          +layers
          +sweeps
          +restarts
          +seed
          +solve(operator)
          +solve_batch(operators)
          #_energy(psi, h)
        }
        class TransientReference {
          +center
          +whitening
          +scatter
          +scale
        }
        class TransientConcentration {
          +references
          +solver
          +fit(nominal_residuals)
          +operator(x, reference)
          +score_windows(residuals, windows)
        }
        DominantDirectionSolver <|.. VQEDirectionSolver
        CommandResponseObserver --> ElectricalConfig
        TransientConcentration --> ElectricalConfig
        TransientConcentration o-- TransientReference
        TransientConcentration o-- DominantDirectionSolver

Data contracts
--------------
* Telemetry frames contain ``timestamp`` (any value parseable by
  :func:`pandas.to_datetime`; converted to naive UTC), the three channel
  columns (float), the controller-state column (any hashable value; compared by
  equality) and optionally ``asset_id`` (missing values become
  ``"asset-unknown"``; always cast to ``str``).
* Residual frames (output of :meth:`CommandResponseObserver.residuals`) add
  ``residual_1``, ``residual_2``, ``residual_3`` (float, residual-scale units)
  and ``safe`` (bool).
* Window frames contain ``window_id``, ``asset_id``, ``start_time`` and
  ``end_time``; windows are half-open, ``start_time <= t < end_time``.
* Duplicate ``(asset_id, timestamp)`` pairs are rejected.

Maintenance notes
-----------------
* Every fit is explicit; scoring never refits observer or reference, which is
  the leakage guard separating nominal partitions from evaluation windows.
* Samples are "safe" only after ``settling_samples`` regular samples since the
  last state switch or gap; the embedding never bridges unsafe samples.
* ``VQEDirectionSolver._energy`` is the single hook through which noisy
  subclasses (see ``quantum.py``) substitute a finite-shot or noisy estimator
  for the exact expectation; keep its signature stable.
* :meth:`VQEDirectionSolver._state` duplicates the circuit used inside
  :meth:`VQEDirectionSolver.solve_batch`; both must be changed together.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ElectricalConfig:
    """Telemetry contract and support policy (timestamps use half-open windows).

    Immutable configuration shared by the observer, the transient reference and
    the window scorer. Values are validated in :meth:`__post_init__`.

    Attributes:
        channels (tuple[str, str, str]): Names of the three distinct electrical
            phase columns, in phase order 1, 2, 3.
        state_column (str): Column holding the commanded controller state used
            to condition the observer and the reference.
        timestamp_column (str): Column holding sample timestamps.
        minimum_samples (int): Minimum number of usable samples (or causal
            embedding rows) required to fit or score a context; at least 4.
        minimum_safe_fraction (float): Minimum coverage, in ``(0, 1]``, of safe
            and supported samples required to declare a window supported.
        settling_samples (int): Number of regular samples after a state switch
            or a gap before a sample is considered settled (``>= 0``).
        maximum_gap_seconds (float): Largest admissible inter-sample spacing in
            seconds; larger gaps reset the settling counter (``> 0``).
        sample_period_seconds (float): Nominal sampling period in seconds, used
            to convert sample counts into time coverage (``> 0``).
        shrinkage (float): Ledoit--Wolf-style shrinkage weight in ``[0, 1]``
            towards a scaled identity when whitening the 8-coordinate embedding.
        inverse_floor (float): Positive relative eigenvalue floor (fraction of
            the mean variance) guarding the inverse square root.

    Raises:
        ValueError: If the phases are not three distinct names, if the support,
            settling/gap or regularization parameters are outside their
            allowed ranges.

    Examples:
        >>> ElectricalConfig().channels
        ('RP1', 'RP2', 'RP3')
    """

    channels: tuple[str, str, str] = ("RP1", "RP2", "RP3")
    state_column: str = "controller_state"
    timestamp_column: str = "timestamp"
    minimum_samples: int = 32
    minimum_safe_fraction: float = 0.9
    settling_samples: int = 3
    maximum_gap_seconds: float = 2.0
    sample_period_seconds: float = 1.0
    shrinkage: float = 0.1
    inverse_floor: float = 1e-6

    def __post_init__(self) -> None:
        """Validate the configuration invariants.

        Raises:
            ValueError: If ``channels`` is not three distinct names, if
                ``minimum_samples < 4`` or ``minimum_safe_fraction`` is outside
                ``(0, 1]``, if ``settling_samples < 0`` or either time
                parameter is non-positive, or if ``shrinkage`` is outside
                ``[0, 1]`` or ``inverse_floor <= 0``.
        """
        if len(self.channels) != 3 or len(set(self.channels)) != 3:
            raise ValueError("Three distinct electrical phases required")
        if self.minimum_samples < 4 or not 0 < self.minimum_safe_fraction <= 1:
            raise ValueError("Invalid electrical support policy")
        if self.settling_samples < 0 or self.maximum_gap_seconds <= 0 or self.sample_period_seconds <= 0:
            raise ValueError("Invalid settling/gap policy")
        if not 0 <= self.shrinkage <= 1 or self.inverse_floor <= 0:
            raise ValueError("Invalid covariance regularization")


class CommandResponseObserver:
    """Robust same-state observer; known switching is excluded by settling guards.

    For every ``(asset_id, controller_state)`` context the observer stores a
    per-phase robust centre and scale estimated on nominal data. Residuals are
    the standardized deviation of each phase from the response expected for the
    commanded state.

    This deliberately simple observer must be validated on separate nominal data
    before use. More detailed command/age/FIR observers can supply residuals to
    ``TransientConcentration`` directly. No measured-output lags are used.

    Attributes:
        config (ElectricalConfig): Telemetry contract and support policy.
        profiles (dict[tuple[str, object], tuple[numpy.ndarray, numpy.ndarray]]):
            Mapping ``(asset_id, state) -> (center, scale)`` where both arrays
            have shape ``(3,)`` in channel units. Empty until :meth:`fit`.

    Notes:
        The centre is the per-phase median :math:`c_s`; the scale is

        .. math::

            s_s = \\max\\bigl(1.4826\\,\\mathrm{MAD},\\;
                  \\max(0.01\\,|c_s|, 10^{-6})\\bigr),

        i.e. a Gaussian-consistent MAD floored at 1 % of the centre magnitude
        (and an absolute :math:`10^{-6}`), so quasi-constant states cannot
        produce exploding residuals.

    See Also:
        :class:`TransientConcentration`, :func:`causal_delay`.
    """

    def __init__(self, config: ElectricalConfig = ElectricalConfig()) -> None:
        """Create an unfitted observer.

        Args:
            config (ElectricalConfig): Telemetry contract; the default instance
                uses ``RP1``-``RP3`` and ``controller_state``.
        """
        self.config = config
        self.profiles: dict[tuple[str, object], tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, nominal: pd.DataFrame) -> CommandResponseObserver:
        """Fit robust per-asset, per-state centres and scales on nominal data.

        Only settled samples (see :func:`_safe_history`) with finite values on
        all three channels are used. Contexts with fewer than
        ``config.minimum_samples`` such samples are skipped. Any previous
        profiles are discarded.

        Args:
            nominal (pandas.DataFrame): Declared nominal telemetry with the
                timestamp, state and channel columns and optionally
                ``asset_id``.

        Returns:
            CommandResponseObserver: ``self``, for chaining.

        Raises:
            ValueError: If timestamps are duplicated within an asset (from
                :func:`_ordered`), if timestamps cannot be parsed, or if no
                context has enough support.
        """
        frame = _ordered(nominal, self.config)
        self.profiles = {}
        for asset, group in frame.groupby("asset_id", sort=False):
            stable = _safe_history(group, self.config)
            for state, rows in group.loc[stable].groupby(self.config.state_column):
                y = rows[list(self.config.channels)].to_numpy(float)
                y = y[np.isfinite(y).all(1)]
                if len(y) < self.config.minimum_samples:
                    continue
                center = np.median(y, axis=0)
                # Why: the relative/absolute floor keeps near-constant states
                # (MAD ~ 0) from producing unbounded residuals.
                scale = np.maximum(1.4826 * np.median(abs(y-center), axis=0),
                                   np.maximum(abs(center)*0.01, 1e-6))
                self.profiles[(str(asset), state)] = (center, scale)
        if not self.profiles:
            raise ValueError("No supported nominal command states")
        return self

    def residuals(self, samples: pd.DataFrame) -> pd.DataFrame:
        """Return signed phase residuals; unknown states remain NaN/unsafe.

        Applies the frozen profiles, :math:`r = (y - c_s)/s_s`, per phase. Rows
        whose ``(asset_id, state)`` context was not fitted keep NaN residuals
        and are therefore unsafe.

        Args:
            samples (pandas.DataFrame): Telemetry with the same contract as the
                nominal data passed to :meth:`fit`.

        Returns:
            pandas.DataFrame: A copy of ``samples`` sorted by
            ``(asset_id, timestamp)`` with a fresh ``RangeIndex``, naive-UTC
            timestamps and the added columns ``residual_1``, ``residual_2``,
            ``residual_3`` (float, residual-scale units) and ``safe`` (bool).
            ``safe`` is true only for settled samples with finite residuals
            whose absolute value is below ``1e6``.

        Raises:
            ValueError: If the observer has not been fitted, or if timestamps
                are duplicated within an asset or cannot be parsed.
        """
        frame = _ordered(samples, self.config)
        r = np.full((len(frame), 3), np.nan)
        if not self.profiles:
            raise ValueError("Fit the command observer on nominal data first")
        # Note: ``frame`` has a RangeIndex after ``_ordered``, so group labels
        # double as positional indices into ``r``.
        for (asset, state), indices in frame.groupby(["asset_id", self.config.state_column]).groups.items():
            profile = self.profiles.get((str(asset), state))
            if profile is not None:
                center, scale = profile
                r[indices] = (frame.loc[indices, list(self.config.channels)].to_numpy(float)-center)/scale
        frame[["residual_1", "residual_2", "residual_3"]] = r
        safe = np.zeros(len(frame), bool)
        for _, group in frame.groupby("asset_id", sort=False):
            safe[group.index] = _safe_history(group, self.config)
        frame["safe"] = safe & np.isfinite(r).all(1) & (np.abs(r).max(1) < 1e6)
        return frame


def _ordered(samples: pd.DataFrame, config: ElectricalConfig) -> pd.DataFrame:
    """Return a normalized, sorted copy of a telemetry frame.

    The copy gets a string ``asset_id`` column (missing column or values become
    ``"asset-unknown"``), timestamps parsed as UTC and converted to naive UTC,
    rows sorted by ``(asset_id, timestamp)`` and a fresh ``RangeIndex``.

    Args:
        samples (pandas.DataFrame): Telemetry or residual frame containing
            ``config.timestamp_column``.
        config (ElectricalConfig): Telemetry contract.

    Returns:
        pandas.DataFrame: The normalized copy; the input is not modified.

    Raises:
        ValueError: If any ``(asset_id, timestamp)`` pair is duplicated, or if
            timestamps cannot be parsed (raised by :func:`pandas.to_datetime`
            with ``errors="raise"``).
    """
    frame = samples.copy()
    if "asset_id" not in frame:
        frame["asset_id"] = "asset-unknown"
    frame["asset_id"] = frame.asset_id.fillna("asset-unknown").astype(str)
    frame[config.timestamp_column] = pd.to_datetime(frame[config.timestamp_column], utc=True, errors="raise").dt.tz_convert(None)
    frame = frame.sort_values(["asset_id", config.timestamp_column]).reset_index(drop=True)
    if frame.duplicated(["asset_id", config.timestamp_column]).any():
        raise ValueError("Duplicate telemetry timestamps within an asset")
    return frame


def _safe_history(group: pd.DataFrame, config: ElectricalConfig) -> np.ndarray:
    """Return the settling mask of one time-ordered single-asset block.

    A sample is *regular* when its spacing to the previous row is in
    ``(0, maximum_gap_seconds]``. The settling age restarts at every
    non-regular sample (including the first row, whose spacing is undefined)
    and at every controller-state change. A sample is safe when it is regular
    and its age is at least ``settling_samples``.

    Args:
        group (pandas.DataFrame): Rows of one asset sorted by timestamp, with
            datetime ``config.timestamp_column`` and ``config.state_column``.
        config (ElectricalConfig): Settling and gap policy.

    Returns:
        numpy.ndarray: Boolean array of shape ``(len(group),)``.

    Notes:
        Because the first row of ``group`` always resets the age, applying this
        function to a window slice re-imposes settling at the window start.
    """
    states = group[config.state_column].to_numpy()
    dt = group[config.timestamp_column].diff().dt.total_seconds().to_numpy()
    if not len(group):
        return np.empty(0, bool)
    # Note: dt[0] is NaN, so the first row is never regular and always resets.
    regular = (dt > 0) & (dt <= config.maximum_gap_seconds)
    reset = ~regular
    reset[1:] |= states[1:] != states[:-1]
    positions = np.arange(len(group))
    # Invariant: running maximum of reset positions gives, for each row, the
    # position of the most recent reset; age counts samples since that reset.
    age = positions-np.maximum.accumulate(np.where(reset, positions, 0))
    return regular & (age >= config.settling_samples)



def causal_delay(residual: np.ndarray, safe: np.ndarray) -> np.ndarray:
    """RP1–3 now/lag1, common mode lag2/lag3; never bridge unsafe samples.

    Builds the causal eight-coordinate embedding

    .. math::

        x_t = \\bigl(r_t,\\; r_{t-1},\\; \\tfrac{1}{\\sqrt3}\\textstyle\\sum_k r_{t-2,k},
              \\; \\tfrac{1}{\\sqrt3}\\textstyle\\sum_k r_{t-3,k}\\bigr)

    for every row ``t >= 3`` and keeps only rows for which all four
    contributing samples are safe and the result is finite. Lags are defined
    by row position, so callers must pass rows in time order; the ``safe``
    mask (settling and gap guards) is what prevents bridging across gaps.

    Args:
        residual (numpy.ndarray): Array of shape ``(N, 3)`` with signed phase
            residuals in residual-scale units.
        safe (numpy.ndarray): Boolean array of shape ``(N,)``; ``False`` rows
            are excluded together with every embedding row that uses them.

    Returns:
        numpy.ndarray: Array of shape ``(M, 8)`` with ``M <= max(N - 3, 0)``;
        an empty ``(0, 8)`` array when ``N < 4``.

    Raises:
        ValueError: If ``residual`` is not ``N x 3`` or ``safe`` does not have
            shape ``(N,)``.

    Examples:
        >>> import numpy as np
        >>> causal_delay(np.ones((5, 3)), np.ones(5, bool)).shape
        (2, 8)
    """
    r = np.asarray(residual, float)
    ok = np.asarray(safe, bool)
    if r.ndim != 2 or r.shape[1] != 3 or ok.shape != (len(r),):
        raise ValueError("Expected N x 3 residuals and N safety flags")
    if len(r) < 4:
        return np.empty((0, 8))
    x = np.column_stack((r[3:], r[2:-1], r[1:-2].sum(1)/np.sqrt(3), r[:-3].sum(1)/np.sqrt(3)))
    valid = ok[3:] & ok[2:-1] & ok[1:-2] & ok[:-3] & np.isfinite(x).all(1)
    return x[valid]


def fourth_scatter(z: np.ndarray) -> np.ndarray:
    """F = mean[(||z||²/d) zzᵀ], a PSD degree-four scatter.

    Computes the FOBI kurtosis matrix

    .. math::

        F = \\frac1n \\sum_{i=1}^n \\frac{\\lVert z_i\\rVert^2}{d}\\, z_i z_i^\\top ,

    which is symmetric positive semidefinite. For white Gaussian coordinates
    its expectation is :math:`\\frac{d+2}{d} I`.

    Args:
        z (numpy.ndarray): Finite array of shape ``(n, d)`` with ``n, d >= 1``
            (typically whitened coordinates, ``d = 8``).

    Returns:
        numpy.ndarray: Array of shape ``(d, d)``.

    Raises:
        ValueError: If ``z`` is not a finite, nonempty two-dimensional array.

    Examples:
        >>> import numpy as np
        >>> fourth_scatter(np.eye(2)).tolist()
        [[0.25, 0.0], [0.0, 0.25]]
    """
    z = np.asarray(z, float)
    if z.ndim != 2 or not len(z) or not z.shape[1] or not np.isfinite(z).all():
        raise ValueError("Finite nonempty coordinate matrix required")
    return z.T @ ((np.sum(z*z, axis=1)/z.shape[1])[:, None]*z)/len(z)


class DominantDirectionSolver(Protocol):
    """Approximation contract; return a normalized direction for the same D.

    Any object with a :meth:`solve` method satisfying this contract can be
    passed to :class:`TransientConcentration` to compute an approximate
    Rayleigh-quotient score alongside the exact eigenvalue.

    See Also:
        :class:`VQEDirectionSolver`.
    """

    def solve(self, operator: np.ndarray) -> np.ndarray:
        """Return an approximate dominant unit eigenvector of ``operator``.

        Args:
            operator (numpy.ndarray): Real symmetric ``8 x 8`` operator
                :math:`D`.

        Returns:
            numpy.ndarray: Finite array of shape ``(8,)`` with unit Euclidean
            norm. :class:`TransientConcentration` rejects anything else.
        """
        ...


class VQEDirectionSolver:
    """Three-qubit real RY/CX statevector VQE maximizing <D> by Rotosolve.

    The 8-dimensional operator is identified with a three-qubit Hamiltonian
    :math:`H = -D/\\lVert D\\rVert_F`, so minimizing :math:`\\langle H\\rangle`
    maximizes the Rayleigh quotient of :math:`D`. The ansatz is a
    hardware-efficient real circuit starting from :math:`|000\\rangle`: each
    layer applies :math:`R_Y` to qubits 0, 1, 2 followed by a CX ladder
    ``(0->1, 1->2)`` on even layers and ``(2->1, 1->0)`` on odd layers. Only
    real amplitudes arise, so the state is a real unit vector in
    :math:`\\mathbb{R}^8`.

    Independent starts, no eigenvector initialization or exact-solver fallback.
    This is a classical simulator, not a claim of quantum advantage.

    Attributes:
        layers (int): Number of RY+CX layers (``3 * layers`` angles).
        sweeps (int): Number of Rotosolve coordinate sweeps per restart.
        restarts (int): Number of independent starts; restart 0 uses all-zero
            angles, later restarts draw angles uniformly in ``[-pi, pi)``.
        seed (int): Seed of the generator re-created on every call to
            :meth:`solve_batch`, making results deterministic per call.

    Notes:
        Qubit 0 is the most significant bit of the basis index. The ``_energy``
        hook lets noisy subclasses replace the exact expectation estimator
        while keeping circuit and optimizer unchanged.

    See Also:
        :class:`DominantDirectionSolver`, :class:`TransientConcentration`.
    """

    def __init__(self, layers: int = 3, sweeps: int = 8, restarts: int = 2, seed: int = 0) -> None:
        """Configure the circuit depth and optimizer budget.

        Args:
            layers (int): Number of ansatz layers, ``>= 1``.
            sweeps (int): Rotosolve sweeps per restart, ``>= 1``.
            restarts (int): Independent restarts, ``>= 1``.
            seed (int): Seed for the random restart angles.

        Raises:
            ValueError: If any of ``layers``, ``sweeps`` or ``restarts`` is
                smaller than 1.
        """
        if min(layers, sweeps, restarts) < 1:
            raise ValueError("Positive VQE resource budgets required")
        self.layers, self.sweeps, self.restarts, self.seed = layers, sweeps, restarts, seed

    def _state(self, theta: np.ndarray) -> np.ndarray:
        """Simulate the ansatz statevector for a single angle vector.

        Unbatched reference implementation of the circuit used inside
        :meth:`solve_batch`; it is not called by the optimizer itself.

        Args:
            theta (numpy.ndarray): Angles of shape ``(3 * layers,)``, ordered
                layer-major then qubit.

        Returns:
            numpy.ndarray: Real unit statevector of shape ``(8,)``.
        """
        psi = np.zeros(8); psi[0] = 1
        basis = np.arange(8)
        for layer, angles in enumerate(theta.reshape(self.layers, 3)):
            for q, angle in enumerate(angles):
                # Note: qubit q is bit (2-q) of the basis index; ``zero``/``one``
                # pair amplitudes differing only in that bit, and the 2x2 RY
                # rotation [[c, -s], [s, c]] is applied to every pair.
                bit = 1 << (2-q); zero = basis[(basis & bit) == 0]; one = zero | bit
                a, b = psi[zero].copy(), psi[one].copy()
                c, s = np.cos(angle/2), np.sin(angle/2)
                psi[zero], psi[one] = c*a-s*b, s*a+c*b
            pairs = ((0, 1), (1, 2)) if layer % 2 == 0 else ((2, 1), (1, 0))
            for control, target in pairs:
                # Note: CX as an index permutation: flip the target bit of every
                # basis index whose control bit is 1 (an involution).
                psi = psi[basis ^ (((basis >> (2-control)) & 1) << (2-target))]
        return psi

    def _energy(self, psi: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Exact statevector expectation; noisy subclasses replace the estimator.

        Args:
            psi (numpy.ndarray): Batch of real statevectors, shape ``(B, 8)``.
            h (numpy.ndarray): Batch of real symmetric Hamiltonians, shape
                ``(B, 8, 8)``.

        Returns:
            numpy.ndarray: Energies :math:`\\psi_b^\\top h_b \\psi_b`, shape
            ``(B,)``.

        Notes:
            This is the only estimator hook; overriding it (e.g. with
            finite-shot or noisy estimates) leaves the circuit and Rotosolve
            loop untouched.
        """
        return np.einsum("bi,bij,bj->b", psi, h, psi)

    def solve(self, operator: np.ndarray) -> np.ndarray:
        """Return one variational direction; see ``solve_batch`` for resource policy.

        Args:
            operator (numpy.ndarray): Finite real ``8 x 8`` operator.

        Returns:
            numpy.ndarray: Real unit vector of shape ``(8,)``.

        Raises:
            ValueError: Propagated from :meth:`solve_batch` for invalid input.
        """
        return self.solve_batch(np.asarray(operator)[None])[0]

    def solve_batch(self, operators: np.ndarray) -> np.ndarray:
        """Solve independent identical-budget RY/CX circuits in vectorized batches.

        Each operator has its own angles and updates. Restart initial angles
        match the single-operator seeded policy. No classical eigenvectors or
        best-case filtering enter optimization, including difficult windows.

        Args:
            operators (numpy.ndarray): Finite real array of shape
                ``(B, 8, 8)``. Operators are normalized by their Frobenius norm
                (floored at ``1e-12``) and negated to form the Hamiltonians.

        Returns:
            numpy.ndarray: Array of shape ``(B, 8)`` with the best statevector
            per operator over all restarts; ``(0, 8)`` for an empty batch.

        Raises:
            ValueError: If ``operators`` is not a finite ``B x 8 x 8`` array.

        Notes:
            Rotosolve updates one angle at a time. With all other angles fixed
            the energy is a sinusoid
            :math:`E(\\theta_j) = a\\cos\\theta_j + b\\sin\\theta_j + c`, so
            three evaluations at :math:`0, \\pm\\pi/2` give
            :math:`a = E_0 - (E_+ + E_-)/2` and :math:`b = (E_+ - E_-)/2`, and
            the global minimizer is
            :math:`\\theta_j^* = \\operatorname{atan2}(b, a) + \\pi`. An update
            is rejected when it does not decrease the (possibly noisy)
            re-evaluated energy. The lowest final energy over restarts is
            retained per operator.
        """
        d = np.asarray(operators, float)
        if d.ndim != 3 or d.shape[1:] != (8, 8) or not np.isfinite(d).all():
            raise ValueError("VQE requires finite B x 8 x 8 real operators")
        n = len(d)
        if not n:
            return np.empty((0, 8))
        # Why: a fresh generator per call keeps solve() and solve_batch()
        # restart angles identical for the same seed.
        rng = np.random.default_rng(self.seed)
        h = -d/np.maximum(np.linalg.norm(d, axis=(1, 2)), 1e-12)[:, None, None]
        basis = np.arange(8)
        def state(theta: np.ndarray) -> np.ndarray:
            """Simulate the ansatz for a batch of angle vectors.

            Args:
                theta (numpy.ndarray): Angles of shape ``(B, 3 * layers)``.

            Returns:
                numpy.ndarray: Real unit statevectors of shape ``(B, 8)``.
            """
            psi = np.zeros((n, 8)); psi[:, 0] = 1
            for layer in range(self.layers):
                for q in range(3):
                    angle = theta[:, 3*layer+q]
                    # Note: same bit-pair RY update as ``_state``, broadcast
                    # over the batch axis.
                    bit = 1 << (2-q); zero = basis[(basis & bit) == 0]; one = zero | bit
                    a, b = psi[:, zero].copy(), psi[:, one].copy()
                    c, sn = np.cos(angle/2)[:, None], np.sin(angle/2)[:, None]
                    psi[:, zero], psi[:, one] = c*a-sn*b, sn*a+c*b
                pairs = ((0, 1), (1, 2)) if layer % 2 == 0 else ((2, 1), (1, 0))
                for control, target in pairs:
                    psi = psi[:, basis ^ (((basis >> (2-control)) & 1) << (2-target))]
            return psi
        def energy(theta: np.ndarray) -> np.ndarray:
            """Return the batch energies of the angle vectors ``theta``.

            Args:
                theta (numpy.ndarray): Angles of shape ``(B, 3 * layers)``.

            Returns:
                numpy.ndarray: Energies of shape ``(B,)`` from ``self._energy``.
            """
            return self._energy(state(theta), h)
        best = np.full(n, np.inf); best_theta = np.zeros((n, 3*self.layers))
        for restart in range(self.restarts):
            initial = np.zeros(3*self.layers) if restart == 0 else rng.uniform(-np.pi, np.pi, 3*self.layers)
            theta = np.tile(initial, (n, 1)); e = energy(theta)
            for _ in range(self.sweeps):
                for j in range(theta.shape[1]):
                    old, old_e = theta[:, j].copy(), e.copy()
                    theta[:, j] = 0; e0 = energy(theta)
                    theta[:, j] = np.pi/2; ep = energy(theta)
                    theta[:, j] = -np.pi/2; em = energy(theta)
                    # Note: closed-form Rotosolve minimizer of
                    # a*cos(t) + b*sin(t) + c with a = e0-(ep+em)/2, b = (ep-em)/2.
                    theta[:, j] = np.arctan2((ep-em)/2, e0-(ep+em)/2)+np.pi
                    e = energy(theta)
                    # Why: with noisy estimators the closed form can worsen the
                    # re-evaluated energy; keep the previous angle in that case.
                    reject = e > old_e
                    theta[reject, j], e[reject] = old[reject], old_e[reject]
            improve = e < best
            best[improve], best_theta[improve] = e[improve], theta[improve]
        return state(best_theta)


@dataclass(frozen=True)
class TransientReference:
    """Frozen context-specific whitening and fourth-order nominal operator.

    Attributes:
        center (numpy.ndarray): Coordinate-wise median of the nominal causal
            embedding, shape ``(8,)``.
        whitening (numpy.ndarray): Symmetric shrinkage whitening matrix
            :math:`W`, shape ``(8, 8)``; coordinates are ``(x - center) @ W``.
        scatter (numpy.ndarray): Nominal fourth-order scatter :math:`F_0`,
            shape ``(8, 8)``.
        scale (float): Normalizer :math:`s_0 = \\max(\\operatorname{tr}F_0/8,
            10^{-8})`.
    """

    center: np.ndarray
    whitening: np.ndarray
    scatter: np.ndarray
    scale: float


class TransientConcentration:
    """Define Tconc independently of the optional variational direction solver.

    Fits one :class:`TransientReference` per ``(asset_id, controller_state)``
    context on nominal residuals and scores evaluation windows with the exact
    largest eigenvalue of :math:`D = (F - F_0)/s_0`.

    Attributes:
        config (ElectricalConfig): Telemetry contract and support policy.
        solver (DominantDirectionSolver | None): Optional approximate solver
            whose Rayleigh score is reported separately.
        references (dict[tuple[str, object], TransientReference]): Fitted
            references keyed by ``(asset_id, state)``. Empty until :meth:`fit`.

    See Also:
        :class:`CommandResponseObserver`, :class:`VQEDirectionSolver`.
    """

    def __init__(self, config: ElectricalConfig = ElectricalConfig(), solver: DominantDirectionSolver | None = None) -> None:
        """Create an unfitted scorer.

        Args:
            config (ElectricalConfig): Telemetry contract and support policy.
            solver (DominantDirectionSolver | None): Optional direction solver;
                ``None`` disables the approximate score columns.
        """
        self.config, self.solver = config, solver
        self.references: dict[tuple[str, object], TransientReference] = {}

    def fit(self, nominal_residuals: pd.DataFrame) -> TransientConcentration:
        """Fit each asset/state on complete contiguous reference runs only.

        For every asset and state the causal embedding is built from rows that
        are flagged ``safe``, settled according to :func:`_safe_history` and in
        that state. Contexts with fewer than ``config.minimum_samples``
        embedding rows are skipped. Previous references are discarded.

        Args:
            nominal_residuals (pandas.DataFrame): Output of
                :meth:`CommandResponseObserver.residuals` on a declared nominal
                partition (needs ``residual_1..3``, ``safe``, state, timestamp
                and optionally ``asset_id``).

        Returns:
            TransientConcentration: ``self``, for chaining.

        Raises:
            ValueError: If no context has enough support, or if timestamps are
                duplicated or unparsable.

        Notes:
            With :math:`x_c = x - \\operatorname{median}(x)`,
            :math:`C = x_c^\\top x_c/n` and :math:`\\tau = \\max(\\operatorname{tr}C/8, 10^{-8})`,
            the regularized covariance is

            .. math::

                C_\\lambda = (1-\\lambda) C + (\\lambda + \\varepsilon)\\,\\tau I ,

            with :math:`\\lambda` = ``shrinkage`` and :math:`\\varepsilon` =
            ``inverse_floor``. The whitening is the symmetric inverse square
            root :math:`W = V \\operatorname{diag}(\\max(\\mu, \\varepsilon\\tau))^{-1/2} V^\\top`.
        """
        frame = _ordered(nominal_residuals, self.config)
        self.references = {}
        for asset, group in frame.groupby("asset_id", sort=False):
            r = group[["residual_1", "residual_2", "residual_3"]].to_numpy(float)
            states = group[self.config.state_column].to_numpy()
            safe = group.safe.to_numpy(bool) & _safe_history(group, self.config)
            for state in pd.unique(states):
                x = causal_delay(r, safe & (states == state))
                if len(x) < self.config.minimum_samples:
                    continue
                center = np.median(x, axis=0); xc = x-center
                cov = xc.T @ xc/len(xc); tau = max(np.trace(cov)/8, 1e-8)
                reg = (1-self.config.shrinkage)*cov+(self.config.shrinkage+self.config.inverse_floor)*tau*np.eye(8)
                values, vectors = np.linalg.eigh(reg)
                w = (vectors/np.sqrt(np.maximum(values, self.config.inverse_floor*tau))) @ vectors.T
                f0 = fourth_scatter(xc @ w)
                self.references[(str(asset), state)] = TransientReference(center, w, f0, max(np.trace(f0)/8, 1e-8))
        if not self.references:
            raise ValueError("No supported transient reference contexts")
        return self

    @staticmethod
    def operator(x: np.ndarray, reference: TransientReference) -> np.ndarray:
        """D = (F((x-center)W)-F0)/s0; shared by exact and VQE solvers.

        Args:
            x (numpy.ndarray): Causal embedding rows of one context, shape
                ``(n, 8)`` with ``n >= 1`` and finite values.
            reference (TransientReference): Frozen reference of that context.

        Returns:
            numpy.ndarray: Symmetrized ``8 x 8`` deviation operator.

        Raises:
            ValueError: Propagated from :func:`fourth_scatter` for empty or
                non-finite input.
        """
        d = (fourth_scatter((x-reference.center) @ reference.whitening)-reference.scatter)/reference.scale
        # Note: symmetrize to remove floating-point asymmetry before eigvalsh.
        return (d+d.T)/2

    def score_windows(self, residuals: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
        """Mean residual norm, signed phases and max supported context Tconc.

        Unsupported windows retain NaN and explicit coverage; the optional VQE
        Rayleigh score is exported separately and never replaces exact Tconc.

        For each window the rows of the same asset with
        ``start_time <= timestamp < end_time`` are selected. Settling is
        re-applied on the slice, so the first ``settling_samples`` rows of a
        window are never safe. Coverage is
        ``min(safe.mean(), safe.sum() * sample_period / duration, 1)``, which
        penalizes both unsafe rows and missing rows.

        Args:
            residuals (pandas.DataFrame): Output of
                :meth:`CommandResponseObserver.residuals`.
            windows (pandas.DataFrame): Windows with ``window_id``,
                ``asset_id``, ``start_time`` and ``end_time`` (half-open).

        Returns:
            pandas.DataFrame: One row per window with columns ``window_id``,
            ``asset_id``, ``command_response_deviation`` (mean residual norm),
            ``transient_concentration`` (exact :math:`T_\\mathrm{conc}`, max
            over supported state contexts), ``electrical_safe_fraction`` and
            ``electrical_status`` (``"supported"`` or ``"unsupported"``).
            When the safe-support gate passes it also contains
            ``command_response_signed_<channel>`` (mean signed residual),
            ``command_response_imbalance`` (mean across-phase standard
            deviation) and ``transient_supported_fraction``; with a solver and
            a supported score it also contains ``transient_concentration_vqe``
            and ``transient_solver_deficit`` (exact minus approximate). Columns
            absent for a given window are NaN in the result.

        Raises:
            ValueError: If the scorer is unfitted, a window has non-positive
                duration, timestamps are duplicated, or the solver returns a
                direction that is not a finite unit vector of shape ``(8,)``.
        """
        if not self.references:
            raise ValueError("Fit the transient reference on nominal data first")
        frame = _ordered(residuals, self.config)
        rows = []
        for window in windows.itertuples(index=False):
            start = pd.to_datetime(window.start_time, utc=True).tz_convert(None)
            end = pd.to_datetime(window.end_time, utc=True).tz_convert(None)
            # Invariant: half-open window [start, end).
            part = frame[(frame.asset_id == str(window.asset_id)) & (frame[self.config.timestamp_column] >= start) & (frame[self.config.timestamp_column] < end)]
            r = part[["residual_1", "residual_2", "residual_3"]].to_numpy(float)
            safe = part.safe.to_numpy(bool) & _safe_history(part, self.config) & np.isfinite(r).all(1)
            duration = (end-start).total_seconds()
            if duration <= 0:
                raise ValueError("Positive window duration required")
            coverage = min(float(safe.mean()), float(safe.sum()*self.config.sample_period_seconds/duration), 1.0) if len(safe) else 0.0
            row = {"window_id": window.window_id, "asset_id": window.asset_id,
                   "command_response_deviation": np.nan, "transient_concentration": np.nan,
                   "electrical_safe_fraction": coverage, "electrical_status": "unsupported"}
            scores, approximate = [], []
            if safe.sum() >= self.config.minimum_samples and coverage >= self.config.minimum_safe_fraction:
                rr = r[safe]
                row["command_response_deviation"] = float(np.linalg.norm(rr, axis=1).mean())
                for j, channel in enumerate(self.config.channels):
                    row[f"command_response_signed_{channel}"] = float(rr[:, j].mean())
                row["command_response_imbalance"] = float(np.std(rr, axis=1).mean())
                states = part[self.config.state_column].to_numpy()
                supported = np.zeros(len(part), bool)
                for state in pd.unique(states):
                    reference = self.references.get((str(window.asset_id), state))
                    x = causal_delay(r, safe & (states == state))
                    if reference is None or len(x) < self.config.minimum_samples:
                        continue
                    supported |= safe & (states == state)
                    d = self.operator(x, reference)
                    exact = float(np.linalg.eigvalsh(d)[-1])
                    scores.append(max(exact, 0.0)**0.25)
                    if self.solver is not None:
                        v = np.asarray(self.solver.solve(d), float)
                        if v.shape != (8,) or not np.isfinite(v).all() or not np.isclose(v @ v, 1):
                            raise ValueError("Solver returned an invalid normalized direction")
                        rayleigh = float(v @ d @ v)
                        approximate.append(max(rayleigh, 0.0)**0.25)
                row["transient_supported_fraction"] = min(float(supported.mean()), float(supported.sum()*self.config.sample_period_seconds/duration), 1.0)
                # Why: Tconc is only reported when the contexts with a frozen
                # reference cover enough of the window; otherwise it stays NaN.
                if scores and row["transient_supported_fraction"] >= self.config.minimum_safe_fraction:
                    row["transient_concentration"] = max(scores)
                    row["electrical_status"] = "supported"
                    if approximate:
                        row["transient_concentration_vqe"] = max(approximate)
                        row["transient_solver_deficit"] = max(scores)-max(approximate)
            rows.append(row)
        return pd.DataFrame(rows)
