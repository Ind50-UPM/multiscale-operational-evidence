r"""Quantum-information view, certificates and NISQ resource model for Tconc.

The fourth-order scatter F = E[(||z||²/p) zzᵀ] of whitened residuals is the
FOBI kurtosis matrix; under a Gaussian nominal it equals (p+2)/p · I. Its trace
is Mardia's multivariate kurtosis divided by p, and F/tr F is a real density
operator. This module separates magnitude (kurtosis ratio) from shape (quantum
Jensen–Shannon divergence and trace distance to the frozen nominal operator),
certifies variational decisions with Kato–Temple a-posteriori bounds and
simulates finite-shot, depolarising and readout-error estimation of the same
three-qubit Hamiltonian. Every routine is a classical simulation: no quantum
advantage is assumed or claimed.

Role in the pipeline:
    :mod:`~iabm_incidents.electrical` defines the scientific indicator
    Tconc :math:`= [\lambda_{\max}(D)]_+^{1/4}` with the context operator

    .. math::

        D = \frac{F - F_0}{s_0}, \qquad
        F = \frac{1}{n}\sum_i \frac{\lVert z_i\rVert^2}{p} z_i z_i^\top,
        \qquad s_0 = \max(\operatorname{tr} F_0 / p, 10^{-8}),

    on :math:`p = 8` causal whitened coordinates, i.e. a real symmetric
    :math:`8\times 8` matrix acting on three qubits. This module adds
    *supplementary* evidence and audits on top of that operator; it never
    changes Tconc, thresholds or detection. It is used by
    ``src/run_model_d_extensions.py`` (stage ``quantum``) and by
    :mod:`~iabm_incidents.injection`.

Main public objects:
    * Information profile: :func:`density_operator`,
      :func:`von_neumann_entropy`, :func:`quantum_jensen_shannon`,
      :func:`trace_distance`, :func:`scatter_from_operator`,
      :func:`information_profile`.
    * Shape-operator signatures: :data:`REAL_PAULI_LABELS`,
      :func:`pauli_vector`, :func:`shape_change` (used for fault-type
      identification in :mod:`iabm_incidents.identification`).
    * Resource estimates: :func:`swap_test_shots`,
      :func:`pauli_decomposition`, :func:`measurement_settings`.
    * NISQ simulation: :class:`NoiseModel`, :class:`ShotEstimator`,
      :class:`NoisyVQEDirectionSolver`.
    * Certificates: :func:`kato_temple_bounds`,
      :func:`deflated_second_direction`.

.. mermaid::

   classDiagram
       class VQEDirectionSolver {
           +layers: int
           +sweeps: int
           +restarts: int
           +seed: int
           +solve(operator) ndarray
           +solve_batch(operators) ndarray
           #_energy(psi, h) ndarray
       }
       class NoisyVQEDirectionSolver {
           +estimator: ShotEstimator
           #_energy(psi, h) ndarray
       }
       class ShotEstimator {
           +shots: int or None
           +noise: NoiseModel
           +layers: int
           +settings: dict
           +labels: list
           +coefficients(hamiltonians) ndarray
           +expectation(states, hamiltonians) ndarray
       }
       class NoiseModel {
           <<frozen dataclass>>
           +single_qubit_error: float
           +two_qubit_error: float
           +readout_error: float
           +mitigate: bool
           +fidelity(layers) float
       }
       VQEDirectionSolver <|-- NoisyVQEDirectionSolver
       NoisyVQEDirectionSolver o-- ShotEstimator
       ShotEstimator o-- NoiseModel

Magnitude versus shape:
    ``kurtosis_ratio`` :math:`= \operatorname{tr}F/\operatorname{tr}F_0`
    measures a uniform change of fourth-order energy (heavier tails in every
    direction). The quantum Jensen–Shannon divergence and the trace distance
    between :math:`\rho = F/\operatorname{tr}F` and
    :math:`\sigma = F_0/\operatorname{tr}F_0` compare trace-normalised
    *shapes* and are invariant to that uniform change.

Measurement model:
    For real symmetric :math:`H`, only the 36 Pauli strings with an even
    number of :math:`Y` factors have non-zero coefficients
    :math:`c_P = \operatorname{tr}(PH)/8`. The greedy qubit-wise-commuting
    grouping of :func:`measurement_settings` measures the 35 non-identity
    strings with 14 basis settings. Under the global depolarising model a
    non-identity Pauli expectation is attenuated by the circuit fidelity

    .. math::

        f = (1-p_1)^{3L}(1-p_2)^{2L},

    (three RY and two CX gates per layer, :math:`L` layers), and symmetric
    readout flips with probability :math:`e` attenuate a weight-:math:`k`
    string by :math:`(1-2e)^k`. Mitigation divides by
    :math:`f(1-2e)^k`, which is exact in expectation for this model.

Certificates:
    For a unit direction :math:`v`, :math:`\rho = v^\top D v \le
    \lambda_{\max}` is always a rigorous lower bound. With
    :math:`\sigma^2 = \lVert Dv\rVert^2 - \rho^2` and any
    :math:`\alpha \ge \lambda_2` with :math:`\rho > \alpha`, Kato–Temple gives
    :math:`\lambda_{\max} \le \rho + \sigma^2/(\rho - \alpha)`. The default
    gap reference :math:`\alpha_F = (\lVert D\rVert_F^2 - \rho^2)^{1/2}`
    satisfies :math:`\alpha_F \ge |\lambda_2|` whenever :math:`\rho \ge 0`
    (because :math:`\sum_i \lambda_i^2 = \lVert D\rVert_F^2` and
    :math:`\lambda_1 \ge \rho \ge 0`), and :math:`\lVert D\rVert_F` caps
    :math:`\lambda_{\max}`. If :math:`\rho < 0`, :math:`\rho > \alpha_F` is
    impossible, Kato–Temple is reported as not applicable and the
    :math:`\lVert D\rVert_F` cap is used. A VQD-estimated second eigenvalue
    is only a *lower* estimate of :math:`\lambda_2`, so that variant is not
    rigorous.

Maintenance notes:
    * **Qubit order:** qubit 0 is the most significant bit of the basis
      index, matching :class:`~iabm_incidents.electrical.VQEDirectionSolver`.
      ``PAULI``/``ROTATION`` Kronecker products and the ``bits`` table in
      :class:`ShotEstimator` must keep that convention.
    * ``ROTATION`` maps the measured basis onto Z before a computational-basis
      readout: Hadamard for X and :math:`HS^\dagger` for Y.
    * :class:`ShotEstimator` caches Pauli coefficients by *object identity*
      of the Hamiltonian batch; mutating that array in place between calls
      returns stale coefficients.
    * All randomness comes from seeded :class:`numpy.random.Generator`
      objects; a shared estimator's stream depends on call order.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np

from .electrical import TransientReference, VQEDirectionSolver

PAULI = {"I": np.eye(2), "X": np.array([[0., 1.], [1., 0.]]),
         "Y": np.array([[0., -1j], [1j, 0.]]), "Z": np.diag([1., -1.])}
# Basis change applied before a computational-basis measurement.
ROTATION = {"Z": np.eye(2), "X": np.array([[1., 1.], [1., -1.]])/np.sqrt(2),
            "Y": np.array([[1., -1j], [1., 1j]])/np.sqrt(2)}


def density_operator(scatter: np.ndarray) -> np.ndarray:
    r"""Return ρ = F/tr F for a PSD fourth-order scatter.

    The input is symmetrised as :math:`(F + F^\top)/2` before normalisation.
    Positive semidefiniteness is not checked; it holds for genuine scatter
    matrices.

    Args:
        scatter: Square real matrix (typically :math:`8\times 8`).

    Returns:
        numpy.ndarray: Unit-trace symmetric matrix.

    Raises:
        ValueError: If the trace is not finite or not strictly positive.

    Examples:
        >>> import numpy as np
        >>> float(np.trace(density_operator(2 * np.eye(8))))
        1.0
    """
    f = (np.asarray(scatter, float)+np.asarray(scatter, float).T)/2
    trace = np.trace(f)
    if not np.isfinite(trace) or trace <= 0:
        raise ValueError("Positive finite scatter trace required")
    return f/trace


def von_neumann_entropy(rho: np.ndarray) -> float:
    r"""Entropy in bits; negative round-off eigenvalues are discarded.

    .. math::

        S(\rho) = -\sum_i \lambda_i \log_2 \lambda_i,

    summing over eigenvalues :math:`\lambda_i > 10^{-15}` after clipping
    negatives to zero.

    Args:
        rho: Symmetric density operator.

    Returns:
        float: Entropy in bits, in :math:`[0, \log_2 d]`.

    Examples:
        >>> import numpy as np
        >>> round(von_neumann_entropy(np.eye(8) / 8), 12)
        3.0
    """
    values = np.clip(np.linalg.eigvalsh(rho), 0, None)
    values = values[values > 1e-15]
    return float(-(values*np.log2(values)).sum())


def quantum_jensen_shannon(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""QJSD in bits, bounded in [0, 1] and symmetric (Majtey et al. 2005).

    .. math::

        \mathrm{QJSD}(\rho,\sigma) = S\!\left(\frac{\rho+\sigma}{2}\right)
        - \frac{S(\rho) + S(\sigma)}{2}.

    Args:
        rho: Density operator.
        sigma: Density operator of the same shape.

    Returns:
        float: Divergence in bits; small negative round-off is clipped to 0.
    """
    return max(von_neumann_entropy((rho+sigma)/2)-(von_neumann_entropy(rho)+von_neumann_entropy(sigma))/2, 0.)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""½||ρ-σ||₁: optimal single-shot discrimination bias (Helstrom).

    Computed as half the sum of absolute eigenvalues of :math:`\rho-\sigma`
    (both assumed symmetric).

    Args:
        rho: Density operator.
        sigma: Density operator of the same shape.

    Returns:
        float: Trace distance in :math:`[0, 1]`.
    """
    return float(np.abs(np.linalg.eigvalsh(rho-sigma)).sum()/2)


def scatter_from_operator(operator: np.ndarray, reference: TransientReference) -> np.ndarray:
    r"""Invert D = (F-F0)/s0 to recover the window scatter F.

    Args:
        operator: Context operator :math:`D` (``8 x 8``).
        reference: Frozen reference providing ``scatter`` (:math:`F_0`) and
            ``scale`` (:math:`s_0`).

    Returns:
        numpy.ndarray: :math:`F = s_0 D + F_0`.
    """
    return np.asarray(operator, float)*reference.scale+reference.scatter


def information_profile(operator: np.ndarray, reference: TransientReference) -> dict[str, float]:
    r"""Magnitude/shape decomposition of one context operator.

    ``kurtosis_ratio`` is tr F/tr F0 (heavier tails in every direction),
    whereas QJSD and trace distance compare trace-normalised shapes and are
    therefore insensitive to a uniform change of transient energy.

    Args:
        operator: Context operator :math:`D` of one window/state context.
        reference: Frozen :class:`~iabm_incidents.electrical.TransientReference`
            of the same context.

    Returns:
        dict[str, float]: ``kurtosis_ratio`` (:math:`\operatorname{tr}F /
        \operatorname{tr}F_0`), ``qjsd`` (bits), ``trace_distance`` and
        ``purity_ratio`` (:math:`\operatorname{tr}\rho^2 /
        \operatorname{tr}\sigma^2`).

    Raises:
        ValueError: If the recovered scatter or the reference scatter has a
            non-positive or non-finite trace.

    See Also:
        :func:`iabm_incidents.injection.context_statistics`
    """
    f = scatter_from_operator(operator, reference)
    rho, sigma = density_operator(f), density_operator(reference.scatter)
    return dict(kurtosis_ratio=float(np.trace(f)/np.trace(reference.scatter)),
                qjsd=quantum_jensen_shannon(rho, sigma), trace_distance=trace_distance(rho, sigma),
                purity_ratio=float(np.trace(rho @ rho)/np.trace(sigma @ sigma)))


def swap_test_shots(overlap: float, precision: float, confidence: float = .95) -> int:
    r"""Hoeffding shots for estimating tr(ρσ) to ±precision with a SWAP test.

    The ancilla returns 0 with probability (1+tr ρσ)/2, so the overlap is
    2p-1 and its estimate has range two; ``overlap`` is reported only to keep
    the call explicit about the quantity being estimated.

    With :math:`\varepsilon` = ``precision`` and :math:`\delta = 1 -`
    ``confidence``, Hoeffding's inequality on the ancilla probability (target
    accuracy :math:`\varepsilon/2`) gives

    .. math::

        N = \left\lceil \frac{2 \ln(2/\delta)}{\varepsilon^2} \right\rceil.

    Args:
        overlap: Expected overlap :math:`\operatorname{tr}(\rho\sigma)` in
            ``[0, 1]``; validated but not used in the bound.
        precision: Half-width :math:`\varepsilon > 0` of the confidence
            interval on the overlap.
        confidence: Two-sided confidence level in ``(0, 1)``.

    Returns:
        int: Number of SWAP-test shots.

    Raises:
        ValueError: If any argument is outside its domain.

    Examples:
        >>> swap_test_shots(0.5, 0.1)
        738
    """
    if not 0 <= overlap <= 1 or precision <= 0 or not 0 < confidence < 1:
        raise ValueError("Invalid SWAP-test request")
    return int(np.ceil(2*np.log(2/(1-confidence))/precision**2))


def pauli_decomposition(matrix: np.ndarray) -> dict[str, float]:
    r"""Real coefficients c_P = tr(PH)/8 of a real symmetric 3-qubit Hamiltonian.

    Qubit 0 is the most significant bit, matching ``VQEDirectionSolver``.
    Strings with an odd number of Y factors vanish for real symmetric input.

    Args:
        matrix: Real symmetric ``8 x 8`` matrix (tolerance ``1e-10``).

    Returns:
        dict[str, float]: Label (e.g. ``"XZI"``) to coefficient, keeping only
        :math:`|c_P| > 10^{-14}`, so that
        :math:`H = \sum_P c_P P`.

    Raises:
        ValueError: If the matrix is not ``8 x 8`` or not symmetric.

    Notes:
        :math:`\operatorname{tr}H^2 = 8\sum_P c_P^2`, which makes the
        Frobenius norm used by :func:`kato_temple_bounds` measurable.
    """
    h = np.asarray(matrix, float)
    if h.shape != (8, 8) or not np.allclose(h, h.T, atol=1e-10):
        raise ValueError("Real symmetric 8 x 8 Hamiltonian required")
    terms = {}
    for label in map("".join, itertools.product("IXYZ", repeat=3)):
        if label.count("Y") % 2:
            continue
        p = np.kron(np.kron(PAULI[label[0]], PAULI[label[1]]), PAULI[label[2]])
        value = float(np.real(np.trace(p @ h))/8)
        if abs(value) > 1e-14:
            terms[label] = value
    return terms


#: The 36 real three-qubit Pauli strings (even number of ``Y`` factors), in a
#: fixed lexicographic order over ``"IXYZ"``; qubit 0 is the most significant bit.
REAL_PAULI_LABELS = tuple(label for label in map("".join, itertools.product("IXYZ", repeat=3))
                          if label.count("Y") % 2 == 0)


def pauli_vector(matrix: np.ndarray) -> np.ndarray:
    r"""Pauli coefficients of a real symmetric 8 x 8 matrix in a fixed label order.

    Unlike :func:`pauli_decomposition`, which drops negligible terms and
    returns a dictionary, this function always returns all 36 coefficients
    :math:`c_P = \operatorname{tr}(PM)/8` in the order of
    :data:`REAL_PAULI_LABELS`, so that the result can be used directly as a
    feature vector.

    Args:
        matrix: Real symmetric ``8 x 8`` matrix (tolerance ``1e-10``).

    Returns:
        numpy.ndarray: Shape ``(36,)``; :math:`M = \sum_P c_P P`.

    Raises:
        ValueError: If the matrix is not ``8 x 8`` or not symmetric.

    Notes:
        The map is linear and invertible on real symmetric matrices, so the
        vector carries exactly the information of the matrix entries; its value
        is that each coefficient is a measurable expectation on three qubits and
        has a structured reading in terms of the embedding bits.
    """
    m = np.asarray(matrix, float)
    if m.shape != (8, 8) or not np.allclose(m, m.T, atol=1e-10):
        raise ValueError("Real symmetric 8 x 8 matrix required")
    return np.array([np.real(np.einsum("ij,ji->", np.kron(np.kron(PAULI[l[0]], PAULI[l[1]]), PAULI[l[2]]), m))/8
                     for l in REAL_PAULI_LABELS])


def shape_change(scatter: np.ndarray, reference: TransientReference) -> np.ndarray:
    r"""Change of transient shape :math:`\Delta\rho = \rho_w - \rho_0`.

    Args:
        scatter: Window fourth-order scatter :math:`F_w` (PSD, ``8 x 8``).
        reference: Frozen transient reference holding :math:`F_0`.

    Returns:
        numpy.ndarray: Symmetric, trace-zero ``8 x 8`` matrix. Its identity
        Pauli coefficient is therefore zero, and the remaining 35 coefficients
        describe how the direction of transient concentration moved.
    """
    return density_operator(scatter)-density_operator(reference.scatter)


def measurement_settings(labels) -> dict[str, tuple[str, ...]]:
    """Greedy qubit-wise-commuting grouping; settings are bases in {X,Y,Z}³.

    Labels are processed by decreasing weight (number of non-identity
    factors; ties keep input order). A label joins the first existing group
    whose basis agrees with it on every qubit where both are non-identity;
    the group basis then absorbs the label's non-identity factors. Otherwise
    the label opens a new group. ``"III"`` is skipped because it needs no
    measurement.

    Args:
        labels: Iterable of three-character Pauli labels over ``IXYZ``.

    Returns:
        dict[str, tuple[str, ...]]: Measurement basis (remaining ``I``
        positions replaced by ``Z``) to the labels it estimates. For the 36
        even-Y labels of a real symmetric Hamiltonian this yields 14 settings.

    Notes:
        Greedy grouping is not guaranteed to be minimal.
    """
    pending = sorted((label for label in labels if label != "III"), key=lambda s: -sum(c != "I" for c in s))
    bases: list[list[str]] = []; members: list[list[str]] = []
    for label in pending:
        for basis, group in zip(bases, members):
            if all("I" in (b, c) or b == c for b, c in zip(basis, label)):
                basis[:] = [c if b == "I" else b for b, c in zip(basis, label)]
                group.append(label)
                break
        else:
            bases.append(list(label)); members.append([label])
    return {"".join(b if b != "I" else "Z" for b in basis): tuple(group) for basis, group in zip(bases, members)}


@dataclass(frozen=True)
class NoiseModel:
    """Global depolarising gate noise and symmetric readout flips.

    ``fidelity`` is the probability that the ansatz output is unaffected by
    depolarisation; ``mitigate`` rescales with the known fidelity and inverts
    readout attenuation, as in calibrated zero-noise-free post-processing.

    Attributes:
        single_qubit_error: Depolarising error :math:`p_1` per RY gate.
        two_qubit_error: Depolarising error :math:`p_2` per CX gate.
        readout_error: Symmetric bit-flip probability :math:`e` per qubit.
        mitigate: Whether :class:`ShotEstimator` divides each Pauli
            expectation by :math:`f(1-2e)^k`.

    Raises:
        ValueError: If any error rate lies outside ``[0, 0.5)``.

    Notes:
        The global model replaces the ideal outcome distribution
        :math:`q` by :math:`f q + (1-f)/8`, i.e. any gate error fully
        depolarises the three-qubit state. This is pessimistic for the
        state but convenient because mitigation is then a known rescaling.
    """

    single_qubit_error: float = 0.
    two_qubit_error: float = 0.
    readout_error: float = 0.
    mitigate: bool = True

    def __post_init__(self) -> None:
        """Validate the error rates.

        Raises:
            ValueError: If any rate lies outside ``[0, 0.5)``.
        """
        if not all(0 <= value < .5 for value in (self.single_qubit_error, self.two_qubit_error, self.readout_error)):
            raise ValueError("Error rates must lie in [0, 0.5)")

    def fidelity(self, layers: int) -> float:
        r"""Probability that no gate error occurs in the ansatz.

        Args:
            layers: Number of ansatz layers :math:`L` (three RY and two CX
                gates each).

        Returns:
            float: :math:`f = (1-p_1)^{3L}(1-p_2)^{2L}`.
        """
        return (1-self.single_qubit_error)**(3*layers)*(1-self.two_qubit_error)**(2*layers)


class ShotEstimator:
    r"""Estimate ⟨ψ|H|ψ⟩ for batches of real states from simulated shots.

    Shots are split equally across qubit-wise-commuting settings. With zero
    noise and infinite shots the estimator equals the exact expectation.

    Args:
        shots_per_setting: Shots per measurement basis, or ``None`` for the
            infinite-shot (noisy-mean) limit.
        noise: Gate and readout noise model.
        layers: Ansatz layers used for the fidelity; should match the solver.
        seed: Seed of the multinomial sampling generator.

    Attributes:
        shots: ``shots_per_setting``.
        noise: Noise model.
        layers: Ansatz depth.
        rng: :class:`numpy.random.Generator` used for shot sampling.
        settings: Basis to label grouping from :func:`measurement_settings`
            (14 settings).
        labels: The 36 even-Y Pauli labels in fixed product order.
        sign: Label to length-8 array of :math:`\pm 1` eigenvalues per
            computational-basis outcome.
        unitary: Basis to ``8 x 8`` rotation applied before readout.
        confusion: ``8 x 8`` readout confusion matrix
            (Kronecker product of three symmetric flip matrices).

    Raises:
        ValueError: If ``shots_per_setting`` is given and smaller than 1.

    Notes:
        The total shot budget per energy evaluation is
        ``14 * shots_per_setting``.
    """

    def __init__(self, shots_per_setting: int | None, noise: NoiseModel = NoiseModel(), layers: int = 3, seed: int = 0) -> None:
        """Precompute labels, groupings, rotations, signs and readout matrix.

        Args:
            shots_per_setting: Shots per basis or ``None``.
            noise: Noise model.
            layers: Ansatz depth for the fidelity.
            seed: Sampling seed.

        Raises:
            ValueError: If ``shots_per_setting`` is smaller than 1.
        """
        if shots_per_setting is not None and shots_per_setting < 1:
            raise ValueError("Positive shot budget required")
        self.shots, self.noise, self.layers = shots_per_setting, noise, layers
        self.rng = np.random.default_rng(seed)
        labels = [label for label in map("".join, itertools.product("IXYZ", repeat=3)) if label.count("Y") % 2 == 0]
        self.settings = measurement_settings(labels)
        self.labels = labels
        bits = (np.arange(8)[:, None] >> np.array([2, 1, 0])) & 1
        # sign[label][outcome] is the ±1 eigenvalue of that Pauli string.
        self.sign = {label: np.prod(np.where(np.array([c != "I" for c in label]), 1-2*bits, 1), axis=1) for label in labels}
        self.unitary = {basis: np.kron(np.kron(ROTATION[basis[0]], ROTATION[basis[1]]), ROTATION[basis[2]]) for basis in self.settings}
        flip = np.array([[1-self.noise.readout_error, self.noise.readout_error], [self.noise.readout_error, 1-self.noise.readout_error]])
        self.confusion = np.kron(np.kron(flip, flip), flip)

    def coefficients(self, hamiltonians: np.ndarray) -> np.ndarray:
        """Return B x L Pauli coefficients for the fixed label order.

        Args:
            hamiltonians: Batch of real ``8 x 8`` matrices, shape ``(B, 8, 8)``.

        Returns:
            numpy.ndarray: Shape ``(B, 36)``; column ``k`` holds
            :math:`\\operatorname{tr}(P_k H_b)/8` for ``labels[k]``.
        """
        h = np.asarray(hamiltonians, float)
        out = np.zeros((len(h), len(self.labels)))
        for k, label in enumerate(self.labels):
            # Even Y count: the Kronecker product is real.
            p = np.real(np.kron(np.kron(PAULI[label[0]], PAULI[label[1]]), PAULI[label[2]]))
            out[:, k] = np.einsum("ij,bji->b", p, h)/8
        return out

    def expectation(self, states: np.ndarray, hamiltonians: np.ndarray) -> np.ndarray:
        r"""Estimate B energies; ``shots_per_setting=None`` gives the noisy mean.

        For each basis setting the ideal outcome distribution is computed from
        the rotated state, depolarised (:math:`f q + (1-f)/8`), passed
        through the readout confusion matrix, renormalised and, with a finite
        budget, replaced by multinomial frequencies. Each member Pauli
        expectation is the signed outcome average, optionally mitigated by
        :math:`1/(f(1-2e)^k)`, and the energy is
        :math:`c_{III} + \sum_P c_P \langle P\rangle`.

        Args:
            states: Real (or complex) state vectors, shape ``(B, 8)``.
            hamiltonians: Matching Hamiltonians, shape ``(B, 8, 8)``.

        Returns:
            numpy.ndarray: Shape ``(B,)`` estimated energies.

        Notes:
            Coefficients are cached on the identity of ``hamiltonians``;
            :meth:`~iabm_incidents.electrical.VQEDirectionSolver.solve_batch`
            reuses one array per call, so the cache hits for every Rotosolve
            evaluation of that batch.
        """
        psi = np.asarray(states, complex)
        # Note: identity-based cache; in-place mutation would go unnoticed.
        if getattr(self, "_cached", (None,))[0] is not hamiltonians:
            self._cached = (hamiltonians, self.coefficients(hamiltonians))
        coefficients = self._cached[1]
        f = self.noise.fidelity(self.layers)
        attenuation = {label: (1-2*self.noise.readout_error)**sum(c != "I" for c in label) for label in self.labels}
        energy = coefficients[:, self.labels.index("III")].copy()
        for basis, members in self.settings.items():
            probabilities = np.abs(psi @ self.unitary[basis].T)**2
            # Global depolarising channel: mix with the uniform distribution.
            probabilities = f*probabilities+(1-f)/8
            probabilities = probabilities @ self.confusion.T
            probabilities = np.clip(probabilities, 0, None); probabilities /= probabilities.sum(1, keepdims=True)
            if self.shots is not None:
                probabilities = self.rng.multinomial(self.shots, probabilities)/self.shots
            for label in members:
                value = probabilities @ self.sign[label]
                if self.noise.mitigate:
                    value = value/attenuation[label]/f
                energy += coefficients[:, self.labels.index(label)]*value
        return energy


class NoisyVQEDirectionSolver(VQEDirectionSolver):
    """Rotosolve VQE whose every energy evaluation goes through ``ShotEstimator``.

    Same ansatz, budget and restart policy as the noiseless solver; accept/
    reject decisions are taken on noisy energies, exactly as on hardware.

    Only ``_energy`` is overridden, so the parent's restart policy, Rotosolve
    sweep and final state reconstruction are reused unchanged. The returned
    directions are exact statevectors of the selected angles; only the
    *selection* is noisy.

    Args:
        estimator: Shot/noise estimator; its ``layers`` should equal
            ``layers``.
        layers: Ansatz layers.
        sweeps: Rotosolve sweeps per restart (default 6, versus 8 in the
            parent).
        restarts: Independent starts.
        seed: Seed of the restart angles.

    Attributes:
        estimator: The energy estimator.

    Raises:
        ValueError: If any resource budget is smaller than 1 (raised by the
            parent constructor).
    """

    def __init__(self, estimator: ShotEstimator, layers: int = 3, sweeps: int = 6, restarts: int = 2, seed: int = 0) -> None:
        """Configure the parent solver and attach the estimator.

        Args:
            estimator: Shot/noise estimator.
            layers: Ansatz layers.
            sweeps: Rotosolve sweeps.
            restarts: Restarts.
            seed: Restart seed.

        Raises:
            ValueError: If ``min(layers, sweeps, restarts) < 1``.
        """
        super().__init__(layers=layers, sweeps=sweeps, restarts=restarts, seed=seed)
        self.estimator = estimator

    def _energy(self, psi: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Replace the exact expectation by the shot/noise estimate.

        Args:
            psi: Batch of states, shape ``(B, 8)``.
            h: Batch of normalised Hamiltonians, shape ``(B, 8, 8)``.

        Returns:
            numpy.ndarray: Estimated energies, shape ``(B,)``.
        """
        return self.estimator.expectation(psi, h)


def kato_temple_bounds(operator: np.ndarray, direction: np.ndarray, second: float | None = None) -> dict[str, float]:
    r"""A-posteriori interval for λ_max from a variational direction.

    :math:`\rho = v^\top D v` is a rigorous lower bound. Kato–Temple gives
    :math:`\lambda_{\max} \le \rho + \sigma^2/(\rho - \alpha)` with
    :math:`\sigma^2 = \lVert Dv\rVert^2 - \rho^2` whenever
    :math:`\alpha \ge \lambda_2` and :math:`\rho > \alpha`. Without
    ``second`` the rigorous
    :math:`\alpha_F = (\lVert D\rVert_F^2 - \rho^2)^{1/2} \ge |\lambda_2|` is
    used (:math:`\sum_i \lambda_i^2 = \lVert D\rVert_F^2` and
    :math:`\lambda_1 \ge \rho \ge 0`), and
    :math:`\lVert D\rVert_F \ge \lambda_{\max}` caps the bound. A supplied
    variational estimate :math:`\hat\lambda_2` is only a lower estimate of
    :math:`\lambda_2`, so that variant is reported as gap-assisted, not
    rigorous. :math:`\rho`, :math:`\langle D^2\rangle` and
    :math:`\operatorname{tr} D^2 = 8\sum_P c_P^2` are all measurable.

    Args:
        operator: Real symmetric operator :math:`D`.
        direction: Unit-norm variational direction :math:`v` (not
            renormalised here).
        second: Optional estimate of :math:`\lambda_2` (e.g. from
            :func:`deflated_second_direction`). When given, it is used as
            :math:`\alpha`, the :math:`\lVert D\rVert_F` cap is *not* applied
            and the result is flagged non-rigorous.

    Returns:
        dict[str, float]: ``rayleigh`` (:math:`\rho`), ``residual_variance``
        (:math:`\sigma^2`, clipped at 0), ``gap_reference`` (:math:`\alpha`),
        ``rigorous`` (``second is None``), ``kato_temple_applicable``
        (:math:`\rho > \alpha`), ``upper`` (Kato–Temple bound, ``inf`` when
        not applicable, capped by :math:`\lVert D\rVert_F` in the rigorous
        variant) and ``weinstein_radius`` (:math:`\sigma`).

    Notes:
        The Weinstein radius certifies that *some* eigenvalue lies in
        :math:`[\rho-\sigma, \rho+\sigma]`. With the Frobenius reference,
        :math:`\alpha_F \ge |\lambda_2|` relies on :math:`\rho \ge 0`; for
        :math:`\rho < 0` Kato–Temple is simply not applicable and only the
        cap is reported.

    Examples:
        >>> import numpy as np
        >>> d = np.diag([3., 1., 0., 0., 0., 0., 0., -1.])
        >>> kato_temple_bounds(d, np.eye(8)[0])["upper"]
        3.0
    """
    d = np.asarray(operator, float); v = np.asarray(direction, float)
    rho = float(v @ d @ v); sigma2 = max(float(v @ d @ d @ v)-rho**2, 0.)
    frobenius = float(np.linalg.norm(d))
    # Why: alpha_F >= |lambda_2| because sum(lambda_i^2) = ||D||_F^2 and
    # lambda_1 >= rho >= 0; for rho < 0 the test rho > alpha below fails.
    alpha = np.sqrt(max(frobenius**2-rho**2, 0.)) if second is None else float(second)
    upper = rho+sigma2/(rho-alpha) if rho > alpha else np.inf
    if second is None:
        upper = min(upper, frobenius)
    return dict(rayleigh=rho, residual_variance=sigma2, gap_reference=float(alpha), rigorous=second is None,
                kato_temple_applicable=bool(rho > alpha), upper=float(upper), weinstein_radius=float(np.sqrt(sigma2)))


def deflated_second_direction(operators: np.ndarray, directions: np.ndarray, solver: VQEDirectionSolver) -> np.ndarray:
    r"""VQD-style second state: maximise D-β vvᵀ with β above the spectral width.

    With :math:`\beta = 2\lVert D\rVert_F + 1` the deflated operator
    :math:`D - \beta vv^\top` penalises overlap with the first direction, and
    the solver's maximiser :math:`w` gives the Rayleigh estimate
    :math:`w^\top D w` of :math:`\lambda_2`.

    Args:
        operators: Batch of operators, shape ``(B, 8, 8)``.
        directions: First variational directions, shape ``(B, 8)``.
        solver: Variational solver (exact-energy or noisy).

    Returns:
        numpy.ndarray: Shape ``(B,)`` second-eigenvalue estimates. These are
        variational estimates of :math:`\lambda_2`; they are guaranteed lower
        estimates only when :math:`v` is the exact top eigenvector, so bounds
        built on them are not rigorous.

    Raises:
        ValueError: If the solver rejects the operator batch.
    """
    d = np.asarray(operators, float); v = np.asarray(directions, float)
    beta = 2*np.linalg.norm(d, axis=(1, 2))+1.
    deflated = d-beta[:, None, None]*np.einsum("bi,bj->bij", v, v)
    w = solver.solve_batch(deflated)
    return np.einsum("bi,bij,bj->b", w, d, w)
