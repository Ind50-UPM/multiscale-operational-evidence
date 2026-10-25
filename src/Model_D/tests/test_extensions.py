"""Tests for quantum-information evidence, conformal fusion and fault injection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iabm_incidents.conformal import ConformalEvidenceFusion, cauchy_combination, smoothed_conformal_pvalues
from iabm_incidents.electrical import TransientReference, VQEDirectionSolver, fourth_scatter
from iabm_incidents.injection import Injection, SecondOrderReference, context_statistics, inject
from iabm_incidents.quantum import (NoiseModel, NoisyVQEDirectionSolver, ShotEstimator, density_operator,
    deflated_second_direction, information_profile, kato_temple_bounds, measurement_settings,
    pauli_decomposition, quantum_jensen_shannon, trace_distance)


def _operator(seed=0):
    a = np.random.default_rng(seed).standard_normal((8, 8))
    return (a+a.T)/2


def _reference(seed=0):
    z = np.random.default_rng(seed).standard_normal((4000, 8))
    f0 = fourth_scatter(z)
    return TransientReference(np.zeros(8), np.eye(8), f0, np.trace(f0)/8)


def test_gaussian_fobi_matrix_is_scaled_identity():
    z = np.random.default_rng(1).standard_normal((400000, 8))
    assert np.allclose(fourth_scatter(z), (8+2)/8*np.eye(8), atol=.02)


def test_qjsd_and_trace_distance_bounds():
    rho = density_operator(np.diag([1., 0, 0, 0, 0, 0, 0, 0])+1e-12*np.eye(8))
    sigma = density_operator(np.diag([0., 1, 0, 0, 0, 0, 0, 0])+1e-12*np.eye(8))
    assert quantum_jensen_shannon(rho, rho) == pytest.approx(0, abs=1e-9)
    assert quantum_jensen_shannon(rho, sigma) == pytest.approx(1, abs=1e-6)
    assert trace_distance(rho, sigma) == pytest.approx(1, abs=1e-6)


def test_information_profile_separates_magnitude_from_shape():
    reference = _reference()
    profile = information_profile((2*reference.scatter-reference.scatter)/reference.scale, reference)
    assert profile["kurtosis_ratio"] == pytest.approx(2)
    assert profile["qjsd"] == pytest.approx(0, abs=1e-9)


def test_pauli_decomposition_reconstructs_hamiltonian():
    h = _operator()
    terms = pauli_decomposition(h)
    from iabm_incidents.quantum import PAULI
    rebuilt = sum(c*np.real(np.kron(np.kron(PAULI[k[0]], PAULI[k[1]]), PAULI[k[2]])) for k, c in terms.items())
    assert np.allclose(rebuilt, h)
    assert all(k.count("Y") % 2 == 0 for k in terms)


def test_measurement_settings_cover_each_string_once():
    labels = [k for k in pauli_decomposition(_operator()) if k != "III"]
    settings = measurement_settings(labels)
    covered = [m for members in settings.values() for m in members]
    assert sorted(covered) == sorted(labels)
    for basis, members in settings.items():
        assert all(c in ("I", b) for m in members for b, c in zip(basis, m))


def test_shot_estimator_is_exact_without_noise_and_unbiased_with_mitigation():
    h = np.stack([_operator(i) for i in range(16)])
    psi = np.random.default_rng(3).standard_normal((16, 8)); psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    exact = np.einsum("bi,bij,bj->b", psi, h, psi)
    assert np.allclose(ShotEstimator(None).expectation(psi, h), exact)
    noisy = ShotEstimator(None, NoiseModel(1e-3, 1e-2, 2e-2, True)).expectation(psi, h)
    assert np.allclose(noisy, exact)
    raw = ShotEstimator(None, NoiseModel(1e-3, 1e-2, 2e-2, False)).expectation(psi, h)
    assert not np.allclose(raw, exact)


def test_noiseless_hook_preserves_vqe_and_noisy_solver_returns_unit_states():
    d = np.stack([_operator(i) for i in range(4)])
    exact = VQEDirectionSolver(layers=2, sweeps=2, restarts=1, seed=5).solve_batch(d)
    hooked = NoisyVQEDirectionSolver(ShotEstimator(None), layers=2, sweeps=2, restarts=1, seed=5).solve_batch(d)
    assert np.allclose(exact, hooked)
    noisy = NoisyVQEDirectionSolver(ShotEstimator(256, seed=1), layers=2, sweeps=2, restarts=1, seed=5).solve_batch(d)
    assert np.allclose(np.linalg.norm(noisy, axis=1), 1)


def test_kato_temple_interval_contains_exact_eigenvalue():
    d = np.stack([_operator(i) for i in range(32)])
    solver = VQEDirectionSolver(layers=3, sweeps=6, restarts=2, seed=1729)
    v = solver.solve_batch(d)
    second = deflated_second_direction(d, v, solver)
    for matrix, direction, s in zip(d, v, second):
        exact = np.linalg.eigvalsh(matrix)
        rigorous = kato_temple_bounds(matrix, direction)
        assert rigorous["rayleigh"] <= exact[-1]+1e-9 <= rigorous["upper"]+2e-9
        assert rigorous["gap_reference"] >= abs(exact[-2])-1e-9
        assisted = kato_temple_bounds(matrix, direction, s)
        if s >= exact[-2]-1e-9:
            assert exact[-1] <= assisted["upper"]+1e-9


def test_exact_eigenvector_gives_zero_width_certificate():
    d = _operator(); values, vectors = np.linalg.eigh(d)
    bounds = kato_temple_bounds(d, vectors[:, -1], values[-2])
    assert bounds["upper"] == pytest.approx(values[-1]) and bounds["residual_variance"] == pytest.approx(0, abs=1e-12)


def test_conformal_pvalues_are_uniform_and_acat_is_valid():
    rng = np.random.default_rng(0)
    # Validity is marginal over calibration draws, not conditional on one set.
    p = np.array([smoothed_conformal_pvalues(rng.standard_normal(99), rng.standard_normal(1), rng)[0] for _ in range(20000)])
    assert abs(np.mean(p <= .05)-.05) < .006
    independent = rng.uniform(size=(20000, 5))
    assert abs(np.mean(cauchy_combination(independent) <= .05)-.05) < .008
    assert np.isnan(cauchy_combination(np.full((1, 3), np.nan))[0])


def test_conformal_fusion_attributes_dominant_coordinate():
    cal = pd.DataFrame(np.random.default_rng(1).standard_normal((500, 3)), columns=list("abc"))
    test = pd.DataFrame({"a": [0., 0.], "b": [9., 0.], "c": [0., np.nan]})
    out = ConformalEvidenceFusion(list("abc")).fit(cal).transform(test)
    assert out.dominant_evidence.iloc[0] == "b" and out.p_fused.iloc[0] < .01
    assert np.isfinite(out.p_fused.iloc[1])


def test_injections_follow_observer_algebra():
    rng = np.random.default_rng(0)
    r = rng.standard_normal((100, 3)); active = np.arange(100) >= 50; offsets = np.full((100, 3), 4.)
    out = inject(r, np.zeros(100), active, offsets, Injection("under_response", .1, phases=(0,)), rng)
    assert np.allclose(out[:50], r[:50]) and np.allclose(out[50:, 0], .9*r[50:, 0]-.4) and np.allclose(out[50:, 1:], r[50:, 1:])
    out = inject(r, np.zeros(100), active, offsets, Injection("dropouts", 1., rate_per_hour=3600., length=1), rng, sample_period_seconds=1.)
    changed = ~np.isclose(out, r)
    assert changed[:50].sum() == 0 and np.allclose(out[changed], -4.)


def test_fourth_order_statistic_reacts_to_sparse_bursts_invisible_to_the_mean():
    rng = np.random.default_rng(2)
    reference = _reference(3)
    z = rng.standard_normal((4000, 8))
    second = SecondOrderReference(z.T @ z/len(z), np.trace(z.T @ z/len(z))/8)
    clean = context_statistics(z, reference, second)
    spiky = z.copy(); spiky[rng.choice(4000, 8, replace=False), 0] += 25.
    burst = context_statistics(spiky, reference, second)
    assert burst["fourth_order"] > 2*clean["fourth_order"]
    assert abs(burst["mean_shift"]-clean["mean_shift"]) < .2


def test_injection_reports_applied_events_and_uses_sample_period():
    rng = np.random.default_rng(4)
    r = np.zeros((4461, 3)); active = np.ones(4461, bool); offsets = np.ones((4461, 3))
    dropout = Injection("dropouts", 1., rate_per_hour=2., length=1)
    counts = [inject(r, np.zeros(4461), active, offsets, dropout, rng, sample_period_seconds=.807, return_events=True)[1]
              for _ in range(4000)]
    # 4461 samples at 0.807 s span one hour, so the Poisson mean is 2.
    assert abs(np.mean(counts)-2) < .1
    out, events = inject(r, np.zeros(4461), active, offsets, Injection("under_response", .1), rng, return_events=True)
    assert events == 0 and np.allclose(out[:, 0], -.1) and np.allclose(out[:, 1:], 0)
