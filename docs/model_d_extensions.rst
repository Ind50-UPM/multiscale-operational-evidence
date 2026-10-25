Model D extensions: certified quantum-ready evidence, conformal fusion and fault injection
==========================================================================================

This page documents the three methodological extensions added to Model D for
the *Results in Engineering* manuscript. The implementations live in
:mod:`iabm_incidents.quantum`, :mod:`iabm_incidents.conformal` and
:mod:`iabm_incidents.injection`; the experiment driver is
:mod:`run_model_d_extensions`. Results are in :doc:`model_d_extensions_results`.

.. mermaid::

   flowchart TB
     R["Frozen observer residuals r_t"] --> E["Causal 8-D embedding x_t<br/>(3 phases lag 0/1, common mode lag 2/3)"]
     E --> W["Context whitening z_t = (x_t − m_c) W_c"]
     W --> F["Fourth-order scatter F_w (FOBI kurtosis matrix)"]
     F --> DOP["Density operator ρ_w = F_w / tr F_w"]
     DOP --> MAG["Magnitude: kurtosis ratio κ_w"]
     DOP --> SHP["Shape: QJSD, trace distance"]
     F --> D["Operator D = (F_w − F_0)/s_0"]
     D --> EX["Exact λ_max"]
     D --> VQE["3-qubit VQE (RY/CX, Rotosolve)"]
     VQE --> CERT["Kato–Temple certificate<br/>Frobenius gap α_F"]
     VQE --> NISQ["Shot / depolarising / readout model"]
     CERT --> DEC{"certified warning / clearance / exact fallback"}

Fourth-order operator as a density operator
-------------------------------------------

For whitened coordinates :math:`\mathbf{z}\in\mathbb{R}^p` (here :math:`p=8`),

.. math::

   F = \frac{1}{p}\,\mathbb{E}\!\left[\|\mathbf{z}\|^2\mathbf{z}\mathbf{z}^\top\right],
   \qquad F_{\text{Gauss}} = \frac{p+2}{p} I,
   \qquad \operatorname{tr}F = \frac{1}{p}\mathbb{E}\|\mathbf{z}\|^4 .

:math:`F` is the FOBI kurtosis matrix (Cardoso, 1989) and its trace is Mardia's
multivariate kurtosis divided by :math:`p`. Because :math:`F\succeq 0`,
:math:`\rho=F/\operatorname{tr}F` is a real density operator on three qubits.
:func:`iabm_incidents.quantum.information_profile` returns the kurtosis ratio
(magnitude) and the quantum Jensen--Shannon divergence and trace distance to
the frozen nominal operator (shape).

Certified variational decisions
-------------------------------

For a variational direction :math:`\mathbf{v}` with Rayleigh quotient
:math:`\rho` and residual variance :math:`\sigma^2=\|D\mathbf{v}\|^2-\rho^2`,
:func:`iabm_incidents.quantum.kato_temple_bounds` returns

.. math::

   \rho \le \lambda_{\max}(D) \le \min\!\left(\rho + \frac{\sigma^2}{\rho-\alpha_F},\ \|D\|_F\right),
   \qquad \alpha_F = \sqrt{\|D\|_F^2-\rho^2} \ge |\lambda_2| .

The Kato--Temple term is used only when :math:`\rho>\alpha_F` (which
implies :math:`\rho>0` and :math:`\rho^2>\|D\|_F^2/2`); otherwise the
Frobenius cap applies. On hardware, :math:`\rho` and :math:`\langle H^2\rangle`
are shot estimates, so the bound must be widened by confidence intervals to
remain rigorous. Passing a variational estimate of :math:`\lambda_2`
produces the *gap-assisted* variant, which is reported for comparison but is
**not** rigorous (a variational estimate is a lower estimate of
:math:`\lambda_2`). Window decisions are certified when the lower score already
exceeds the onset threshold or the upper score stays below it; remaining
windows fall back to the exact solver.

Measurement and noise model
---------------------------

:class:`iabm_incidents.quantum.ShotEstimator` expands
:math:`H=-D/\|D\|_F` in the 36 real Pauli strings, measures the 35
non-identity strings in 14 qubit-wise-commuting bases, samples multinomial
shots and applies global depolarisation (fidelity
:math:`f=(1-p_1)^{3L}(1-p_2)^{2L}`) and symmetric readout flips, with optional
mitigation by :math:`1/(f(1-2e)^k)`.
:class:`iabm_incidents.quantum.NoisyVQEDirectionSolver` runs the same
Rotosolve optimiser with every energy evaluated through the estimator, via
the ``_energy`` hook of :class:`iabm_incidents.electrical.VQEDirectionSolver`.

Conformal Cauchy-combination fusion
-----------------------------------

:class:`iabm_incidents.conformal.ConformalEvidenceFusion` converts each
directional excess into a smoothed conformal p-value against the calibration
partition and fuses them with the Cauchy combination test (Liu and Xie, 2020):

.. math::

   T_w=\frac{1}{K_w}\sum_j \tan\big[(0.5-p_{j,w})\pi\big],\qquad
   p_w=\tfrac12-\frac{\arctan T_w}{\pi}.

The onset/recovery levels :math:`p_w\le0.05` / :math:`0.20` mirror the
95th/80th-percentile policy of the equal-weight score. The coordinate (and,
for multiscale fusion, the scale) with the smallest p-value is recorded as the
dominant evidence of each warning.

Semi-synthetic fault injection
------------------------------

:mod:`iabm_incidents.injection` injects labelled faults into genuine nominal
residuals through the frozen observer, :math:`r'=(1-\delta)r-\delta c_s/s_s`:
persistent under-response, short power dropouts and additive bursts.
:func:`iabm_incidents.injection.window_statistics` evaluates the mean residual
norm, whitened mean shift, second-order score, fourth-order score, kurtosis
ratio and QJSD with identical whitening and context aggregation.

Running the extensions
----------------------

.. code-block:: bash

   PYTHONPATH=src/Model_D python src/run_model_d_extensions.py --stage model_a
   PYTHONPATH=src/Model_D python src/run_model_d_extensions.py --stage quantum
   PYTHONPATH=src/Model_D python src/run_model_d_extensions.py --stage telemetry   # needs data/*.parquet
   PYTHONPATH=src/Model_D python src/run_model_d_extensions.py --stage fusion
   PYTHONPATH=src/Model_D python src/run_model_d_extensions.py --stage robustness  # after quantum and telemetry
   PYTHONPATH=src:src/Model_D python src/export_extension_figure_data.py

The runner hashes every primary and multiscale artifact before and after the
run and fails if any of them changed.
