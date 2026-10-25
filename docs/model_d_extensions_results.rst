Model D extensions: results
===========================

All tables are read from ``reports/extensions`` (compact copies in
``docs/_static/extensions``). See :doc:`model_d_extensions` for the methods.

Certified variational decisions (held-out windows)
--------------------------------------------------

Certification is evaluated for two decisions. The *composite* M2 warning is
often fixed by the eight non-transient coordinates (only 0–10 windows per
scale are *pivotal*, i.e. change when the transient coordinate is removed),
so certifying it says little about the solver. The *transient-only* decision
(``T4 >= q95`` of the calibration ``T4``) depends on the solver alone.

.. csv-table::
   :file: _static/extensions/certificate_dependence.csv
   :header-rows: 1

At 1 h, 6 h and 24 h every decision is certified by the elementary bracket
``[mu, ||D||_F]``. At 15 min the rigorous protocol certifies only 39.5 % of
transient-only decisions and routes 164 of 271 windows to the exact solver;
the non-rigorous VQD-gap variant would appear to certify 98.5 %. No certified
decision was wrong.

Bound coverage and Kato–Temple tightness:

.. csv-table::
   :file: _static/extensions/certificates.csv
   :header-rows: 1

.. csv-table::
   :file: _static/extensions/kato_temple.csv
   :header-rows: 1

When applicable, Kato–Temple brings the upper bound within about 0.02 % of the
exact eigenvalue (against 5.6 % for the Frobenius cap on the same 24-h
contexts). The VQD-gap variant violates the bound in 0.1–2.4 % of contexts.

Finite shots and noise
----------------------

Primary 24-h held-out windows (composite decisions, score error):

.. csv-table::
   :file: _static/extensions/shots.csv
   :header-rows: 1

15-min transient-only decisions (agreement with the exact solver):

.. csv-table::
   :file: _static/extensions/shots_15min.csv
   :header-rows: 1

At 24 h decisions never change (smallest composite margin 2.0 robust units).
At 15 min the noiseless VQE already changes 2 of 271 decisions, few shots
reduce agreement to about 97 %, and unmitigated noise keeps it near 97 %
regardless of the shot budget.

Fault injection: detection power at the clean 95th-percentile threshold
-----------------------------------------------------------------------

Power is window-level: the fraction of one-hour windows exceeding the clean
threshold after injection. The telemetry period is 0.807 s, so durations are
given in seconds and converted to samples, rates refer to wall-clock hours,
and the realised number of events per window is logged
(``detection_power_given_event`` conditions on windows that received one).


.. csv-table::
   :file: _static/extensions/injection_power.csv
   :header-rows: 1

Fourth-order evidence dominates sparse power dropouts; first-order evidence
dominates persistent under-response; additive bursts are masked by the already
heavy-tailed nominal residual.

Scoring rules and multiscale fusion
-----------------------------------

.. csv-table::
   :file: _static/extensions/fusion.csv
   :header-rows: 1

Seed sensitivity of the smoothed conformal p-values (50 seeds):

.. csv-table::
   :file: _static/extensions/conformal_seeds.csv
   :header-rows: 1

Fault-type identification from the shape operator
-------------------------------------------------

Macro-F1 and accuracy of out-of-fold predictions (grouped 5-fold logistic
regression on held-out injection windows):

.. csv-table::
   :file: _static/extensions/identification.csv
   :header-rows: 1

The Pauli signature of :math:`\Delta\rho` raises the identification of
single- versus three-phase under-response from 0.59 to 0.83 macro-F1 and the
family identification from 0.76 to 0.82. The raw matrix entries give the same
phase result (0.835), confirming that the information lies in the full shape
operator. The physical reading of individual Pauli terms is limited by the
current embedding (three phases at lags 0--1 and the common mode at lags
2--3), which does not factorise over the qubit bits; an embedding of four
channels by two lags would give each term a direct channel/lag meaning.

Model A class-level diagnostics
-------------------------------

.. csv-table::
   :file: _static/extensions/model_a_classes.csv
   :header-rows: 1
