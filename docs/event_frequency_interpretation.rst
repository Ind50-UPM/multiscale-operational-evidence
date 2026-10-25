Event-conditioned frequency-response interpretation
===================================================

Purpose and scope
-----------------

This optional Model D stage explains already detected episodes through a frozen
linear readout of command-conditioned electrical residuals. It answers which
residual channels and causal delays participate in the transient projection,
which frequencies that projection amplifies, and where its largest projected
impulse lies inside the selected window.

The stage is **post hoc**. It adds no component to the nine-dimensional
reliability vector, changes no score or threshold, and creates no additional
incident family. Callers may restrict the standalone interpreter to the abrupt
episodes they want to investigate; the pipeline does not infer a new physical
abrupt-failure label from an explanation.

The responses are **not pump or plant transfer functions**. No input/output
system identification is performed. They are also not frequency responses of
the nonlinear transient score. Their interpretation is limited to the selected,
event-frozen standardized-residual projection.

Frozen spatial and delay readout
--------------------------------

For the fitted controller-state reference, let ``W`` denote the whitening
matrix, ``c`` the eight-coordinate nominal center, and ``v`` the unit dominant
eigenvector of the fourth-order operator. The exact indicator and the projected
readout are

.. math::

   T_w^{\mathrm{conc}} = [\lambda_{\max}(D_w)]_+^{1/4},
   \qquad y[n] = (x[n]-c)^\top Wv.

The symmetric whitening implementation uses the same row-vector convention as
the detector, ``z = (x-c) @ W``. Setting ``a = W @ v`` gives the channel/delay
coefficients

.. math::

   b_{p,0}=a_p,\quad b_{p,1}=a_{p+3},\quad
   b_{p,2}=a_6/\sqrt{3},\quad b_{p,3}=a_7/\sqrt{3},
   \qquad p=0,1,2.

Consequently,

.. math::

   y[n]=\sum_{p=0}^{2}\sum_{k=0}^{3}b_{p,k}r_p[n-k]-c^\top a.

These coefficients operate on standardized command residuals, not raw electrical
power. The reference offset contributes a constant term to the projected trace
and is excluded from the AC frequency response. No extra gain normalization is
introduced. The eigenvector sign is fixed by making the largest absolute
unwhitened coefficient positive, keeping the taps, phase and trace consistent.
A small dominant-eigenvalue gap indicates that the selected direction may be
unstable; a sign convention does not resolve a degenerate eigenspace.

For an input direction ``d`` and the declared measured cadence ``dt``, the
frequency response is

.. math::

   H_d(f)=\sum_{p=0}^{2}\sum_{k=0}^{3}
   d_p b_{p,k}\exp(-\mathrm{i}2\pi f k\Delta t).

The outputs include individual phases, normalized common mode
``(1,1,1)/sqrt(3)``, and normalized differential mode ``(1,-1,0)/sqrt(2)``.
Gain is ``abs(H)`` and gain in dB is ``20*log10(abs(H))``. At true zero gain,
dB is negative infinity and phase is undefined. Phase unwraps separately across
nonzero segments. CSV tables include DC and Nyquist; the HTML Bode plots omit
DC on their logarithmic frequency axis. Display clipping at -120 dB does not
change the exported numerical values.

Channel and delay coefficient-energy shares sum to one when the projection is
nonzero. Summing these shares by channel or lag describes participation in the
filter structure. It does **not** establish causal responsibility for a fault:
channel interference and cross-channel fourth-order relationships remain.
Named frequency bands summarize maximum gain, the frequency of maximum gain,
and sampled mean squared gain. They do not measure observed spectral energy,
fault capture, or sensitivity of the complete nonlinear score. The default
``low``, ``middle`` and ``high`` bands occupy 0--25%, 25--75% and 75--100% of
Nyquist, respectively; they are descriptive conventions, not equipment limits.

Event selection, replay and localization
----------------------------------------

For each supplied detected episode, the interpreter selects its source window
with the largest positive, supported exact ``transient_concentration``. The
earliest start breaks equal-score ties. It then selects the controller context
with the largest eigenvalue in that window and reconstructs its exact direction
using the original frozen reference. The optional detector VQE is not rerun:
this stage explains the scientific exact indicator rather than claiming to
replay a previously recorded variational direction.

Use the same residuals and frozen reference that produced the scored windows.
A replay mismatch raises ``ValueError`` instead of silently explaining changed
data. Unsupported or nonpositive events retain an explicit
``no_supported_transient_window`` status. Missing command/reference support is
never replaced by a guessed projection.

No interpolation or resampling is performed. Each four-sample projection must
match the declared cadence within the configured relative tolerance; otherwise
the event receives ``irregular_cadence`` and no frequency curves. Causal vectors
retain the detector's command, settling, safety and gap masks. Projected traces
and half-height widths never bridge invalid samples or context changes.

The peak is the maximum absolute projected residual on admissible vectors. Its
half-height width is the contiguous span above half that peak plus one sample
period. This is retrospective localization within a completed window. The
reported ``decision_available_at`` is the selected window's end, not an online
warning time or the episode's eventual end.

Each projected sample also exports the nonnegative fourth-order contribution

.. math::

   q[n] = \frac{\|z[n]\|^2}{8s_0}\,y[n]^2.

The mean contribution minus ``v.T @ F0 @ v / s0`` reconstructs the dominant
Rayleigh quotient. Individual contributions are not physical-failure
probabilities, and clipping/selection still belongs to the detector. The trace
exports current signed phase residuals alongside the projected evidence.

Usage and artifacts
-------------------

Enable post hoc interpretation using already fitted electrical models:

.. code-block:: python

   from iabm_incidents import (
       IndicatorPipeline, FrequencyInterpretationConfig,
   )

   pipeline = IndicatorPipeline(
       model_d_config,
       command_observer=frozen_observer,
       transient_model=frozen_transients,
       interpretation_config=FrequencyInterpretationConfig(
           frequency_points=257,
           cadence_relative_tolerance=0.05,
       ),
   )
   result = pipeline.run(
       sequences=sequences,
       assignments=assignments,
       analogue=aligned_electrical_telemetry,
       digital=digital,
   )
   if result.event_interpretation is not None:
       paths = result.event_interpretation.export("reports/event_interpretation")

Omitting ``interpretation_config`` preserves the original execution path and
returns ``event_interpretation=None``. Frozen electrical models are required
when opting in. ``ModelDExperimentRunner`` accepts the same configuration and
its report exporter retains the artifacts under the indicator mode's
``event_interpretation`` directory.

The standalone interpreter also supports a selected subset of episodes:

.. code-block:: python

   from iabm_incidents import EventFrequencyResponseInterpreter

   explanation = EventFrequencyResponseInterpreter(frozen_transients).explain(
       episodes=selected_detected_episodes,
       window_scores=original_window_scores,
       residuals=original_residuals,
   )
   paths = explanation.export("reports/selected_events")

Exports contain ``events.csv``, ``taps.csv``, ``frequency_response.csv``,
``frequency_bands.csv``, ``sample_projections.csv``, scientific interpretation
metadata, and a standalone ``event_frequency_response.html`` report with inline
SVG magnitude/phase curves, projected traces and participation tables. No
plotting dependency or network access is required to view that report. Empty
results retain table headers so exports remain readable by downstream tools.

API reference
-------------

.. autosummary::
   :nosignatures:

   iabm_incidents.interpretation
