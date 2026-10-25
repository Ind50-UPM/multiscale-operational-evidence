"""Event-conditioned frequency-response interpretation of transient evidence.

These are responses of an event-frozen residual FIR projection, NOT pump/plant
transfer functions and NOT the frequency response of the nonlinear reliability
score. Explanation is performed after detection and never alters decisions.

Role in the pipeline
--------------------
For each detected episode the interpreter replays the strongest supported
electrical window, recomputes the exact deviation operator
:math:`D = (F - F_0)/s_0` of its strongest controller context (see
:mod:`iabm_incidents.electrical`), and freezes the dominant eigenvector
:math:`v`. Because the whitened coordinates are linear in the causal
embedding, the projection :math:`z^\\top v` is a linear finite impulse
response (FIR) readout of the standardized residuals:

.. math::

    z_t^\\top v = x_t^\\top w - m^\\top w, \\qquad w = W v ,

where :math:`m` is the reference centre. :func:`residual_delay_taps` maps the
eight weights :math:`w` to a ``3 x 4`` tap matrix :math:`b_{ck}` (channel
:math:`c`, causal lag :math:`k = 0..3`), and
:func:`residual_frequency_response` evaluates, for an input direction
:math:`d \\in \\mathbb{R}^3`,

.. math::

    H_d(f) = \\sum_{c,k} d_c\\, b_{ck}\\, e^{-i 2\\pi f k \\Delta t},
    \\qquad 0 \\le f \\le \\tfrac{1}{2\\Delta t} .

.. mermaid::

    flowchart LR
        EP["episodes"] --> SEL["select strongest supported window<br/>(source ids, else overlap)"]
        WS["window_scores"] --> SEL
        RES["residuals (same as scoring)"] --> REP["replay context operators D"]
        SEL --> REP
        REP --> CHK{"replayed Tconc == scored Tconc?"}
        CHK -- no --> ERR["ValueError"]
        CHK -- yes --> CAD{"regular cadence?"}
        CAD -- no --> IRR["status irregular_cadence"]
        CAD -- yes --> FIR["sign gauge, taps, projection trace,<br/>frequency response, bands"]
        FIR --> OUT["EventFrequencyInterpretationResult"]

Main public objects
-------------------
* :class:`FrequencyInterpretationConfig` -- grid, cadence and band policy.
* :func:`residual_delay_taps`, :func:`residual_frequency_response`.
* :class:`EventFrequencyInterpretationResult` -- tables, export and HTML.
* :class:`EventFrequencyResponseInterpreter` -- builds the result.

Data contracts
--------------
* ``events``: one row per episode with ``episode_id``, ``asset_id``,
  ``status`` (``"explained"``, ``"irregular_cadence"`` or
  ``"no_supported_transient_window"``), ``window_id`` once a window is
  selected, and the explanation fields listed in
  :meth:`EventFrequencyResponseInterpreter.explain` when explained.
* ``taps``: one row per channel and lag with ``coefficient`` and
  ``coefficient_energy_share``.
* ``frequency_response``: one row per input direction (the three channels,
  ``common`` and ``differential12``) and frequency.
* ``frequency_bands``: gain summaries per direction and band.
* ``sample_projections``: one row per admissible embedding sample.
* Windows are half-open ``[window_start, window_end)``; timestamps are naive
  UTC.

Maintenance notes
-----------------
* The interpreter never fits anything; it requires the *same* residuals and
  fitted :class:`~iabm_incidents.electrical.TransientConcentration` used for
  scoring and raises if the replayed score differs.
* The cadence check is applied after the replay on purpose, so that the
  replay consistency check always runs on exactly the scored operator.
* The eigenvector sign is fixed once (largest-magnitude unwhitened weight
  positive) and shared by taps and traces.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from html import escape

import numpy as np
import pandas as pd

from .electrical import TransientConcentration, _ordered, _safe_history, causal_delay
from .features import _slice_episode_windows


@dataclass(frozen=True)
class FrequencyInterpretationConfig:
    """Post hoc plotting grid, cadence tolerance and normalized gain bands.

    Attributes:
        frequency_points: Linear grid size including DC and Nyquist.
        cadence_relative_tolerance: Maximum relative cadence discrepancy for
            each four-sample projection. No resampling is performed.
        zero_gain_tolerance: Gains at or below this value have undefined phase.
        bands: Named fractions of Nyquist, used only for descriptive summaries.
            Each entry is ``(name, low, high)`` with ``0 <= low < high <= 1``;
            a band covers ``[low, high)`` of Nyquist, and the Nyquist point
            itself is included when ``high == 1``.

    Raises:
        ValueError: If the grid has fewer than two points, the cadence
            tolerance is outside ``[0, 1)``, the zero-gain tolerance is not
            positive and finite, band names are not unique and nonempty, or
            band bounds are not valid fractions of Nyquist.
    """

    frequency_points: int = 257
    cadence_relative_tolerance: float = 0.05
    zero_gain_tolerance: float = 1e-12
    bands: tuple[tuple[str, float, float], ...] = (
        ("low", 0., .25), ("middle", .25, .75), ("high", .75, 1.),
    )

    def __post_init__(self) -> None:
        """Validate grid, tolerances and band definitions.

        Raises:
            ValueError: On any invalid field (see class docstring).
        """
        if self.frequency_points < 2 or not 0 <= self.cadence_relative_tolerance < 1:
            raise ValueError("Invalid frequency grid or cadence tolerance")
        if not np.isfinite(self.zero_gain_tolerance) or self.zero_gain_tolerance <= 0:
            raise ValueError("Positive finite zero-gain tolerance required")
        if len({name for name, _, _ in self.bands}) != len(self.bands):
            raise ValueError("Unique frequency band names required")
        if any(not name or not 0 <= low < high <= 1 for name, low, high in self.bands):
            raise ValueError("Band bounds must be fractions of Nyquist")


def residual_delay_taps(weights: np.ndarray) -> np.ndarray:
    """Map eight unwhitened coordinate weights to a three-channel/four-tap FIR.

    Columns are causal lags 0, 1, 2, 3. Common-mode coordinates contribute
    equally to all phases at lags two and three.

    Args:
        weights: Unwhitened coordinate weights :math:`w = W v` of shape
            ``(8,)``, ordered as the embedding of
            :func:`~iabm_incidents.electrical.causal_delay`.

    Returns:
        Tap matrix of shape ``(3, 4)``: ``taps[:, 0] = w[0:3]``,
        ``taps[:, 1] = w[3:6]``, ``taps[:, 2] = w[6]/sqrt(3)`` and
        ``taps[:, 3] = w[7]/sqrt(3)``.

    Raises:
        ValueError: If ``weights`` is not a finite array of shape ``(8,)``.

    Examples:
        >>> import numpy as np
        >>> residual_delay_taps(np.arange(8.)).shape
        (3, 4)
    """
    w = np.asarray(weights, float)
    if w.shape != (8,) or not np.isfinite(w).all():
        raise ValueError("Eight finite real coordinate weights required")
    taps = np.zeros((3, 4))
    taps[:, 0], taps[:, 1] = w[:3], w[3:6]
    taps[:, 2], taps[:, 3] = w[6]/np.sqrt(3), w[7]/np.sqrt(3)
    return taps


def residual_frequency_response(taps: np.ndarray, frequencies_hz: np.ndarray,
                                sample_period_seconds: float, direction: np.ndarray) -> np.ndarray:
    """H_d(f) = sum_c,k d_c b_ck exp(-i 2 pi f k dt).

    Input units are standardized residuals; the observer and whitening are
    frozen. This linear readout precedes fourth-order weighting, centering,
    reference subtraction, eigenvalue selection and positive fourth-root clipping.

    Args:
        taps: Tap matrix of shape ``(3, 4)``.
        frequencies_hz: One-dimensional frequency grid in hertz within
            ``[0, 0.5/sample_period_seconds]``.
        sample_period_seconds: Sampling period :math:`\\Delta t` in seconds.
        direction: Input direction :math:`d` over the three channels, shape
            ``(3,)``.

    Returns:
        Complex response of shape ``(len(frequencies_hz),)``.

    Raises:
        ValueError: On wrong shapes, a non-positive or non-finite sample
            period, non-finite inputs, or frequencies outside DC..Nyquist.

    Examples:
        >>> import numpy as np
        >>> taps = np.zeros((3, 4)); taps[0, 0] = 1.0
        >>> residual_frequency_response(taps, np.array([0.0]), 1.0, np.eye(3)[0])
        array([1.+0.j])
    """
    b, f, d = np.asarray(taps, float), np.asarray(frequencies_hz, float), np.asarray(direction, float)
    if b.shape != (3, 4) or d.shape != (3,) or f.ndim != 1:
        raise ValueError("Expected 3 x 4 taps, three-channel direction and frequency vector")
    if not np.isfinite(sample_period_seconds) or sample_period_seconds <= 0:
        raise ValueError("Positive finite sample period required")
    if not all(np.isfinite(a).all() for a in (b, f, d)) or (f < 0).any() or (f > .5/sample_period_seconds).any():
        raise ValueError("Finite inputs and frequencies within DC/Nyquist required")
    return (d @ b) @ np.exp(-2j*np.pi*np.outer(np.arange(4)*sample_period_seconds, f))


def _phase(response: np.ndarray, tolerance: float) -> np.ndarray:
    """Unwrap phase independently across each nonzero-gain frequency segment.

    Args:
        response: Complex response on an ordered frequency grid.
        tolerance: Gains at or below this value are treated as zero.

    Returns:
        Phase in degrees with ``NaN`` where the gain is at or below
        ``tolerance``; unwrapping restarts after each such gap.
    """
    valid = np.abs(response) > tolerance
    phase = np.full(len(response), np.nan)
    starts = np.flatnonzero(valid & ~np.r_[False, valid[:-1]])
    for start in starts:
        end = start+1
        while end < len(response) and valid[end]:
            end += 1
        phase[start:end] = np.rad2deg(np.unwrap(np.angle(response[start:end])))
    return phase


@dataclass(frozen=True)
class EventFrequencyInterpretationResult:
    """Separate explanatory artifacts, never additional reliability dimensions.

    Attributes:
        events: Selection provenance, support/abstention, dominant context,
            eigenvalue gap, projected peak and contiguous half-height width.
        taps: Signed FIR coefficients, channel and delay participation. Squared
            coefficient shares describe filter structure, not causal attribution.
        frequency_response: Real/imaginary response, magnitude and phase for
            individual phases, common mode and RP1–RP2 differential mode.
        frequency_bands: Gain summaries; these are not observed spectral energy.
        sample_projections: Admissible timestamps and projected fourth-order
            contributions. Sample indices refer to the selected window.
    """

    events: pd.DataFrame
    taps: pd.DataFrame
    frequency_response: pd.DataFrame
    frequency_bands: pd.DataFrame
    sample_projections: pd.DataFrame

    def export(self, output_dir: str | Path) -> dict[str, Path]:
        """Write CSV tables and scientific interpretation metadata as JSON.

        Also writes ``event_frequency_response.html`` from :meth:`to_html`.
        The directory is created if needed; existing files are overwritten.

        Args:
            output_dir: Destination directory.

        Returns:
            Mapping from artifact name (the five table names, ``"metadata"``
            and ``"html_report"``) to the written path.
        """
        output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        for name in ("events", "taps", "frequency_response", "frequency_bands", "sample_projections"):
            path = output / f"{name}.csv"
            getattr(self, name).to_csv(path, index=False)
            artifacts[name] = path
        path = output / "interpretation_metadata.json"
        path.write_text(json.dumps({
            "title": "Event-conditioned frequency-response interpretation",
            "meaning": "Event-frozen standardized-residual FIR readout; not pump or plant transfer functions",
            "reliability_score_modified": False,
            "direction_solver": "exact largest eigenvalue; deterministic sign gauge",
            "localization": "retrospective within a completed selected window, not warning lead time",
            "band_summaries": "filter gain, not observed frequency energy or nonlinear score sensitivity",
        }, indent=2), encoding="utf-8")
        artifacts["metadata"] = path
        report = output / "event_frequency_response.html"
        report.write_text(self.to_html(), encoding="utf-8")
        artifacts["html_report"] = report
        return artifacts

    def to_html(self) -> str:
        """Build a standalone SVG/HTML report without plotting dependencies.

        Frequency plots use a logarithmic axis, excluding DC (retained in CSV).
        Only the display clips gain to -120 dB; numerical tables retain true
        zero gain as -inf dB and undefined phase as NaN. Masked samples break
        projection traces rather than drawing across unsupported intervals.

        Returns:
            Complete HTML document as a string. Only events with status
            ``"explained"`` get per-episode plots and tables.
        """
        sections = ["<!doctype html><html lang='en'><meta charset='utf-8'>",
            "<title>Event-conditioned frequency-response interpretation</title>",
            "<style>body{font-family:sans-serif;max-width:1100px;margin:24px auto;padding:12px}"
            "table{border-collapse:collapse;font-size:12px}td,th{padding:5px;border:1px solid #ddd}"
            "svg{max-width:100%;height:auto}figure{margin:16px 0}</style>",
            "<h1>Event-conditioned frequency-response interpretation</h1>",
            "<p>Post hoc responses of event-frozen standardized-residual FIR projections. "
            "These are <strong>not pump or plant transfer functions</strong>, observed spectral "
            "energy, or frequency responses of the nonlinear reliability score. "
            "Detection scores and decisions are unchanged.</p>",
            "<p>Projection peaks are retrospective locations inside completed windows, "
            "not warning lead times. Coefficient shares describe the readout structure, "
            "not causal responsibility. Exact dominant directions can be unstable "
            "when the reported eigenvalue gap is small.</p>", self.events.to_html(index=False)]
        for event in self.events.itertuples(index=False):
            if event.status != "explained":
                continue
            sections.append(f"<h2>Episode {escape(str(event.episode_id))}</h2>")
            curves = self.frequency_response[self.frequency_response.episode_id == event.episode_id]
            curves = curves[curves.frequency_hz > 0]
            sections.append(_svg_plot(curves, "frequency_hz", "gain_db", "input_direction",
                "Frozen residual projection magnitude", "Frequency (Hz, log scale)", "Gain (dB)", log_x=True, floor=-120))
            sections.append(_svg_plot(curves, "frequency_hz", "phase_degrees", "input_direction",
                "Frozen residual projection phase", "Frequency (Hz, log scale)", "Phase (degrees)", log_x=True))
            trace = self.sample_projections[self.sample_projections.episode_id == event.episode_id].copy()
            trace["elapsed_seconds"] = (pd.to_datetime(trace.timestamp)-pd.Timestamp(event.window_start)).dt.total_seconds()
            trace["readout"] = "projection"
            sections.append(_svg_plot(trace, "elapsed_seconds", "projection", "readout",
                "Admissible projected residual and retrospective localization", "Seconds since window start", "Projection", break_on="sample_index"))
            sections.append("<h3>Channels and causal delay structure</h3>"+
                self.taps[self.taps.episode_id == event.episode_id].to_html(index=False))
            sections.append("<h3>Frequency-band filter gain (not observed energy)</h3>"+
                self.frequency_bands[self.frequency_bands.episode_id == event.episode_id].to_html(index=False))
        return "".join(sections)+"</html>"



def _svg_plot(frame: pd.DataFrame, x_column: str, y_column: str, group_column: str,
              title: str, x_label: str, y_label: str, *, log_x: bool = False,
              floor: float | None = None, break_on: str | None = None) -> str:
    """Render finite segments with explicit axes; never bridge masked samples.

    Args:
        frame: Long-format table with one row per point.
        x_column: Abscissa column.
        y_column: Ordinate column.
        group_column: Column defining one curve per value (legend label).
        title: Figure caption and accessible label.
        x_label: Abscissa label.
        y_label: Ordinate label.
        log_x: Plot ``log10(x)`` (tick labels show the original values);
            non-positive ``x`` becomes non-finite and is skipped.
        floor: Optional lower clip applied to ``y`` for display only.
        break_on: Optional integer column; a curve is split wherever
            consecutive values differ by anything other than 1.

    Returns:
        HTML ``<figure>`` with an inline SVG, or a short paragraph when no
        finite point exists.
    """
    width, height, left, top, plot_w, plot_h = 900, 340, 90, 35, 760, 235
    curves = []
    for name, group in frame.groupby(group_column, sort=False):
        x, y = group[x_column].to_numpy(float), group[y_column].to_numpy(float)
        if log_x:
            x = np.log10(x)
        if floor is not None:
            y = np.maximum(y, floor)
        valid = np.isfinite(x) & np.isfinite(y)
        curves.append((str(name), x, y, valid, group[break_on].to_numpy() if break_on else None))
    if not curves or not any(valid.any() for _, _, _, valid, _ in curves):
        return f"<p>{escape(title)}: no finite supported values.</p>"
    xx = np.concatenate([x[valid] for _, x, _, valid, _ in curves])
    yy = np.concatenate([y[valid] for _, _, y, valid, _ in curves])
    xmin, xmax, ymin, ymax = xx.min(), xx.max(), yy.min(), yy.max()
    if xmax == xmin:
        xmax = xmin+1
    if ymax == ymin:
        ymin, ymax = ymin-1, ymax+1
    def point(x: float, y: float) -> str:
        """Map data coordinates to the ``"x,y"`` SVG pixel string."""
        return f"{left+(x-xmin)/(xmax-xmin)*plot_w:.2f},{top+plot_h-(y-ymin)/(ymax-ymin)*plot_h:.2f}"
    svg = [f"<figure><figcaption>{escape(title)}</figcaption><svg viewBox='0 0 {width} {height}' role='img' aria-label='{escape(title, quote=True)}'>",
        f"<path d='M {left} {top} V {top+plot_h} H {left+plot_w}' fill='none' stroke='#555' />"]
    for fraction in np.linspace(0, 1, 5):
        xv, yv = xmin+fraction*(xmax-xmin), ymin+fraction*(ymax-ymin)
        svg.append(f"<text x='{left+fraction*plot_w:.1f}' y='{top+plot_h+20}' text-anchor='middle' font-size='11'>{(10**xv if log_x else xv):.3g}</text>")
        svg.append(f"<text x='{left-8}' y='{top+plot_h-fraction*plot_h:.1f}' text-anchor='end' font-size='11'>{yv:.3g}</text>")
    colors = ("#2563eb", "#dc2626", "#059669", "#9333ea", "#d97706")
    for number, (name, x, y, valid, indices) in enumerate(curves):
        color = colors[number % len(colors)]
        segments, current = [], []
        for i in range(len(x)):
            if not valid[i] or (indices is not None and i > 0 and indices[i]-indices[i-1] != 1):
                if current:
                    segments.append(current); current = []
            if valid[i]:
                current.append(point(x[i], y[i]))
        if current:
            segments.append(current)
        for segment in segments:
            if len(segment) == 1:
                a, b = segment[0].split(",")
                svg.append(f"<circle cx='{a}' cy='{b}' r='2' fill='{color}' />")
            else:
                svg.append(f"<polyline points='{' '.join(segment)}' fill='none' stroke='{color}' stroke-width='1.4' />")
        svg.append(f"<text x='{left+number*145}' y='325' font-size='12' fill='{color}'>{escape(name)}</text>")
    svg.extend([f"<text x='{left+plot_w/2}' y='305' text-anchor='middle' font-size='12'>{escape(x_label)}</text>",
        f"<text x='15' y='18' font-size='12'>{escape(y_label)}</text>", "</svg></figure>"])
    return "".join(svg)


class EventFrequencyResponseInterpreter:
    """Explain detected episodes using their strongest supported transient window.

    Selection uses only source windows of each detected episode (falling back
    to temporal overlap where source IDs are unavailable). Each explanation
    freezes the largest-eigenvalue direction of the strongest controller context
    in that window, with the original fitted whitening and residual units.
    Unsupported, nonpositive and irregular-cadence events remain explicit.
    No nominal model is fitted, and no threshold or episode is changed.

    Attributes:
        model: Fitted transient-concentration model used for scoring.
        config: Interpretation policy.

    See Also:
        :class:`~iabm_incidents.electrical.TransientConcentration`.
    """

    def __init__(self, transient_model: TransientConcentration,
                 config: FrequencyInterpretationConfig = FrequencyInterpretationConfig()) -> None:
        """Bind the interpreter to a fitted transient model.

        Args:
            transient_model: The *fitted* model whose references and
                configuration produced the window scores.
            config: Interpretation policy.
        """
        self.model, self.config = transient_model, config

    def explain(self, *, episodes: pd.DataFrame, window_scores: pd.DataFrame,
                residuals: pd.DataFrame) -> EventFrequencyInterpretationResult:
        """Return post hoc tables for detected episodes, including abstentions.

        Args:
            episodes: Already detected Model D episodes with episode/asset IDs.
            window_scores: Original scored windows, including Tconc and support.
            residuals: The same signed residuals and safety flags used in scoring.
                Changing telemetry or the fitted reference invalidates replay;
                inconsistent exact scores raise ``ValueError``.

        Returns:
            :class:`EventFrequencyInterpretationResult`. Empty tables keep a
            minimal set of named columns.

        Raises:
            ValueError: If the replayed exact score of the selected window does
                not match its ``transient_concentration`` (relative tolerance
                ``1e-7``, absolute ``1e-9``), or propagated from timestamp
                normalization.

        Notes:
            Window selection keeps windows of the episode's asset that belong
            to the episode, have a finite positive ``transient_concentration``
            and (if the column exists) ``electrical_status == "supported"``;
            the highest score wins and the earliest ``start_time`` breaks ties.

            For an explained event the row additionally contains
            ``controller_state``, ``transient_concentration``,
            ``direction_solver`` (``"exact"``), ``selection_policy``,
            ``window_start``, ``window_end``, ``decision_available_at`` (the
            window end), ``sample_period_seconds``, ``admissible_vectors``,
            ``dominant_eigenvalue``, ``eigenvalue_gap``
            (:math:`\\lambda_1 - \\lambda_2`), ``projection_offset``
            (:math:`-m^\\top w`), ``peak_time``, ``peak_sample_index``,
            ``peak_signed_projection``, ``half_height_width_seconds``,
            ``reference_projected_fourth_moment``
            (:math:`v^\\top F_0 v / s_0`) and ``projected_fourth_moment``
            (:math:`v^\\top F v / s_0`, the mean of the per-sample
            contributions :math:`\\lVert z\\rVert^2/8\\,(z^\\top v)^2/s_0`).
            Their difference equals ``dominant_eigenvalue``.
        """
        cfg = self.model.config
        frame = _ordered(residuals, cfg)
        events, taps_rows, frequency_rows, bands_rows, trace_rows = [], [], [], [], []
        for _, episode in episodes.iterrows():
            base = {"episode_id": episode.episode_id, "asset_id": episode.asset_id}
            event = {**base, "status": "no_supported_transient_window"}
            events.append(event)
            candidates = window_scores[window_scores.asset_id.astype(str) == str(episode.asset_id)]
            source_ids = episode.get("source_window_ids", ())
            if isinstance(source_ids, str):
                source_ids = (source_ids,)
            candidates = _slice_episode_windows(candidates, source_ids, episode)
            if "transient_concentration" not in candidates:
                continue
            score = pd.to_numeric(candidates.transient_concentration, errors="coerce")
            candidates = candidates.loc[np.isfinite(score) & score.gt(0)]
            if "electrical_status" in candidates:
                candidates = candidates[candidates.electrical_status.eq("supported")]
            if candidates.empty:
                continue
            window = candidates.sort_values(["transient_concentration", "start_time"], ascending=[False, True]).iloc[0]
            base["window_id"] = window.window_id
            event.update(base)
            start = pd.to_datetime(window.start_time, utc=True).tz_convert(None)
            end = pd.to_datetime(window.end_time, utc=True).tz_convert(None)
            part = frame[(frame.asset_id == str(episode.asset_id)) & (frame[cfg.timestamp_column] >= start) & (frame[cfg.timestamp_column] < end)]
            r = part[["residual_1", "residual_2", "residual_3"]].to_numpy(float)
            states = part[cfg.state_column].to_numpy()
            safe = part.safe.to_numpy(bool) & _safe_history(part, cfg) & np.isfinite(r).all(1)
            best = None
            for state in pd.unique(states):
                reference = self.model.references.get((str(episode.asset_id), state))
                x = causal_delay(r, safe & (states == state))
                if reference is None or len(x) < cfg.minimum_samples:
                    continue
                values, vectors = np.linalg.eigh(self.model.operator(x, reference))
                if best is None or values[-1] > best[0]:
                    best = (values[-1], state, reference, values, vectors[:, -1], x)
            if best is None:
                continue
            eigenvalue, state, reference, values, v, x = best
            if not np.isclose(max(eigenvalue, 0)**.25, window.transient_concentration, rtol=1e-7, atol=1e-9):
                raise ValueError("Transient replay differs from scored evidence; use the original residuals and frozen reference")
            # Note: ``ids`` are row positions (within the window slice) of the
            # current sample of each admissible embedding vector, i.e. the same
            # rows that ``causal_delay`` kept, in the same order as ``x``.
            context_safe = safe & (states == state)
            valid = context_safe[3:] & context_safe[2:-1] & context_safe[1:-2] & context_safe[:-3]
            ids = np.arange(3, len(r))[valid]
            # Enforce nominal sampling only after reproducing the original operator.
            dt = part[cfg.timestamp_column].diff().dt.total_seconds().to_numpy()
            regular = np.abs(dt/cfg.sample_period_seconds-1) <= self.config.cadence_relative_tolerance
            # Note: regular[i] judges the spacing between rows i-1 and i, so
            # k = 0, 1, 2 checks the three spacings inside each 4-sample vector.
            if not all(np.all(regular[ids-k]) for k in range(3)):
                event["status"] = "irregular_cadence"
                continue
            # Resolve arbitrary eigenvector sign once, for taps and traces together.
            weights = reference.whitening @ v
            if weights[np.argmax(abs(weights))] < 0:
                v, weights = -v, -weights
            taps = residual_delay_taps(weights)
            z = (x-reference.center) @ reference.whitening
            projection = z @ v
            contributions = np.sum(z*z, axis=1)/8 * projection**2 / reference.scale
            peak = int(np.argmax(abs(projection)))
            # Note: grow the half-height interval only across consecutive
            # admissible samples (ids differing by 1).
            left = right = peak
            while left > 0 and ids[left]-ids[left-1] == 1 and abs(projection[left-1]) >= abs(projection[peak])/2:
                left -= 1
            while right+1 < len(ids) and ids[right+1]-ids[right] == 1 and abs(projection[right+1]) >= abs(projection[peak])/2:
                right += 1
            times = part[cfg.timestamp_column].to_numpy()
            norm = float(np.sum(taps*taps))
            event.update(status="explained", controller_state=state,
                transient_concentration=float(window.transient_concentration),
                direction_solver="exact", selection_policy="maximum positive supported Tconc; earliest window breaks ties",
                window_start=start, window_end=end, decision_available_at=end,
                sample_period_seconds=cfg.sample_period_seconds, admissible_vectors=len(x),
                dominant_eigenvalue=float(eigenvalue), eigenvalue_gap=float(values[-1]-values[-2]),
                projection_offset=float(-reference.center @ weights),
                peak_time=pd.Timestamp(times[ids[peak]]), peak_sample_index=int(ids[peak]),
                peak_signed_projection=float(projection[peak]),
                half_height_width_seconds=float((pd.Timestamp(times[ids[right]])-pd.Timestamp(times[ids[left]])).total_seconds()+cfg.sample_period_seconds),
                reference_projected_fourth_moment=float(v @ reference.scatter @ v/reference.scale),
                projected_fourth_moment=float(contributions.mean()))
            for c, channel in enumerate(cfg.channels):
                for k in range(4):
                    taps_rows.append({**base, "channel": channel, "lag_samples": k,
                        "lag_seconds": k*cfg.sample_period_seconds, "coefficient": taps[c, k],
                        "coefficient_energy_share": taps[c, k]**2/norm if norm else 0.})
            for j, index in enumerate(ids):
                trace_rows.append({**base, "timestamp": times[index], "sample_index": index,
                    "projection": projection[j], "fourth_order_contribution": contributions[j],
                    **{f"residual_{channel}": r[index, c] for c, channel in enumerate(cfg.channels)}})
            frequencies = np.linspace(0, .5/cfg.sample_period_seconds, self.config.frequency_points)
            directions = {name: np.eye(3)[i] for i, name in enumerate(cfg.channels)}
            directions.update(common=np.ones(3)/np.sqrt(3), differential12=np.array([1., -1., 0.])/np.sqrt(2))
            for name, direction in directions.items():
                response = residual_frequency_response(taps, frequencies, cfg.sample_period_seconds, direction)
                gain = abs(response); phase = _phase(response, self.config.zero_gain_tolerance)
                gain_db = np.full(len(gain), -np.inf)
                nonzero = gain > 0; gain_db[nonzero] = 20*np.log10(gain[nonzero])
                for i, frequency in enumerate(frequencies):
                    frequency_rows.append({**base, "input_direction": name, "frequency_hz": frequency,
                        "response_real": response[i].real, "response_imag": response[i].imag,
                        "gain": gain[i], "gain_db": gain_db[i], "phase_degrees": phase[i]})
                for band, low, high in self.config.bands:
                    # Note: half-open band [low, high) of Nyquist; the Nyquist
                    # bin is added explicitly to the band ending at 1.
                    mask = (frequencies >= low*frequencies[-1]) & (frequencies < high*frequencies[-1])
                    if high == 1:
                        mask |= frequencies == frequencies[-1]
                    if mask.any():
                        bands_rows.append({**base, "input_direction": name, "band": band,
                            "lower_hz": low*frequencies[-1], "upper_hz": high*frequencies[-1],
                            "maximum_gain": float(gain[mask].max()),
                            "peak_gain_frequency_hz": float(frequencies[mask][np.argmax(gain[mask])]),
                            "mean_squared_gain": float(np.mean(gain[mask]**2))})
        return EventFrequencyInterpretationResult(
            pd.DataFrame(events, columns=None if events else ["episode_id", "asset_id", "status"]),
            pd.DataFrame(taps_rows, columns=None if taps_rows else ["episode_id", "window_id", "channel", "lag_samples", "coefficient"]),
            pd.DataFrame(frequency_rows, columns=None if frequency_rows else ["episode_id", "window_id", "input_direction", "frequency_hz", "gain", "phase_degrees"]),
            pd.DataFrame(bands_rows, columns=None if bands_rows else ["episode_id", "window_id", "input_direction", "band", "maximum_gain"]),
            pd.DataFrame(trace_rows, columns=None if trace_rows else ["episode_id", "window_id", "timestamp", "projection"]))
