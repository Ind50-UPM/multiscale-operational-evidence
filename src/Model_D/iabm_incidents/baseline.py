r"""Nominal baseline helpers for Model_D indicator workflows.

Role in the pipeline:
    Provides a compact, robust (median/MAD) summary of the seven behavioural
    window indicators produced upstream (Model_B/Model_C derived evidence or
    :class:`~iabm_incidents.behavioral.BehavioralWindowEncoder`). The summary
    is used to centre indicators against a nominal reference.

Main public objects:
    * :class:`NominalBaseline` -- frozen per-indicator ``median``/``mad``.
    * :func:`fit_nominal_baseline` -- selects nominal rows and fits it.

Data contract:
    The input frame may contain any subset of the indicator columns
    ``sequence_divergence``, ``duration_drift``, ``recurrence_excess``,
    ``persistence_excess``, ``consumption_deviation``, ``state_error_rate``
    and ``mode_divergence`` (numeric or coercible; non-numeric values are
    treated as missing). Nominal selection uses ``semantic_status`` (string,
    ``"ANOMALOUS"`` marks non-nominal rows) when present, otherwise
    ``deviation_score``.

Robust statistics:
    For each indicator with finite values :math:`x_1,\dots,x_n` in the
    selected reference,

    .. math::

       m = \operatorname{median}(x_i), \qquad
       \mathrm{MAD} = \operatorname{median}(|x_i - m|).

    The MAD is **not** multiplied by the normal-consistency constant 1.4826.

Maintenance notes:
    The baseline is fitted on whatever frame the caller passes; callers are
    responsible for passing only the chronological reference partition to
    avoid evaluation-period leakage. Absent or empty indicator columns yield
    ``median = mad = 0`` rather than raising, so downstream centering
    silently becomes a no-op for those indicators.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class NominalBaseline:
    """Compact nominal reference statistics for indicator centering.

    Each attribute is a :class:`pandas.Series` indexed by ``"median"`` and
    ``"mad"`` holding the robust location and (unscaled) median absolute
    deviation of one indicator over the nominal reference rows.

    Attributes:
        sequence_divergence: Statistics of the sequence-divergence indicator.
        duration_drift: Statistics of the duration-drift indicator.
        recurrence_excess: Statistics of the recurrence-excess indicator.
        persistence_excess: Statistics of the persistence-excess indicator.
        consumption_deviation: Statistics of the consumption deviation.
        state_error_rate: Statistics of the state-error rate.
        mode_divergence: Statistics of the mode-divergence indicator.
    """

    sequence_divergence: pd.Series
    duration_drift: pd.Series
    recurrence_excess: pd.Series
    persistence_excess: pd.Series
    consumption_deviation: pd.Series
    state_error_rate: pd.Series
    mode_divergence: pd.Series

    def medians(self) -> dict[str, float]:
        """Return the nominal median of every indicator.

        Returns:
            Mapping from indicator column name to its nominal median as a
            Python ``float``; ``0.0`` when the statistic is unavailable.
        """
        return {
            "sequence_divergence": float(self.sequence_divergence.get("median", 0.0)),
            "duration_drift": float(self.duration_drift.get("median", 0.0)),
            "recurrence_excess": float(self.recurrence_excess.get("median", 0.0)),
            "persistence_excess": float(self.persistence_excess.get("median", 0.0)),
            "consumption_deviation": float(self.consumption_deviation.get("median", 0.0)),
            "state_error_rate": float(self.state_error_rate.get("median", 0.0)),
            "mode_divergence": float(self.mode_divergence.get("median", 0.0)),
        }


def fit_nominal_baseline(frame: pd.DataFrame) -> NominalBaseline:
    """Fit a simple robust baseline from nominal or lowest-deviation rows.

    Reference-row selection follows a fixed precedence:

    1. If ``semantic_status`` exists, keep rows whose status is not
       ``"ANOMALOUS"`` (missing statuses count as nominal).
    2. Otherwise, if ``deviation_score`` exists, keep rows whose numeric
       score is at or below its 25th percentile.
    3. If the selection is empty (or neither column exists), all rows are
       used.

    Args:
        frame: Window-level indicator table (see the module data contract).
            It is copied and never modified.

    Returns:
        A :class:`NominalBaseline` with median/MAD per indicator.

    Notes:
        The function performs no chronological filtering. Pass only the
        reference partition to keep the baseline free of evaluation data.
    """

    reference = frame.copy()
    if "semantic_status" in reference.columns:
        nominal = reference[reference["semantic_status"].ne("ANOMALOUS")]
        if not nominal.empty:
            reference = nominal
    elif "deviation_score" in reference.columns and not reference.empty:
        threshold = pd.to_numeric(reference["deviation_score"], errors="coerce").quantile(0.25)
        subset = reference[pd.to_numeric(reference["deviation_score"], errors="coerce") <= threshold]
        if not subset.empty:
            reference = subset

    return NominalBaseline(
        sequence_divergence=_summarize(reference, "sequence_divergence"),
        duration_drift=_summarize(reference, "duration_drift"),
        recurrence_excess=_summarize(reference, "recurrence_excess"),
        persistence_excess=_summarize(reference, "persistence_excess"),
        consumption_deviation=_summarize(reference, "consumption_deviation"),
        state_error_rate=_summarize(reference, "state_error_rate"),
        mode_divergence=_summarize(reference, "mode_divergence"),
    )


def _summarize(frame: pd.DataFrame, column: str) -> pd.Series:
    """Compute the median and unscaled MAD of one indicator column.

    Args:
        frame: Reference rows.
        column: Indicator column name.

    Returns:
        Series with index ``["median", "mad"]``. Both are ``0.0`` when the
        column is absent or has no numeric values after coercion.
    """
    if column not in frame.columns:
        return pd.Series({"median": 0.0, "mad": 0.0})
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return pd.Series({"median": 0.0, "mad": 0.0})
    median = float(values.median())
    mad = float((values - median).abs().median())
    return pd.Series({"median": median, "mad": mad})
