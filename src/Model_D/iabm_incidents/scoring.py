"""Window-level scoring interfaces for the indicator-driven Model_D route.

Role in the pipeline:
    :class:`~iabm_incidents.pipeline.IndicatorPipeline` computes one row of
    deviation indicators per analysis window and then needs a single scalar
    ``deviation_score`` per window to drive onset/recovery segmentation. This
    module defines the scorer contract and the default implementation.

Main public objects:
    * :class:`DeviationScorer` - structural :class:`typing.Protocol` with a
      ``fit``/``score`` pair, so alternative scorers (robust z-scores,
      conformal fusion, learned models) can be swapped in without inheritance.
    * :class:`WeightedDeviationScorer` - stateless weighted sum delegating to
      :func:`~iabm_incidents.detection.compute_deviation_score`.

Data contract:
    ``score`` receives the window-indicator frame and must return a
    :class:`pandas.Series` aligned with its index. The weighted scorer reads
    the nine evidence columns of
    :data:`~iabm_incidents.detection.INDICATOR_COLUMNS` (seven behavioural
    indicators, ``command_response_deviation`` and
    ``transient_concentration``); absent or non-numeric values contribute
    ``0``. If none of those columns exists, it falls back to a binary
    ``semantic_status == "ANOMALOUS"`` score.

.. mermaid::

   classDiagram
       class DeviationScorer {
           <<Protocol>>
           +fit(nominal_indicators) None
           +score(indicators) Series
       }
       class WeightedDeviationScorer {
           -_weights: DeviationWeights
           +fit(nominal_indicators) None
           +score(indicators) Series
       }
       DeviationScorer <|.. WeightedDeviationScorer

Maintenance notes:
    * ``fit`` is called by the pipeline on windows whose ``semantic_status`` is
      not ``"ANOMALOUS"``. The weighted scorer ignores it, but stateful
      scorers must only learn from that nominal subset (leakage guard).
    * A missing indicator contributes zero: it means *unavailable evidence*,
      not healthy telemetry.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from .config import DeviationWeights
from .detection import compute_deviation_score


class DeviationScorer(Protocol):
    """Protocol for reproducible window-level scoring methods.

    Any object with compatible ``fit`` and ``score`` methods satisfies the
    protocol (structural typing); no subclassing is required.
    """

    def fit(self, nominal_indicators: pd.DataFrame) -> None:
        """Learn scorer state from nominal windows only.

        Args:
            nominal_indicators: Window-indicator rows considered nominal by the
                caller. Implementations must not look at evaluated windows.
        """
        ...
    def score(self, indicators: pd.DataFrame) -> pd.Series:
        """Return one deviation score per window.

        Args:
            indicators: Window-indicator frame to score.

        Returns:
            pandas.Series: Scores aligned with ``indicators.index``; larger
            means more anomalous.
        """
        ...


class WeightedDeviationScorer:
    """Simple deterministic scorer backed by the existing weighted score logic.

    The score of window :math:`w` is

    .. math::

        S_w = \\sum_{j} \\omega_j \\, x_{w,j},

    where :math:`x_{w,j}` are the available evidence columns (missing values
    set to zero) and :math:`\\omega_j` the corresponding fields of
    :class:`~iabm_incidents.config.DeviationWeights`.

    Args:
        weights: Non-learned weights per evidence dimension. Setting a weight
            to zero removes that dimension (ablation).

    See Also:
        :func:`iabm_incidents.detection.compute_deviation_score`
    """

    def __init__(self, weights: DeviationWeights) -> None:
        """Store the weight configuration.

        Args:
            weights: Weight dataclass used by every subsequent ``score`` call.
        """
        self._weights = weights

    def fit(self, nominal_indicators: pd.DataFrame) -> None:
        """No-op: the weighted score has no data-driven parameters.

        Args:
            nominal_indicators: Ignored; accepted for protocol compatibility.

        Returns:
            None
        """
        return None

    def score(self, indicators: pd.DataFrame) -> pd.Series:
        """Compute the weighted deviation score.

        Args:
            indicators: Window-indicator frame.

        Returns:
            pandas.Series: Float scores aligned with ``indicators.index``.

        Raises:
            ValueError: If the frame has neither ``semantic_status`` nor any
                recognised indicator column (raised by
                :func:`~iabm_incidents.detection.compute_deviation_score`).
        """
        return compute_deviation_score(indicators, self._weights)
