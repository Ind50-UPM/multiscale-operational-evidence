"""Data preparation utilities for Model_A industrial-state classifiers.

This module is the data-ingestion layer of Model_A, the first stage of the
multi-scale pipeline (Model_A state inference -> Model_B behavioural
sequences -> Model_C semantic modes -> Model_D incident evidence). It loads
the raw analog and digital Parquet datasets, applies the legacy
study-compatible NaN imputation, separates active from inactive rows using an
active-power threshold, and transfers digital state labels onto analog
timestamps.

Main public objects:
    * :class:`IndustrialDataProcessor` -- stateful loader/preprocessor.
    * :class:`TrainingDataset` -- active rows with integer labels.
    * :class:`InferenceDataset` -- features plus activity mask and the
      imputed source window, for timestamp-aligned outputs.
    * :class:`EvaluationDataset` -- like :class:`InferenceDataset`, plus
      optional real labels over the full window.

Data contracts:
    *Analog Parquet* (``analog_path``): indexed by timestamp (a
    :class:`pandas.DatetimeIndex`, optionally timezone-aware; it is sorted on
    load). Required columns:

    - features (default ``DEFAULT_FEATURE_COLUMNS``): ``Vrms1..3`` (RMS
      phase voltage), ``Irms1..3`` (RMS phase current), ``PF1..3`` (power
      factor);
    - activity: ``RP1..3`` (per-phase active power, same units as
      ``threshold``, nominally W).

    Optional auxiliary single-channel columns ``Vrms4``, ``RP4``, ``Irms4``,
    ``PF4`` are interpolated when present but are not features by default.
    Numeric float columns are expected.

    *Digital Parquet* (``digital_path``): timestamp index with a column
    ``estado`` holding integer operating-state codes, where ``0`` means
    "no/idle state" and non-zero values are the classes to learn. The index
    must be unique and comparable with the analog index (same timezone
    awareness) because labels are transferred with
    :meth:`pandas.Index.get_indexer` (``method="pad"``).

    *Time windows*: ``start``/``end`` are strings (``"YYYY-MM-DD HH:MM:SS"``)
    compared directly against the index; both bounds are inclusive.

.. mermaid::

    classDiagram
        class IndustrialDataProcessor {
            +DEFAULT_FEATURE_COLUMNS
            +POWER_COLUMNS
            +THREE_PHASE_BLOCKS
            +SINGLE_PHASE_BLOCKS
            +analog_df
            +digital_df
            +threshold
            +feature_columns
            +prepare_training_data(start, end) TrainingDataset
            +prepare_inference_data(start, end, drop_inactive) InferenceDataset
            +prepare_evaluation_data(start, end) EvaluationDataset
            -_get_analog_window(start, end)
            -_slice_by_period(df, start, end)
            -_build_activity_mask(df)
            -_attach_labels(analog_df)
            -_impute_nans(df)
            -_impute_three_phase_blocks(df)
            -_impute_single_phase_blocks(df)
        }
        class TrainingDataset {
            +features
            +labels
        }
        class InferenceDataset {
            +features
            +active_mask
            +source_frame
        }
        class EvaluationDataset {
            +features
            +labels
            +active_mask
            +source_frame
        }
        IndustrialDataProcessor ..> TrainingDataset : creates
        IndustrialDataProcessor ..> InferenceDataset : creates
        IndustrialDataProcessor ..> EvaluationDataset : creates

.. mermaid::

    flowchart LR
        S[_slice_by_period] --> I[_impute_nans] --> W[imputed window]
        W --> M[_build_activity_mask]
        W --> L[_attach_labels]
        L --> F{estado != 0}
        F --> TD[TrainingDataset]
        M --> ID[InferenceDataset]
        ID --> ED[EvaluationDataset]
        L --> ED

Maintenance notes:
    The imputation and labelling rules reproduce the original study
    ("legacy") so that results remain comparable; change them only together
    with a documented re-run of the experiments. Imputation is applied to the
    requested window only (never to the full dataset), and temporal fills
    (``bfill``/``ffill``) therefore never cross the window boundaries. Note
    that ``bfill`` uses later samples *within* the window, which is acceptable
    for offline studies but is not strictly causal for streaming use. The
    column name ``estado`` (Spanish for "state") is part of the data contract
    and of downstream packages; do not rename it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrainingDataset:
    """Bundle supervised features and labels for classifier training.

    The dataclass keeps the public API explicit and avoids passing loosely
    coupled tuples around the codebase when training workflows evolve.

    Attributes:
        features (pandas.DataFrame): Feature matrix of *active, labelled*
            rows only (rows with ``estado != 0``), restricted to
            :attr:`IndustrialDataProcessor.feature_columns` and indexed by the
            original analog timestamps.
        labels (pandas.Series): ``int32`` operating-state codes aligned with
            ``features`` on the same index. Never contains ``0``.
    """

    features: pd.DataFrame
    labels: pd.Series


@dataclass(frozen=True)
class InferenceDataset:
    """Bundle inference-ready features together with activity bookkeeping.

    The source frame and activity mask allow the CLI to reconstruct outputs
    aligned with the original timestamps, including optional inactive rows.

    Attributes:
        features (pandas.DataFrame): Feature matrix to feed the classifier.
            With the default ``drop_inactive=True`` it contains only rows
            where ``active_mask`` is ``True``.
        active_mask (pandas.Series): Boolean Series indexed like
            ``source_frame``; ``True`` where the maximum of ``RP1..3`` is at
            least the processor threshold.
        source_frame (pandas.DataFrame): The full imputed analog window
            (all columns, active and inactive rows) after NaN imputation and
            dropping of rows that remained incomplete.
    """

    features: pd.DataFrame
    active_mask: pd.Series
    source_frame: pd.DataFrame


@dataclass(frozen=True)
class EvaluationDataset:
    """Bundle features, labels, and alignment data for quality assessment.

    Produced by :meth:`IndustrialDataProcessor.prepare_evaluation_data` and
    consumed by the ``evaluate`` mode of :mod:`iabm.main`.

    Attributes:
        features (pandas.DataFrame): Active-row feature matrix (same as
            :attr:`InferenceDataset.features` with ``drop_inactive=True``).
        labels (pandas.Series | None): ``int32`` real labels for *every* row
            of ``source_frame`` (inactive rows and active rows without a
            previous digital event are ``0``), or ``None`` when the processor
            has no digital dataset.
        active_mask (pandas.Series): Boolean activity mask aligned with
            ``source_frame``.
        source_frame (pandas.DataFrame): Full imputed analog window.
    """

    features: pd.DataFrame
    labels: pd.Series | None
    active_mask: pd.Series
    source_frame: pd.DataFrame


class IndustrialDataProcessor:
    """Prepare industrial analog and digital signals for Model_A workflows.

    The processor encapsulates the study-specific preprocessing rules so the
    rest of the package can work with clean training and inference datasets
    through a stable, object-oriented interface.

    Both datasets are loaded eagerly in the constructor and kept in memory;
    every ``prepare_*`` call slices and imputes a fresh copy, so the stored
    frames are never mutated.

    Attributes:
        DEFAULT_FEATURE_COLUMNS (list[str]): Default classifier inputs:
            ``Vrms1..3``, ``Irms1..3``, ``PF1..3``.
        POWER_COLUMNS (list[str]): Active-power columns ``RP1..3`` used for
            the activity mask; they are required in the analog dataset.
        THREE_PHASE_BLOCKS (list[list[str]]): Groups of redundant phase
            columns imputed together by :meth:`_impute_three_phase_blocks`.
        SINGLE_PHASE_BLOCKS (list[list[str]]): Optional auxiliary channels
            (``Vrms4``, ``RP4``, ``Irms4``, ``PF4``) imputed by interpolation.
        analog_df (pandas.DataFrame): Full analog dataset, sorted by index.
        digital_df (pandas.DataFrame | None): Full digital dataset sorted by
            index, or ``None`` when no digital path was given.
        threshold (float): Minimum active power (same units as ``RP*``) for a
            row to be considered active.
        feature_columns (list[str]): Columns exposed as classifier features.

    Examples:
        >>> processor = IndustrialDataProcessor(  # doctest: +SKIP
        ...     "data/analogicas_nonans.parquet", "data/digitales.parquet"
        ... )
        >>> train = processor.prepare_training_data(  # doctest: +SKIP
        ...     "2022-01-18 00:00:00", "2022-02-18 00:00:00"
        ... )
    """

    DEFAULT_FEATURE_COLUMNS = [
        "Vrms1",
        "Vrms2",
        "Vrms3",
        "Irms1",
        "Irms2",
        "Irms3",
        "PF1",
        "PF2",
        "PF3",
    ]
    POWER_COLUMNS = ["RP1", "RP2", "RP3"]
    THREE_PHASE_BLOCKS = [
        ["Vrms1", "Vrms2", "Vrms3"],
        ["RP1", "RP2", "RP3"],
        ["Irms1", "Irms2", "Irms3"],
        ["PF1", "PF2", "PF3"],
    ]
    SINGLE_PHASE_BLOCKS = [["Vrms4"], ["RP4"], ["Irms4"], ["PF4"]]

    def __init__(
        self,
        analog_path: str,
        digital_path: Optional[str] = None,
        *,
        threshold: float = 50.0,
        feature_columns: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize the processor and eagerly load the configured datasets.

        Args:
            analog_path (str): Path to the analog Parquet dataset (see module
                data contract).
            digital_path (str | None): Optional path to the digital Parquet
                dataset used for labels (column ``estado``). Required for
                :meth:`prepare_training_data`; optional for evaluation; unused
                for inference. Any falsy value (``None`` or ``""``) disables it.
            threshold (float): Minimum active power threshold used to discard
                inactive rows. A row is active when
                ``max(RP1, RP2, RP3) >= threshold``. Defaults to ``50.0``.
            feature_columns (Sequence[str] | None): Feature columns to expose
                to the classifier. Defaults to :attr:`DEFAULT_FEATURE_COLUMNS`
                when ``None`` or empty.

        Raises:
            FileNotFoundError: Propagated from :func:`pandas.read_parquet`
                when a path does not exist.
        """
        self.analog_df = pd.read_parquet(analog_path).sort_index()
        self.digital_df = (
            pd.read_parquet(digital_path).sort_index() if digital_path else None
        )
        self.threshold = threshold
        self.feature_columns = list(feature_columns or self.DEFAULT_FEATURE_COLUMNS)

    def prepare_training_data(self, start: str, end: str) -> TrainingDataset:
        """Return supervised features and labels for the requested time range.

        Args:
            start (str): Inclusive lower timestamp bound
                (``"YYYY-MM-DD HH:MM:SS"``).
            end (str): Inclusive upper timestamp bound.

        Returns:
            TrainingDataset: A :class:`TrainingDataset` containing active rows
            only, with labels synchronized from the digital signal stream.
            Rows whose transferred label is ``0`` (inactive rows, or active
            rows preceding the first non-zero digital event) are excluded.

        Raises:
            ValueError: If no digital dataset was configured, or if it has no
                non-zero ``estado`` values while the window contains active
                rows.
            KeyError: If required feature or power columns are missing.
        """
        analog_window = self._get_analog_window(start, end)
        labeled_window = self._attach_labels(analog_window)
        # Note: filtering on ``estado != 0`` implicitly removes inactive rows,
        # because _attach_labels only assigns labels to active rows.
        active_window = labeled_window[labeled_window["estado"] != 0].copy()

        return TrainingDataset(
            features=active_window[self.feature_columns],
            labels=active_window["estado"].astype(np.int32),
        )

    def prepare_inference_data(
        self,
        start: str,
        end: str,
        *,
        drop_inactive: bool = True,
    ) -> InferenceDataset:
        """Return inference-ready analog features without requiring digital labels.

        Args:
            start (str): Inclusive lower timestamp bound.
            end (str): Inclusive upper timestamp bound.
            drop_inactive (bool): Whether to keep only rows above the activity
                threshold. When ``False``, ``features`` covers every row of the
                imputed window while ``active_mask`` is still returned
                unchanged.

        Returns:
            InferenceDataset: An :class:`InferenceDataset` with the feature
            matrix, a boolean mask identifying active rows, and the imputed
            source analog window.

        Raises:
            KeyError: If required feature or power columns are missing.

        Notes:
            The CLI helper ``_build_prediction_frame`` in :mod:`iabm.main`
            assumes one prediction per *active* row, i.e. the default
            ``drop_inactive=True``. With ``drop_inactive=False`` the caller is
            responsible for aligning predictions with ``source_frame``.
        """
        analog_window = self._get_analog_window(start, end)
        active_mask = self._build_activity_mask(analog_window)
        features = analog_window.loc[active_mask, self.feature_columns].copy()

        if not drop_inactive:
            features = analog_window[self.feature_columns].copy()

        return InferenceDataset(
            features=features,
            active_mask=active_mask,
            source_frame=analog_window,
        )

    def prepare_evaluation_data(self, start: str, end: str) -> EvaluationDataset:
        """Return aligned features and optional labels for model evaluation.

        Args:
            start (str): Inclusive lower timestamp bound.
            end (str): Inclusive upper timestamp bound.

        Returns:
            EvaluationDataset: An :class:`EvaluationDataset` containing the
            active feature matrix, optional real labels aligned to the full
            analog window, the activity mask, and the imputed source analog
            frame. ``labels`` is ``None`` when no digital dataset is loaded.

        Raises:
            ValueError: If the digital dataset contains no non-zero states
                while the window has active rows.
            KeyError: If required feature or power columns are missing.
        """
        inference_dataset = self.prepare_inference_data(start, end)
        labels: pd.Series | None = None
        if self.digital_df is not None:
            # Invariant: labels are computed on the same imputed source frame,
            # so they share its index with ``active_mask``.
            labeled_frame = self._attach_labels(inference_dataset.source_frame)
            labels = labeled_frame["estado"].astype(np.int32)

        return EvaluationDataset(
            features=inference_dataset.features,
            labels=labels,
            active_mask=inference_dataset.active_mask,
            source_frame=inference_dataset.source_frame,
        )

    def _get_analog_window(self, start: str, end: str) -> pd.DataFrame:
        """Filter analog signals by period and apply the configured imputations.

        Args:
            start (str): Inclusive lower timestamp bound.
            end (str): Inclusive upper timestamp bound.

        Returns:
            pandas.DataFrame: An imputed analog window ready for activity
            filtering or labeling. Rows still missing any feature or power
            value after imputation are dropped.

        Notes:
            Slicing happens *before* imputation so no information from outside
            the requested window leaks into it through temporal fills.
        """
        analog_window = self._slice_by_period(self.analog_df, start, end)
        return self._impute_nans(analog_window)

    @staticmethod
    def _slice_by_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        """Return a copy of the rows included in the requested time interval.

        Args:
            df (pandas.DataFrame): Time-indexed DataFrame to slice.
            start (str): Inclusive lower timestamp bound.
            end (str): Inclusive upper timestamp bound.

        Returns:
            pandas.DataFrame: A copy of the requested time window
            (``start <= index <= end``).

        Notes:
            A boolean mask is used rather than label slicing, so the index does
            not need to be monotonic and duplicated timestamps are preserved.
        """
        mask = (df.index >= start) & (df.index <= end)
        return df.loc[mask].copy()

    def _build_activity_mask(self, df: pd.DataFrame) -> pd.Series:
        """Mark rows whose maximum active power reaches the configured threshold.

        Args:
            df (pandas.DataFrame): Analog signal window containing the
                configured power columns (``RP1..3``).

        Returns:
            pandas.Series: A boolean Series, indexed like ``df``, where ``True``
            identifies active rows (``max(RP1, RP2, RP3) >= threshold``; the
            comparison is inclusive).

        Raises:
            KeyError: If any of :attr:`POWER_COLUMNS` is missing.
        """
        return df[self.POWER_COLUMNS].max(axis=1) >= self.threshold

    def _attach_labels(self, analog_df: pd.DataFrame) -> pd.DataFrame:
        """Synchronize analog rows with the nearest previous active digital label.

        The implementation mirrors the original Random Forest labeling logic by
        using the nearest earlier non-zero digital state for active analog rows.

        Args:
            analog_df (pandas.DataFrame): Imputed analog window.

        Returns:
            pandas.DataFrame: A copy of ``analog_df`` with an extra ``int32``
            column ``estado``. Inactive rows, and active rows that occur before
            the first non-zero digital event, keep ``estado == 0``.

        Raises:
            ValueError: If no digital dataset is configured, or if the window
                has active rows but the digital dataset contains no non-zero
                ``estado`` values.

        Notes:
            Labels are looked up as-of each analog timestamp (backward-looking
            "pad" match, exact matches included), so a label is never taken
            from the future. The digital index must be unique and monotonic
            for :meth:`pandas.Index.get_indexer` with ``method="pad"``. The
            lookup is unbounded in time: an old digital event can label a much
            later analog row.
        """
        if self.digital_df is None:
            raise ValueError("Digital data is required to prepare supervised labels.")

        labeled_df = analog_df.copy()
        labeled_df["estado"] = 0
        active_mask = self._build_activity_mask(labeled_df)
        if not active_mask.any():
            return labeled_df

        relevant_digitals = self.digital_df[self.digital_df["estado"] != 0]
        if relevant_digitals.empty:
            raise ValueError("Digital dataset does not contain active state labels.")

        # Label transfer uses the nearest previous non-zero digital event so the
        # analog timeline inherits the operational state known at that instant.
        indexer = relevant_digitals.index.get_indexer(
            labeled_df.index[active_mask], method="pad"
        )
        # Note: -1 means no earlier digital event exists; those rows keep 0.
        valid_positions = indexer >= 0
        if valid_positions.any():
            active_index = labeled_df.index[active_mask]
            labeled_df.loc[active_index[valid_positions], "estado"] = (
                relevant_digitals["estado"].iloc[indexer[valid_positions]].to_numpy()
            )

        labeled_df["estado"] = labeled_df["estado"].astype(np.int32)
        return labeled_df

    def _impute_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing analog values with the legacy study-compatible strategy.

        Three-phase variables follow the original heuristic of using the minimum
        available phase value and then propagating neighbouring observations. The
        optional fourth-channel variables are interpolated when present.

        Args:
            df (pandas.DataFrame): Analog window to impute (not modified).

        Returns:
            pandas.DataFrame: An imputed copy of ``df`` from which rows that
            still contain NaN in any feature or power column are dropped.

        Raises:
            KeyError: If any of :attr:`feature_columns` or
                :attr:`POWER_COLUMNS` is absent from ``df``.
        """
        imputed_df = df.copy()
        self._impute_three_phase_blocks(imputed_df)
        self._impute_single_phase_blocks(imputed_df)
        return imputed_df.dropna(subset=self.feature_columns + self.POWER_COLUMNS)

    def _impute_three_phase_blocks(self, df: pd.DataFrame) -> None:
        """Impute three-phase channels via phase minimum, next, and previous values.

        For each block in :attr:`THREE_PHASE_BLOCKS` (columns missing from
        ``df`` are skipped), every column is imputed in two steps:

        1. NaN cells take the row-wise minimum of the block's phases;
        2. remaining NaNs (all phases missing) are back-filled, then
           forward-filled along time.

        Args:
            df (pandas.DataFrame): DataFrame updated in place.

        Notes:
            ``block_frame`` is a snapshot taken *before* any column in the
            block is modified, so the row-wise minimum always uses the original
            (un-imputed) phase values, never values filled earlier in the loop.
            Back-fill is applied before forward-fill, which is not causal
            within the window (see module maintenance notes).
        """
        for block in self.THREE_PHASE_BLOCKS:
            available_block = [column for column in block if column in df.columns]
            if not available_block:
                continue

            # Note: column selection returns a copy; it is not updated by the
            # in-place assignments below.
            block_frame = df[available_block]
            for column in available_block:
                nan_mask = df[column].isna()
                if nan_mask.any():
                    # The legacy heuristic first borrows the minimum valid value
                    # from the remaining phases before applying temporal fills.
                    df.loc[nan_mask, column] = block_frame.loc[nan_mask].min(axis=1)
                df[column] = df[column].bfill().ffill()

    def _impute_single_phase_blocks(self, df: pd.DataFrame) -> None:
        """Impute single-channel variables with linear interpolation when present.

        Args:
            df (pandas.DataFrame): DataFrame updated in place. Columns from
                :attr:`SINGLE_PHASE_BLOCKS` that are absent are skipped.

        Notes:
            Interpolation is positional (``method="linear"``), not time
            weighted, and ``limit_direction="both"`` also fills leading and
            trailing gaps.
        """
        for block in self.SINGLE_PHASE_BLOCKS:
            column = block[0]
            if column not in df.columns:
                continue

            # Single auxiliary channels do not have multi-phase redundancy, so
            # interpolation is the least invasive default imputation rule here.
            df[column] = df[column].interpolate(
                method="linear", limit_direction="both"
            )
