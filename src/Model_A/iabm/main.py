"""Command-line entry point for training and using Model_A classifiers.

This module is the orchestration layer of Model_A (first stage of the
multi-scale pipeline: Model_A state inference -> Model_B behavioural
sequences -> Model_C semantic modes -> Model_D incident evidence). It exposes
the ``industrial-id`` console script (``[tool.poetry.scripts]`` in
``pyproject.toml``) and can also be run as ``python -m iabm.main``. All data
handling is delegated to :class:`~iabm.data_processor.IndustrialDataProcessor`
and all model logic to :class:`~iabm.models.StateClassifier`.

Execution modes (``--mode``):
    * ``train`` -- load analog + digital data for ``[start, end]``, run
      cross-validation, fit the final model, write a fold-metrics Excel report
      and a ``joblib`` artifact.
    * ``predict`` -- load an artifact and label a new analog window (digital
      data not needed); write ``.xlsx`` (default) or ``.csv`` predictions.
    * ``evaluate`` -- compare one or more artifacts on a window; write an Excel
      workbook (per-timestamp predictions, accuracy/ROC-AUC summary,
      confusion matrices) and, when real labels exist, confusion-matrix and
      ROC plots (PNG with title plus title-free PDF).

Default paths (resolved from the repository root, three levels above this file's package directory):
    * analog input: ``data/analogicas_nonans.parquet``
    * digital input: ``data/digitales.parquet``
    * model artifact: ``src/models/Model_A/<analog file name>_<ALGO>.joblib``
    * reports/predictions: ``src/predictions/Model_A/``

Output contracts:
    * Prediction table: index ``Time`` (original analog timestamps), column
      ``Predicted_State`` (state code). Only active rows unless
      ``--include-inactive``, in which case inactive rows carry ``0``. File
      name ``predictions_<analog file name>_<MODEL_TYPE>.{xlsx,csv}``.
    * Training metrics: ``<analog file name>_<ALGO>_metrics_<YYYY-MM-DD>.xlsx``
      with columns ``Fold`` and ``Accuracy``.
    * Evaluation workbook: ``evaluation_<analog stem>_<start>_<end>.xlsx``
      with sheet ``predictions`` (index ``Time``, ``Real_Class`` as nullable
      ``Int64`` and one ``<TYPE>_Prediction`` column per model), sheet
      ``summary`` (``Model``, ``Accuracy``, ``ROC_AUC_Class_<k>``) and one
      ``cm_<model>`` sheet per model.

.. mermaid::

    flowchart TD
        M[main] --> DL[_detect_language] --> I18N[setup_i18n] --> PA[parse_arguments]
        PA --> RP[_resolve_paths]
        RP --> MODE{args.mode}
        MODE -- train --> TR[prepare_training_data]
        TR --> CV[StateClassifier.cross_validate]
        CV --> FIT[StateClassifier.fit]
        FIT --> SMR[_save_metrics_report]
        SMR --> SAVE[StateClassifier.save]
        MODE -- predict --> LD[StateClassifier.load]
        LD --> INF[prepare_inference_data]
        INF --> BPF[_build_prediction_frame]
        BPF --> OUT[CSV or _prepare_excel_frame then XLSX]
        MODE -- evaluate --> RE[_run_evaluation]
        RE --> BEO[_build_evaluation_outputs]
        BEO --> XL[evaluation workbook]
        RE --> CM[_save_confusion_matrix_plots]
        RE --> ROC[_save_roc_plots]
        CM --> SPV[_save_plot_variants]
        ROC --> SPV

Maintenance notes:
    Every user-facing string is wrapped in the translator and must exist as
    a ``msgid`` in ``locales/*/LC_MESSAGES/messages.po``; changing a message
    literal silently drops its translation. Plot axis labels, legend entries,
    sheet names and column names are deliberately *not* translated so output
    files keep a stable schema for downstream models. Matplotlib is forced to
    the non-interactive ``Agg`` backend at import time so the CLI works on
    headless servers. Prediction inputs are reindexed to the artifact's
    stored feature order without a fill value; missing columns therefore
    become NaN. ``StandardScaler`` passes NaN through, and whether the
    estimator then rejects or tolerates them depends on the estimator and
    library version (XGBoost treats NaN as missing values), so feature-set
    mismatches are not guaranteed to fail loudly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_processor import EvaluationDataset, IndustrialDataProcessor
from .models import StateClassifier
from .utils import setup_i18n


def parse_arguments(translator: Callable[[str], str]) -> argparse.Namespace:
    """Build the CLI parser with translated help messages and parse ``sys.argv``.

    Options (underscore aliases such as ``--data_ana`` are also accepted
    where noted):

    * ``--mode {train,predict,evaluate}`` (required).
    * ``--algo {rf,xgb}`` (default ``rf``): algorithm for training and for
      the default artifact name.
    * ``--lang {es,en}`` (default ``en``): interface language.
    * ``--data-ana`` / ``--data_ana``: analog Parquet path.
    * ``--data-dig`` / ``--data_dig``: digital Parquet path.
    * ``--model-out`` / ``--model_out``: artifact path (written in ``train``,
      read in ``predict`` and as fallback in ``evaluate``).
    * ``--pred-out`` / ``--pred_out``: output directory for reports,
      predictions and plots.
    * ``--model-paths`` / ``--model_paths``: one or more artifacts to
      compare in ``evaluate``.
    * ``--start``, ``--end`` (required): inclusive window bounds,
      ``"YYYY-MM-DD HH:MM:SS"``.
    * ``--threshold`` (float, default ``50.0``): activity power threshold.
    * ``--cv-splits`` (int, default ``5``): requested CV folds.
    * ``--include-inactive`` (flag): emit inactive rows with state ``0`` in
      ``predict`` output.
    * ``--pred-format`` / ``--pred_format`` ``{xlsx,csv}`` (default
      ``xlsx``): prediction export format.

    Args:
        translator (Callable[[str], str]): Translation function returned by
            :func:`~iabm.utils.setup_i18n`, applied to help texts.

    Returns:
        argparse.Namespace: Parsed command-line arguments ready to drive the
        main workflow. Destinations use underscores (``data_ana``,
        ``model_out``, ``cv_splits``, ``include_inactive``...).

    Raises:
        SystemExit: Raised by :mod:`argparse` on ``--help`` or invalid
            arguments.
    """
    _ = translator
    parser = argparse.ArgumentParser(
        description=_("Industrial state identification framework for IIoT assets")
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "evaluate"],
        required=True,
        help=_("Execution mode: train a new model, generate predictions, or evaluate saved models."),
    )
    parser.add_argument(
        "--algo",
        choices=["rf", "xgb"],
        default="rf",
        help=_("Algorithm to use for Model_A training or inference."),
    )
    parser.add_argument(
        "--lang",
        default="en",
        choices=["es", "en"],
        help=_("Interface language."),
    )
    parser.add_argument(
        "--data-ana",
        "--data_ana",
        dest="data_ana",
        type=str,
        help=_("Path to the analog Parquet dataset."),
    )
    parser.add_argument(
        "--data-dig",
        "--data_dig",
        dest="data_dig",
        type=str,
        help=_("Path to the digital Parquet dataset used for training labels."),
    )
    parser.add_argument(
        "--model-out",
        "--model_out",
        dest="model_out",
        type=str,
        help=_("Output path for the trained model artifact."),
    )
    parser.add_argument(
        "--pred-out",
        "--pred_out",
        dest="pred_out",
        type=str,
        help=_("Output directory for metrics or predictions."),
    )
    parser.add_argument(
        "--model-paths",
        "--model_paths",
        dest="model_paths",
        nargs="+",
        help=_("One or more saved model artifacts to compare during evaluation."),
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help=_("Start date (YYYY-MM-DD HH:MM:SS)."),
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help=_("End date (YYYY-MM-DD HH:MM:SS)."),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help=_("Power threshold used to mark inactive rows."),
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help=_("Number of cross-validation folds."),
    )
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help=_("Include inactive rows with state 0 in the prediction output."),
    )
    parser.add_argument(
        "--pred-format",
        "--pred_format",
        dest="pred_format",
        choices=["xlsx", "csv"],
        default="xlsx",
        help=_("Prediction export format."),
    )
    return parser.parse_args()


def main() -> None:
    """Run the end-to-end Model_A workflow for training or prediction.

    The entry point keeps orchestration concerns in one place while delegating
    data preparation and model lifecycle logic to their respective classes.

    Training mode prepares labeled features, runs cross-validation, fits the
    final classifier, and persists both the model artifact and fold metrics.
    Prediction mode loads a previously trained artifact and applies it to a new
    analog time window without requiring digital labels at inference time.
    Evaluation mode is delegated to :func:`_run_evaluation`.

    Raises:
        SystemExit: In ``predict`` mode when ``--model-out`` or ``--pred-out``
            is not given explicitly (even though defaults exist for other
            modes), and on argument-parsing errors.
        ValueError: Propagated from data preparation or cross-validation
            (e.g. missing digital labels, fewer than two samples).

    Notes:
        The language is detected from raw ``sys.argv`` *before* parsing so
        that the ``--help`` text itself can be translated.
    """
    lang = _detect_language(sys.argv)
    translator = setup_i18n(lang)
    args = parse_arguments(translator)

    analog_path, digital_path, model_output, prediction_output = _resolve_paths(args)

    if args.mode == "train":
        processor = IndustrialDataProcessor(
            analog_path=str(analog_path),
            digital_path=str(digital_path),
            threshold=args.threshold,
        )
        training_data = processor.prepare_training_data(args.start, args.end)
        classifier = StateClassifier(model_type=args.algo, translator=translator)

        # Validation is reported before the final fit so experiment quality and
        # deployable artifact generation remain clearly separated.
        print(translator("Starting cross-validation from {} to {}").format(args.start, args.end))
        cv_result = classifier.cross_validate(
            training_data.features,
            training_data.labels,
            splits=args.cv_splits,
        )
        for fold_index, score in enumerate(cv_result.scores, start=1):
            print(translator("Fold {}: accuracy = {:.4f}").format(fold_index, score))
        print(
            translator("Average accuracy: {:.4f} (+/- {:.4f})").format(
                cv_result.mean,
                cv_result.std,
            )
        )

        training_accuracy = classifier.fit(training_data.features, training_data.labels)
        print(translator("Training accuracy: {:.4f}").format(training_accuracy))

        _save_metrics_report(
            prediction_output,
            analog_path,
            args.algo,
            args.start,
            cv_result.scores,
            translator,
        )
        model_output.parent.mkdir(parents=True, exist_ok=True)
        classifier.save(str(model_output))
        print(translator("Model saved to: {}").format(model_output))
        return

    if args.mode == "evaluate":
        _run_evaluation(
            args=args,
            analog_path=analog_path,
            digital_path=digital_path,
            prediction_output=prediction_output,
            translator=translator,
        )
        return

    if not args.model_out:
        sys.exit(translator("Model path is required for prediction mode (--model-out)."))
    if not args.pred_out:
        sys.exit(translator("Prediction output directory is required (--pred-out)."))

    processor = IndustrialDataProcessor(
        analog_path=str(analog_path),
        threshold=args.threshold,
    )
    classifier = StateClassifier.load(str(model_output), translator=translator)
    print(translator("Loading model from: {}").format(model_output))
    print(
        translator("Loading data for prediction from {} to {}").format(
            args.start,
            args.end,
        )
    )

    inference_data = processor.prepare_inference_data(args.start, args.end)
    prediction_features = inference_data.features
    if classifier.feature_columns:
        # Reindex to the training-time feature order stored with the artifact.
        # Note: no fill_value, so absent columns become NaN (see module
        # maintenance notes); extra columns are dropped.
        prediction_features = prediction_features.reindex(
            columns=classifier.feature_columns
        )
    prediction_frame = _build_prediction_frame(
        source_frame=inference_data.source_frame,
        active_mask=inference_data.active_mask,
        predictions=_predict_or_empty(classifier, prediction_features),
        include_inactive=args.include_inactive,
    )

    prediction_output.mkdir(parents=True, exist_ok=True)
    prediction_stem = f"predictions_{analog_path.name}_{classifier.model_type.upper()}"
    if args.pred_format == "csv":
        prediction_file = prediction_output / f"{prediction_stem}.csv"
        prediction_frame.to_csv(prediction_file, index=True)
    else:
        prediction_file = prediction_output / f"{prediction_stem}.xlsx"
        _prepare_excel_frame(prediction_frame).to_excel(prediction_file, index=True)
    print(translator("Predictions saved to: {}").format(prediction_file))


def _run_evaluation(
    *,
    args: argparse.Namespace,
    analog_path: Path,
    digital_path: Path,
    prediction_output: Path,
    translator: Callable[[str], str],
) -> None:
    """Compare one or more saved models against the requested time window.

    Loads the analog window (and the digital labels when the digital file
    exists), predicts with every artifact, writes the evaluation workbook and,
    when real labels are available, confusion-matrix and ROC plots.

    Args:
        args (argparse.Namespace): Parsed CLI arguments (uses ``start``,
            ``end``, ``threshold``, ``model_paths`` and ``model_out``).
        analog_path (pathlib.Path): Analog Parquet input.
        digital_path (pathlib.Path): Digital Parquet input; silently ignored
            if the file does not exist (labels are then unavailable).
        prediction_output (pathlib.Path): Output directory (created if
            needed).
        translator (Callable[[str], str]): Translation function for messages.

    Raises:
        SystemExit: If no model artifact path can be resolved.
        FileNotFoundError: If a resolved artifact does not exist.

    Notes:
        Output file names: ``evaluation_<analog stem>_<start>_<end>.xlsx``
        (timestamps sanitised with :func:`_sanitize_period_component`),
        ``cm_<model>.png/.pdf`` and ``roc_<model>.png/.pdf``. The ``summary``
        sheet is written first, then the per-model confusion-matrix sheets.
    """
    model_paths = _resolve_evaluation_model_paths(args, analog_path)
    if not model_paths:
        sys.exit(
            translator(
                "At least one model artifact is required for evaluation (--model-paths or --model-out)."
            )
        )

    processor = IndustrialDataProcessor(
        analog_path=str(analog_path),
        digital_path=str(digital_path) if digital_path.exists() else None,
        threshold=args.threshold,
    )
    evaluation_data = processor.prepare_evaluation_data(args.start, args.end)
    evaluation_frame, summary_frames = _build_evaluation_outputs(
        evaluation_data=evaluation_data,
        model_paths=model_paths,
        translator=translator,
    )

    prediction_output.mkdir(parents=True, exist_ok=True)
    report_stem = (
        f"evaluation_{analog_path.stem}_{_sanitize_period_component(args.start)}"
        f"_{_sanitize_period_component(args.end)}"
    )
    report_path = prediction_output / f"{report_stem}.xlsx"

    with pd.ExcelWriter(report_path) as writer:
        _prepare_excel_frame(evaluation_frame).to_excel(
            writer,
            sheet_name="predictions",
            index=True,
        )
        if summary_frames:
            summary_frames["summary"].to_excel(writer, sheet_name="summary", index=False)
            for sheet_name, frame in summary_frames.items():
                if sheet_name == "summary":
                    continue
                frame.to_excel(writer, sheet_name=sheet_name, index=True)

    print(translator("Evaluation report saved to: {}").format(report_path))

    if evaluation_data.labels is None:
        print(
            translator(
                "Real labels were not available, so confusion matrices and ROC curves were skipped."
            )
        )
        return

    for model_name, confusion_plot in _save_confusion_matrix_plots(
        evaluation_frame=evaluation_frame,
        evaluation_data=evaluation_data,
        model_paths=model_paths,
        output_dir=prediction_output,
        translator=translator,
    ):
        print(translator("Confusion matrix saved to: {}").format(confusion_plot))

    for model_name, roc_plot in _save_roc_plots(
        evaluation_frame=evaluation_frame,
        evaluation_data=evaluation_data,
        model_paths=model_paths,
        output_dir=prediction_output,
        translator=translator,
    ):
        print(translator("ROC curve saved to: {}").format(roc_plot))


def _resolve_evaluation_model_paths(
    args: argparse.Namespace,
    analog_path: Path,
) -> list[Path]:
    """Return the list of model artifacts to compare in evaluation mode.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        analog_path (pathlib.Path): Analog input, used to derive default
            artifact names.

    Returns:
        list[pathlib.Path]: In order of precedence: the ``--model-paths``
        values; else ``[--model-out]``; else the default RF and XGB artifacts
        ``src/models/Model_A/<analog file name>_{RF,XGB}.joblib`` under the
        repository root. Existence is not checked here.
    """
    if args.model_paths:
        return [Path(path) for path in args.model_paths]
    if args.model_out:
        return [Path(args.model_out)]

    repo_root = Path(__file__).resolve().parents[3]
    return [
        repo_root / "src" / "models" / "Model_A" / f"{analog_path.name}_RF.joblib",
        repo_root / "src" / "models" / "Model_A" / f"{analog_path.name}_XGB.joblib",
    ]


def _build_evaluation_outputs(
    *,
    evaluation_data: EvaluationDataset,
    model_paths: list[Path],
    translator: Callable[[str], str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Assemble timestamp-level predictions and tabular quality reports.

    Args:
        evaluation_data (EvaluationDataset): Prepared evaluation window.
        model_paths (list[pathlib.Path]): Artifacts to evaluate, in order.
        translator (Callable[[str], str]): Translation function for errors.

    Returns:
        tuple[pandas.DataFrame, dict[str, pandas.DataFrame]]:
        ``(evaluation_frame, sheet_frames)``. ``evaluation_frame`` is indexed
        by ``Time`` over the full source window and holds ``Real_Class``
        (nullable ``Int64``, all ``<NA>`` without labels) plus one ``int32``
        prediction column per model (inactive rows are ``0``).
        ``sheet_frames`` maps Excel-safe sheet names to per-model confusion
        matrices (``cm_<model>``, rows ``Real_<k>``, columns
        ``Predicted_<k>``) and, if any model was scored, a ``summary`` frame
        with ``Model``, ``Accuracy`` and ``ROC_AUC_Class_<k>`` columns. It is
        empty when real labels are unavailable.

    Raises:
        FileNotFoundError: If an artifact path does not exist.

    Notes:
        Accuracy and confusion matrices are computed over *all* timestamps of
        the window, including inactive rows where both real and predicted
        states are ``0`` by construction; this inflates accuracy relative to
        active-only scoring. ROC AUC values, in contrast, use only active rows
        with a non-zero real label (see :func:`_build_roc_summary`).
    """
    evaluation_frame = pd.DataFrame(index=evaluation_data.source_frame.index)
    evaluation_frame.index.name = "Time"
    if evaluation_data.labels is not None:
        evaluation_frame["Real_Class"] = evaluation_data.labels.astype("Int64")
    else:
        evaluation_frame["Real_Class"] = pd.Series(pd.NA, index=evaluation_frame.index, dtype="Int64")

    summaries: list[dict[str, object]] = []
    sheet_frames: dict[str, pd.DataFrame] = {}

    for model_path in model_paths:
        if not model_path.exists():
            raise FileNotFoundError(
                translator("Model artifact not found: {}").format(model_path)
            )
        classifier = StateClassifier.load(str(model_path), translator=translator)
        model_name = _build_model_label(classifier, model_path, evaluation_frame.columns)
        feature_frame = evaluation_data.features
        if classifier.feature_columns:
            feature_frame = feature_frame.reindex(columns=classifier.feature_columns)

        full_prediction_frame = _build_prediction_frame(
            source_frame=evaluation_data.source_frame,
            active_mask=evaluation_data.active_mask,
            predictions=_predict_or_empty(classifier, feature_frame),
            include_inactive=True,
        ).rename(columns={"Predicted_State": model_name})
        evaluation_frame[model_name] = full_prediction_frame[model_name].astype(np.int32)

        if evaluation_data.labels is None:
            continue

        labels = evaluation_data.labels.astype(np.int32)
        predictions = evaluation_frame[model_name].astype(np.int32)
        # Note: the union of real and predicted codes keeps the confusion
        # matrix square even when a class is never predicted (or never real).
        class_labels = np.unique(np.concatenate([labels.to_numpy(), predictions.to_numpy()]))
        matrix = confusion_matrix(labels, predictions, labels=class_labels)
        confusion_frame = pd.DataFrame(
            matrix,
            index=[f"Real_{label}" for label in class_labels],
            columns=[f"Predicted_{label}" for label in class_labels],
        )
        sheet_frames[_sheet_name(f"cm_{model_name}")] = confusion_frame

        summary_row: dict[str, object] = {
            "Model": model_name,
            "Accuracy": float(accuracy_score(labels, predictions)),
        }
        probabilities = _predict_probabilities_or_empty(classifier, feature_frame)
        roc_summary = _build_roc_summary(
            labels=labels,
            active_mask=evaluation_data.active_mask,
            classifier=classifier,
            probabilities=probabilities,
        )
        for class_name, class_auc in roc_summary.items():
            summary_row[f"ROC_AUC_Class_{class_name}"] = class_auc
        summaries.append(summary_row)

    if summaries:
        sheet_frames["summary"] = pd.DataFrame(summaries)
    return evaluation_frame, sheet_frames


def _build_model_label(
    classifier: StateClassifier,
    model_path: Path,
    existing_columns: pd.Index,
) -> str:
    """Build a readable, unique prediction-column name for a model artifact.

    Args:
        classifier (StateClassifier): Loaded classifier (its ``model_type``
            forms the base name).
        model_path (pathlib.Path): Artifact path; its stem disambiguates
            duplicates.
        existing_columns (pandas.Index): Column names already in use.

    Returns:
        str: ``"<TYPE>_Prediction"`` (e.g. ``"RF_Prediction"``), or
        ``"<TYPE>_Prediction_<artifact stem>"`` if the base name is taken.
    """
    base_name = f"{classifier.model_type.upper()}_Prediction"
    if base_name not in existing_columns:
        return base_name
    return f"{base_name}_{model_path.stem}"


def _predict_probabilities_or_empty(
    classifier: StateClassifier,
    features: pd.DataFrame,
) -> np.ndarray:
    """Predict class probabilities or return an empty 2D array.

    Args:
        classifier (StateClassifier): Fitted classifier.
        features (pandas.DataFrame): Active-row feature matrix.

    Returns:
        numpy.ndarray: Array of shape ``(n_rows, K)`` from
        :meth:`StateClassifier.predict_proba`, or an empty ``(0, K)`` array
        when ``features`` is empty (``K`` = number of encoder classes, ``0``
        if the encoder is unfitted).

    Raises:
        AttributeError: Propagated if the estimator has no probability
            support.
    """
    if features.empty:
        class_count = len(getattr(classifier.label_encoder, "classes_", []))
        return np.empty((0, class_count), dtype=float)
    return classifier.predict_proba(features)


def _build_roc_summary(
    *,
    labels: pd.Series,
    active_mask: pd.Series,
    classifier: StateClassifier,
    probabilities: np.ndarray,
) -> dict[int, float]:
    """Compute one-vs-rest ROC AUC values for each evaluable active class.

    Args:
        labels (pandas.Series): Real labels over the full source window.
        active_mask (pandas.Series): Boolean activity mask aligned with
            ``labels``.
        classifier (StateClassifier): Fitted classifier; its
            ``label_encoder.classes_`` define the probability columns.
        probabilities (numpy.ndarray): Probabilities for the *active* rows
            only, in the same order as ``labels.loc[active_mask]``.

    Returns:
        dict[int, float]: Mapping ``state code -> AUC`` for every class whose
        one-vs-rest target has both positives and negatives. Empty when there
        are no active labelled rows, no probabilities, fewer than two model
        classes, or fewer than two distinct real classes.

    Notes:
        Active rows with real label ``0`` (no earlier digital event) are
        excluded, because state ``0`` is not a class the model was trained on.
        With exactly two classes :func:`~sklearn.preprocessing.label_binarize`
        returns a single column, which is expanded to two complementary
        columns so that indices match ``probabilities``.
    """
    active_labels = labels.loc[active_mask].astype(np.int32)
    nonzero_mask = active_labels != 0
    active_labels = active_labels.loc[nonzero_mask]
    if active_labels.empty or probabilities.size == 0:
        return {}

    # Invariant: probabilities rows follow labels.loc[active_mask]; the same
    # positional nonzero mask selects matching rows in both.
    active_probabilities = probabilities[nonzero_mask.to_numpy()]
    classes = np.asarray(classifier.label_encoder.classes_)
    if len(classes) < 2 or len(active_labels.unique()) < 2:
        return {}

    binarized = label_binarize(active_labels, classes=classes)
    if binarized.ndim == 1:
        binarized = np.column_stack([1 - binarized, binarized])

    roc_summary: dict[int, float] = {}
    for column_index, class_label in enumerate(classes):
        y_true = binarized[:, column_index]
        if np.unique(y_true).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, active_probabilities[:, column_index])
        roc_summary[int(class_label)] = float(auc(fpr, tpr))
    return roc_summary


def _save_roc_plots(
    *,
    evaluation_frame: pd.DataFrame,
    evaluation_data: EvaluationDataset,
    model_paths: list[Path],
    output_dir: Path,
    translator: Callable[[str], str],
) -> list[tuple[str, Path]]:
    """Render one ROC PNG per model and a title-free PDF companion.

    Args:
        evaluation_frame (pandas.DataFrame): Output of
            :func:`_build_evaluation_outputs` (currently unused here).
        evaluation_data (EvaluationDataset): Prepared evaluation window.
        model_paths (list[pathlib.Path]): Artifacts to plot.
        output_dir (pathlib.Path): Directory for ``roc_<model>.png/.pdf``.
        translator (Callable[[str], str]): Translation function for errors.

    Returns:
        list[tuple[str, pathlib.Path]]: ``(model_name, png_path)`` for every
        model that produced at least one ROC curve. Models are skipped under
        the same conditions as in :func:`_build_roc_summary`.

    Notes:
        Each artifact is re-loaded and probabilities are recomputed (the
        logic duplicates :func:`_build_roc_summary`). ``model_name`` is built
        against an empty index, so it never carries the artifact-stem suffix
        used in the workbook for duplicate model types.
    """
    saved_plots: list[tuple[str, Path]] = []
    labels = evaluation_data.labels
    if labels is None:
        return saved_plots

    for model_path in model_paths:
        classifier = StateClassifier.load(str(model_path), translator=translator)
        model_name = _build_model_label(classifier, model_path, pd.Index([]))
        feature_frame = evaluation_data.features
        if classifier.feature_columns:
            feature_frame = feature_frame.reindex(columns=classifier.feature_columns)
        probabilities = _predict_probabilities_or_empty(classifier, feature_frame)

        active_labels = labels.loc[evaluation_data.active_mask].astype(np.int32)
        nonzero_mask = active_labels != 0
        active_labels = active_labels.loc[nonzero_mask]
        if active_labels.empty or probabilities.size == 0:
            continue

        classes = np.asarray(classifier.label_encoder.classes_)
        if len(classes) < 2 or len(active_labels.unique()) < 2:
            continue

        active_probabilities = probabilities[nonzero_mask.to_numpy()]
        binarized = label_binarize(active_labels, classes=classes)
        if binarized.ndim == 1:
            binarized = np.column_stack([1 - binarized, binarized])

        figure, axis = plt.subplots(figsize=(7, 5))
        plotted_any = False
        for column_index, class_label in enumerate(classes):
            y_true = binarized[:, column_index]
            if np.unique(y_true).size < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, active_probabilities[:, column_index])
            axis.plot(
                fpr,
                tpr,
                label=f"Class {class_label} (AUC={auc(fpr, tpr):.3f})",
            )
            plotted_any = True

        if not plotted_any:
            plt.close(figure)
            continue

        axis.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.legend(loc="lower right")
        axis.grid(alpha=0.2)
        plot_stem = output_dir / f"roc_{_sheet_name(model_name)}"
        plot_path = _save_plot_variants(
            figure=figure,
            axis=axis,
            stem=plot_stem,
            title=f"ROC - {model_name}",
        )
        plt.close(figure)
        saved_plots.append((model_name, plot_path))

    return saved_plots


def _save_confusion_matrix_plots(
    *,
    evaluation_frame: pd.DataFrame,
    evaluation_data: EvaluationDataset,
    model_paths: list[Path],
    output_dir: Path,
    translator: Callable[[str], str],
) -> list[tuple[str, Path]]:
    """Render one confusion-matrix PNG per model and a title-free PDF companion.

    Args:
        evaluation_frame (pandas.DataFrame): Output of
            :func:`_build_evaluation_outputs`; prediction columns are read
            from it.
        evaluation_data (EvaluationDataset): Prepared evaluation window.
        model_paths (list[pathlib.Path]): Artifacts to plot.
        output_dir (pathlib.Path): Directory for ``cm_<model>.png/.pdf``.
        translator (Callable[[str], str]): Translation function for errors.

    Returns:
        list[tuple[str, pathlib.Path]]: ``(model_name, png_path)`` per model;
        empty when real labels are unavailable.

    Raises:
        KeyError: If the expected prediction column is missing from
            ``evaluation_frame``.

    Notes:
        Matrices cover all timestamps of the window (inactive rows included),
        consistent with the workbook's ``cm_*`` sheets. Cell annotations switch
        to white text above half of the maximum count for readability.
    """
    saved_plots: list[tuple[str, Path]] = []
    labels = evaluation_data.labels
    if labels is None:
        return saved_plots

    true_labels = labels.astype(np.int32)
    for model_path in model_paths:
        classifier = StateClassifier.load(str(model_path), translator=translator)
        model_name = _build_model_label(classifier, model_path, pd.Index([]))
        predictions = evaluation_frame[model_name].astype(np.int32)
        class_labels = np.unique(
            np.concatenate([true_labels.to_numpy(), predictions.to_numpy()])
        )
        matrix = confusion_matrix(true_labels, predictions, labels=class_labels)

        figure, axis = plt.subplots(figsize=(6.5, 5.5))
        image = axis.imshow(matrix, cmap="Blues")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks(np.arange(len(class_labels)))
        axis.set_yticks(np.arange(len(class_labels)))
        axis.set_xticklabels(class_labels)
        axis.set_yticklabels(class_labels)

        threshold = matrix.max() / 2 if matrix.size else 0
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                )

        plot_stem = output_dir / f"cm_{_sheet_name(model_name)}"
        plot_path = _save_plot_variants(
            figure=figure,
            axis=axis,
            stem=plot_stem,
            title=f"Confusion Matrix - {model_name}",
        )
        plt.close(figure)
        saved_plots.append((model_name, plot_path))

    return saved_plots


def _save_plot_variants(
    *,
    figure: plt.Figure,
    axis: plt.Axes,
    stem: Path,
    title: str,
) -> Path:
    """Save a titled PNG and a title-free vector PDF for the same figure.

    Args:
        figure (matplotlib.figure.Figure): Figure to save.
        axis (matplotlib.axes.Axes): Axis whose title is toggled.
        stem (pathlib.Path): Output path without suffix; ``.png`` and
            ``.pdf`` are appended.
        title (str): Title shown on the PNG only.

    Returns:
        pathlib.Path: Path of the PNG file (150 dpi).

    Notes:
        The PDF variant is title-free so it can be embedded in papers with a
        caption. On return the axis title is left empty; the caller is
        responsible for closing the figure.
    """
    axis.set_title(title)
    figure.tight_layout()
    png_path = stem.with_suffix(".png")
    figure.savefig(png_path, dpi=150)

    axis.set_title("")
    figure.tight_layout()
    figure.savefig(stem.with_suffix(".pdf"))
    return png_path


def _sanitize_period_component(value: str) -> str:
    """Convert a timestamp string into a filename-safe suffix.

    Args:
        value (str): Timestamp text, e.g. ``"2022-02-21 00:00:00"``.

    Returns:
        str: ``value`` with ``:`` replaced by ``-`` and spaces by ``_``
        (e.g. ``"2022-02-21_00-00-00"``).

    Examples:
        >>> _sanitize_period_component("2022-02-21 00:00:00")
        '2022-02-21_00-00-00'
    """
    return value.replace(":", "-").replace(" ", "_")


def _sheet_name(value: str) -> str:
    """Return an Excel-safe sheet/file stem fragment.

    Args:
        value (str): Arbitrary label (e.g. ``"cm_RF_Prediction"``).

    Returns:
        str: ``value`` with every character other than alphanumerics, ``_``
        and ``-`` replaced by ``_``, truncated to 31 characters (Excel's
        sheet-name limit).

    Notes:
        Truncation can make two long labels collide; sheet names (and plot
        file names, which reuse this helper) are then overwritten.

    Examples:
        >>> _sheet_name("cm_RF Prediction")
        'cm_RF_Prediction'
    """
    sanitized = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in value)
    return sanitized[:31]


def _detect_language(argv: list[str]) -> str:
    """Extract the requested language before parsing the translated CLI.

    Args:
        argv (list[str]): Raw command-line token list (``sys.argv``).

    Returns:
        str: The token that follows ``--lang``, or ``"en"`` when ``--lang`` is
        absent or is the last token. The value is not validated here; an
        unsupported code reaches :func:`~iabm.utils.setup_i18n` (which falls
        back to untranslated messages) before :mod:`argparse` rejects it.

    Notes:
        Only the space-separated form ``--lang es`` is detected; the
        ``--lang=es`` form is parsed correctly by argparse but the interface
        stays in English.

    Examples:
        >>> _detect_language(["industrial-id", "--lang", "es"])
        'es'
        >>> _detect_language(["industrial-id"])
        'en'
    """
    for option in ("--lang",):
        if option in argv:
            try:
                return argv[argv.index(option) + 1]
            except (IndexError, ValueError):
                return "en"
    return "en"


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Resolve input and output paths relative to the repository layout.

    Args:
        args (argparse.Namespace): Parsed command-line arguments (uses
            ``data_ana``, ``data_dig``, ``model_out``, ``pred_out`` and
            ``algo``).

    Returns:
        tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]: A tuple
        with analog input path, digital input path, model artifact path, and
        report/prediction output directory. Explicit arguments win; otherwise
        ``data/analogicas_nonans.parquet``, ``data/digitales.parquet``,
        ``src/models/Model_A/<analog file name>_<ALGO>.joblib`` and
        ``src/predictions/Model_A`` under the repository root are used.

    Notes:
        Paths are only built, never checked for existence or created here.
    """
    # Note: parents[3] climbs iabm/ -> Model_A/ -> src/ -> repository root.
    repo_root = Path(__file__).resolve().parents[3]
    analog_path = Path(args.data_ana) if args.data_ana else repo_root / "data" / "analogicas_nonans.parquet"
    digital_path = Path(args.data_dig) if args.data_dig else repo_root / "data" / "digitales.parquet"
    default_model = repo_root / "src" / "models" / "Model_A" / f"{analog_path.name}_{args.algo.upper()}.joblib"
    default_reports = repo_root / "src" / "predictions" / "Model_A"
    model_output = Path(args.model_out) if args.model_out else default_model
    prediction_output = Path(args.pred_out) if args.pred_out else default_reports
    return analog_path, digital_path, model_output, prediction_output


def _save_metrics_report(
    output_dir: Path,
    analog_path: Path,
    algorithm: str,
    start: str,
    scores: np.ndarray,
    translator: Callable[[str], str],
) -> None:
    """Persist fold-wise cross-validation metrics to an Excel report.

    Args:
        output_dir (pathlib.Path): Directory where the report should be
            written (created if needed).
        analog_path (pathlib.Path): Input analog dataset path used to derive
            the report name.
        algorithm (str): Short identifier of the trained algorithm
            (``"rf"``/``"xgb"``; upper-cased in the file name).
        start (str): Training-period start timestamp used for report naming
            (only its first 10 characters, the date, are used).
        scores (numpy.ndarray): Fold-wise cross-validation scores.
        translator (Callable[[str], str]): Translation function for
            user-facing messages.

    Notes:
        Writes ``<analog file name>_<ALGO>_metrics_<YYYY-MM-DD>.xlsx`` with
        columns ``Fold`` (1-based) and ``Accuracy``. Runs with the same
        analog file, algorithm and start date overwrite each other.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(
        {
            "Fold": np.arange(1, len(scores) + 1),
            "Accuracy": scores,
        }
    )
    report_path = output_dir / f"{analog_path.name}_{algorithm.upper()}_metrics_{start[:10]}.xlsx"
    report.to_excel(report_path, index=False)
    print(translator("Metrics saved to: {}").format(report_path))


def _build_prediction_frame(
    *,
    source_frame: pd.DataFrame,
    active_mask: pd.Series,
    predictions: np.ndarray,
    include_inactive: bool,
) -> pd.DataFrame:
    """Create a prediction table aligned with the original analog timestamps.

    Args:
        source_frame (pandas.DataFrame): Full analog window used for
            inference.
        active_mask (pandas.Series): Boolean mask selecting active rows in the
            source frame (aligned with its index).
        predictions (numpy.ndarray): Predicted labels for active rows, in the
            order of ``source_frame.index[active_mask]``. Its length must equal
            the number of active rows.
        include_inactive (bool): Whether to emit inactive rows with a default
            zero state.

    Returns:
        pandas.DataFrame: A DataFrame with index ``Time`` and a single column
        ``Predicted_State``, ready to be exported or further post-processed.
        Contains only active rows unless ``include_inactive`` is ``True``.

    Raises:
        ValueError: If ``predictions`` and the number of active rows differ in
            length.
    """
    active_index = source_frame.index[active_mask]
    active_predictions = pd.DataFrame(
        {"Predicted_State": predictions},
        index=active_index,
    )
    if not include_inactive:
        active_predictions.index.name = "Time"
        return active_predictions

    full_predictions = pd.DataFrame(
        {"Predicted_State": np.zeros(len(source_frame), dtype=np.int32)},
        index=source_frame.index,
    )
    # Note: label-based assignment on the active timestamps; inactive rows
    # keep the default state 0.
    full_predictions.loc[active_index, "Predicted_State"] = predictions
    full_predictions.index.name = "Time"
    return full_predictions


def _predict_or_empty(
    classifier: StateClassifier,
    features: pd.DataFrame,
) -> np.ndarray:
    """Predict labels or return an empty array when no active rows are available.

    Args:
        classifier (StateClassifier): Fitted classifier artifact.
        features (pandas.DataFrame): Inference feature matrix.

    Returns:
        numpy.ndarray: A NumPy array of predicted labels. The array is empty
        (``int32``) when the requested time window contains no active rows
        after preprocessing.

    Notes:
        Why: scikit-learn scalers reject zero-row inputs, so the empty case is
        short-circuited.
    """
    if features.empty:
        return np.array([], dtype=np.int32)
    return classifier.predict(features)


def _prepare_excel_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert timezone-aware indices into Excel-safe string representations.

    Args:
        frame (pandas.DataFrame): DataFrame about to be exported to Excel.

    Returns:
        pandas.DataFrame: A copy of the DataFrame whose index can be
        serialized by Excel writers. A timezone-aware
        :class:`pandas.DatetimeIndex` is converted to strings formatted as
        ``"%Y-%m-%d %H:%M:%S%z"`` (e.g. ``"2022-01-01 00:00:00+0100"``);
        any other index is left unchanged.

    Notes:
        Why: pandas refuses to write timezone-aware datetimes to Excel. The
        string conversion preserves the UTC offset instead of dropping it.
    """
    excel_frame = frame.copy()
    if isinstance(excel_frame.index, pd.DatetimeIndex) and excel_frame.index.tz is not None:
        excel_frame.index = excel_frame.index.strftime("%Y-%m-%d %H:%M:%S%z")
    return excel_frame


if __name__ == "__main__":
    main()
