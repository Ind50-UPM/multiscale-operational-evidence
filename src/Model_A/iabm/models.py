"""Model abstractions for Model_A industrial-state classifiers.

This module holds the learning component of Model_A: a thin, persistent
wrapper around scikit-learn compatible tree ensembles (Random Forest and
XGBoost) that maps analog feature vectors to operating-state codes. The
predicted states are the input of Model_B (behavioural sequences).

Main public objects:
    * :class:`StateClassifier` -- owns the full estimator lifecycle: feature
      standardisation, label encoding, cross-validation, final fit,
      persistence (``joblib``) and inference.
    * :class:`CrossValidationResult` -- fold-wise accuracy scores with
      ``mean``/``std`` summaries.
    * :class:`FoldLabelEncoderClassifier` -- internal adapter that re-encodes
      labels per CV fold for XGBoost.

Data contracts:
    ``X`` is a :class:`pandas.DataFrame` of numeric features (by default
    ``Vrms1..3``, ``Irms1..3``, ``PF1..3``; see
    :class:`~iabm.data_processor.IndustrialDataProcessor`). ``y`` contains the
    original integer state codes (column ``estado``), which need not be
    contiguous (e.g. ``{1, 2, 4}``); they are encoded internally to
    ``0..K-1`` and decoded back on prediction. Persisted artifacts are
    ``joblib`` dictionaries with keys ``model``, ``scaler``,
    ``label_encoder``, ``model_type``, ``params`` and ``feature_columns``.

.. mermaid::

    classDiagram
        class StateClassifier {
            +model_type: str
            +params: dict
            +scaler: StandardScaler
            +label_encoder: LabelEncoder
            +estimator: RandomForestClassifier | XGBClassifier
            +feature_columns: list~str~
            +fit(X, y) float
            +cross_validate(X, y, splits, shuffle, random_state) CrossValidationResult
            +predict(X) ndarray
            +predict_proba(X) ndarray
            +save(file_path)
            +load(file_path, translator)$ StateClassifier
        }
        class CrossValidationResult {
            +scores: ndarray
            +mean: float
            +std: float
        }
        class FoldLabelEncoderClassifier {
            +estimator
            +label_encoder_
            +estimator_
            +fit(X, y)
            +predict(X)
        }
        BaseEstimator <|-- FoldLabelEncoderClassifier
        ClassifierMixin <|-- FoldLabelEncoderClassifier
        StateClassifier ..> CrossValidationResult : returns
        StateClassifier ..> FoldLabelEncoderClassifier : wraps XGB in CV

Maintenance notes:
    Cross-validation builds a fresh ``StandardScaler`` + estimator pipeline
    per fold, so scaling statistics are never fitted on validation data (no
    leakage). The final :meth:`StateClassifier.fit` refits the instance-level
    scaler and label encoder on all training rows. Folds are shuffled
    ``(Stratified)KFold`` splits over individual timestamps, so temporally
    adjacent (highly correlated) samples can fall in both train and validation
    folds; CV accuracy is therefore an optimistic estimate compared with a
    blocked/time-ordered split. Supported ``model_type`` values are ``"rf"``
    and ``"xgb"``; adding a new family requires updating
    :meth:`StateClassifier._get_default_params`,
    :meth:`StateClassifier._initialize_estimator` and the CLI ``--algo``
    choices. Artifact keys are a compatibility contract with saved models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


@dataclass(frozen=True)
class CrossValidationResult:
    """Summarize fold-wise validation scores produced by the classifier API.

    Exposing the result as a dataclass makes downstream reporting clearer and
    keeps aggregate statistics close to the original fold-level scores.

    Attributes:
        scores (numpy.ndarray): One-dimensional ``float`` array with one
            accuracy value (in ``[0, 1]``) per fold, in fold order.
    """

    scores: np.ndarray

    @property
    def mean(self) -> float:
        """float: Average score across all folds."""
        return float(self.scores.mean())

    @property
    def std(self) -> float:
        """float: Population standard deviation (``ddof=0``) across all folds."""
        return float(self.scores.std())


class FoldLabelEncoderClassifier(BaseEstimator, ClassifierMixin):
    """Wrap an estimator so each fit uses fold-local contiguous class labels.

    XGBoost expects class labels presented during ``fit`` to be contiguous
    integers starting at zero. During cross-validation, some training folds may
    not contain every class present in the global dataset, which makes a
    globally encoded target vector invalid for that fold. This wrapper applies a
    fresh label encoding on every fit and maps predictions back to the original
    labels expected by scikit-learn scorers.

    Attributes:
        estimator (sklearn.base.BaseEstimator): Unfitted template estimator.
        label_encoder_ (sklearn.preprocessing.LabelEncoder): Fold-local encoder
            fitted in :meth:`fit`.
        estimator_ (sklearn.base.BaseEstimator): Fitted clone of
            ``estimator`` (reconfigured for XGBoost class counts).

    Notes:
        "Original labels" here are the labels passed to :meth:`fit`; inside
        :meth:`StateClassifier.cross_validate` these are already the *global*
        encoded labels, so this wrapper performs a second, fold-local encoding.
        A validation fold containing a class unseen in its training fold can
        never be predicted correctly, which lowers that fold's score.
    """

    def __init__(self, estimator: BaseEstimator) -> None:
        """Store the wrapped estimator used for fold-local training.

        Args:
            estimator (sklearn.base.BaseEstimator): Base estimator cloned
                inside each cross-validation fit.
        """
        self.estimator = estimator

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "FoldLabelEncoderClassifier":
        """Fit the wrapped estimator with a fold-local label encoding.

        Args:
            X (pandas.DataFrame | numpy.ndarray): Fold-local feature matrix of
                shape ``(n_samples, n_features)``.
            y (pandas.Series | numpy.ndarray): Fold-local label vector of
                length ``n_samples``.

        Returns:
            FoldLabelEncoderClassifier: The fitted wrapper instance.
        """
        self.label_encoder_ = LabelEncoder()
        encoded_y = self.label_encoder_.fit_transform(np.asarray(y))
        self.estimator_ = clone(self.estimator)
        # XGBoost requires contiguous zero-based labels inside each fold, which
        # is not guaranteed when rare classes disappear from a split.
        self.estimator_ = _configure_xgb_estimator(self.estimator_, encoded_y)
        self.estimator_.fit(X, encoded_y)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict labels and map them back to the original fold label space.

        Args:
            X (pandas.DataFrame | numpy.ndarray): Fold-local feature matrix.

        Returns:
            numpy.ndarray: Predictions expressed in the original label space
            expected by the scoring function.

        Notes:
            If the wrapped estimator returns a 2-D array (per-class scores),
            the arg-max column is taken as the encoded class.
        """
        encoded_predictions = np.asarray(self.estimator_.predict(X))
        if encoded_predictions.ndim > 1:
            encoded_predictions = encoded_predictions.argmax(axis=1)
        return self.label_encoder_.inverse_transform(np.asarray(encoded_predictions))

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Expose wrapped-estimator parameters for scikit-learn compatibility.

        Args:
            deep (bool): Whether to include nested estimator parameters using
                the ``estimator__<name>`` convention.

        Returns:
            Dict[str, Any]: A parameter dictionary compatible with
            scikit-learn cloning. Always contains the ``estimator`` key.
        """
        params = {"estimator": self.estimator}
        if deep and hasattr(self.estimator, "get_params"):
            for key, value in self.estimator.get_params(deep=True).items():
                params[f"estimator__{key}"] = value
        return params

    def set_params(self, **params: Any) -> "FoldLabelEncoderClassifier":
        """Propagate parameter updates to the wrapped estimator when requested.

        Args:
            **params (Any): Wrapper parameter ``estimator`` (replaces the
                template) and/or nested ``estimator__<name>`` parameters
                forwarded to ``estimator.set_params``. Other keys are
                silently ignored.

        Returns:
            FoldLabelEncoderClassifier: The updated wrapper instance.
        """
        estimator = params.pop("estimator", None)
        if estimator is not None:
            self.estimator = estimator
        nested_params = {
            key.split("__", 1)[1]: value
            for key, value in params.items()
            if key.startswith("estimator__")
        }
        if nested_params and hasattr(self.estimator, "set_params"):
            self.estimator.set_params(**nested_params)
        return self


def _configure_xgb_estimator(
    estimator: BaseEstimator,
    labels: np.ndarray,
) -> BaseEstimator:
    """Inject class-count settings required by XGBoost when fitting labels.

    Args:
        estimator (sklearn.base.BaseEstimator): Estimator about to be fitted.
        labels (numpy.ndarray): Encoded label vector used for fitting
            (contiguous ``0..K-1``).

    Returns:
        sklearn.base.BaseEstimator: The original estimator (same object) for
        non-XGBoost models, or an unfitted configured clone with
        ``num_class=K`` (at least ``1``) and ``objective="multi:softprob"``
        for XGBoost.

    Notes:
        Because a clone is returned for XGBoost, callers must rebind the
        result (``est = _configure_xgb_estimator(est, y)``) before fitting.
    """
    if not isinstance(estimator, XGBClassifier):
        return estimator

    configured = clone(estimator)
    unique_classes = np.unique(labels)
    configured.set_params(
        num_class=max(len(unique_classes), 1),
        objective="multi:softprob",
    )
    return configured


class StateClassifier:
    """High-level wrapper around the estimator lifecycle used by Model_A.

    The class keeps scaling, label encoding, validation, persistence, and
    inference in one cohesive object so command-line orchestration stays thin
    and future model variants can share the same interface.

    Attributes:
        model_type (str): Estimator family, ``"rf"`` or ``"xgb"``.
        params (Dict[str, Any]): Hyperparameters passed to the estimator.
        scaler (sklearn.preprocessing.StandardScaler): Feature scaler fitted
            in :meth:`fit`.
        label_encoder (sklearn.preprocessing.LabelEncoder): Maps original
            state codes to ``0..K-1``; fitted in :meth:`fit` (and also refit
            by :meth:`cross_validate`).
        estimator (sklearn.ensemble.RandomForestClassifier |
            xgboost.XGBClassifier): Underlying classifier.
        feature_columns (list[str]): Training-time feature order, empty until
            :meth:`fit` or :meth:`load`.

    Examples:
        >>> clf = StateClassifier(model_type="rf")  # doctest: +SKIP
        >>> cv = clf.cross_validate(X, y, splits=5)  # doctest: +SKIP
        >>> clf.fit(X, y)  # doctest: +SKIP
        >>> clf.save("rf_model.joblib")  # doctest: +SKIP
        >>> StateClassifier.load("rf_model.joblib").predict(X_new)  # doctest: +SKIP
    """

    def __init__(
        self,
        model_type: str = "rf",
        params: Optional[Dict[str, Any]] = None,
        translator: Optional[Callable[[str], str]] = None,
    ) -> None:
        """Initialize the classifier wrapper for the selected algorithm.

        Args:
            model_type (str): Short identifier of the underlying estimator
                family: ``"rf"`` (Random Forest) or ``"xgb"`` (XGBoost).
            params (Dict[str, Any] | None): Optional hyperparameter overrides.
                When given (non-empty), they *replace* the defaults entirely
                rather than being merged with them.
            translator (Callable[[str], str] | None): Optional translation
                function for user-facing errors (see
                :func:`iabm.utils.setup_i18n`). Defaults to identity.

        Raises:
            ValueError: If ``model_type`` is unsupported (raised by
                :meth:`_get_default_params` when ``params`` is falsy, otherwise
                by :meth:`_initialize_estimator`).
        """
        self._ = translator or (lambda s: s)
        self.model_type = model_type
        self.params = params or self._get_default_params(model_type)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.estimator = self._initialize_estimator()
        self.feature_columns: list[str] = []

    @staticmethod
    def _get_default_params(model_type: str) -> Dict[str, Any]:
        """Return sensible default hyperparameters for the requested model.

        Args:
            model_type (str): Short identifier of the underlying estimator
                family (``"rf"`` or ``"xgb"``).

        Returns:
            Dict[str, Any]: A dictionary of baseline hyperparameters. Both
            families use ``n_estimators=100``, ``random_state=42`` and
            ``n_jobs=8``; RF uses ``criterion="entropy"``; XGB uses
            ``learning_rate=0.1``, ``max_depth=6``, ``eval_metric="mlogloss"``.

        Raises:
            ValueError: If ``model_type`` is not ``"rf"`` or ``"xgb"``
                (message not translated).
        """
        if model_type == "rf":
            return {
                "n_estimators": 100,
                "criterion": "entropy",
                "random_state": 42,
                "n_jobs": 8,
            }
        if model_type == "xgb":
            return {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "random_state": 42,
                "n_jobs": 8,
                "eval_metric": "mlogloss",
            }
        raise ValueError(f"Unsupported model type: {model_type}")

    def _initialize_estimator(self) -> RandomForestClassifier | XGBClassifier:
        """Build the underlying scikit-learn compatible estimator.

        Returns:
            RandomForestClassifier | XGBClassifier: A scikit-learn compatible
            classifier instance matching ``self.model_type``, configured with
            ``self.params``. XGBoost is created with
            ``objective="multi:softprob"``.

        Raises:
            ValueError: If ``self.model_type`` is unsupported (translated
                message).
        """
        if self.model_type == "rf":
            return RandomForestClassifier(**self.params)
        if self.model_type == "xgb":
            # Note: ``objective`` is passed explicitly, so ``self.params`` must
            # not also contain an ``objective`` key (duplicate keyword).
            return XGBClassifier(**self.params, objective="multi:softprob")
        raise ValueError(
            self._("Unsupported algorithm: {}.").format(self.model_type)
        )

    def _build_cv_estimator(self) -> BaseEstimator:
        """Return the estimator instance that should be used inside CV folds.

        Returns:
            sklearn.base.BaseEstimator: An unfitted clone of
            ``self.estimator`` for Random Forest, or a
            :class:`FoldLabelEncoderClassifier` wrapping ``self.estimator``
            for XGBoost (it clones internally on every fit).
        """
        if self.model_type == "xgb":
            return FoldLabelEncoderClassifier(self.estimator)
        return clone(self.estimator)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> float:
        """Fit the scaler and estimator and return the training accuracy.

        Args:
            X (pandas.DataFrame): Training feature matrix; its column order is
                stored in :attr:`feature_columns`.
            y (pandas.Series | numpy.ndarray): Original training labels
                (state codes).

        Returns:
            float: In-sample accuracy measured on the fitted training data
            (optimistic; use :meth:`cross_validate` for generalisation).
        """
        self.feature_columns = list(X.columns)
        encoded_labels = self.label_encoder.fit_transform(np.asarray(y))
        scaled_features = self.scaler.fit_transform(X)
        # The training artifact stores globally fitted preprocessing plus the
        # final estimator configured for the full label set.
        self.estimator = _configure_xgb_estimator(self.estimator, encoded_labels)
        self.estimator.fit(scaled_features, encoded_labels)
        predictions = np.asarray(self.estimator.predict(scaled_features))
        if predictions.ndim > 1:
            predictions = predictions.argmax(axis=1)
        return float(accuracy_score(encoded_labels, predictions))

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> CrossValidationResult:
        """Evaluate the configured estimator with stratified cross-validation.

        Args:
            X (pandas.DataFrame): Feature matrix.
            y (pandas.Series | numpy.ndarray): Original state labels before
                encoding.
            splits (int): Requested number of folds. The effective number is
                ``min(splits, n_samples)`` and, when stratifying, also capped
                by the size of the smallest class.
            shuffle (bool): Whether to shuffle the samples before splitting.
            random_state (int): Seed used when shuffling folds.

        Returns:
            CrossValidationResult: A :class:`CrossValidationResult` with
            per-fold accuracy scores and summary statistics.

        Raises:
            ValueError: If fewer than two samples are available (translated
                message). scikit-learn may also raise :class:`ValueError`
                from the splitter (for example when ``shuffle=False`` is
                combined with a non-``None`` ``random_state``).

        Notes:
            * Uses :class:`~sklearn.model_selection.StratifiedKFold` when the
              smallest class has at least two samples, otherwise falls back to
              plain :class:`~sklearn.model_selection.KFold`.
            * Each fold fits its own ``StandardScaler`` inside a
              :class:`~sklearn.pipeline.Pipeline`, which prevents leakage of
              validation statistics into scaling.
            * Side effect: :attr:`label_encoder` is refit on ``y``. The
              instance estimator and scaler are not fitted.
        """
        y_array = np.asarray(y)
        encoded_labels = self.label_encoder.fit_transform(y_array)
        min_class_count = int(pd.Series(encoded_labels).value_counts().min())
        effective_splits = min(splits, len(encoded_labels))
        if effective_splits < 2:
            raise ValueError(
                self._("Cross-validation requires at least two total samples.")
            )

        if min_class_count >= 2:
            # Preserve class proportions whenever the smallest class supports
            # stratification; otherwise fall back to plain K-Fold.
            effective_splits = min(effective_splits, min_class_count)
            splitter = StratifiedKFold(
                n_splits=effective_splits,
                shuffle=shuffle,
                random_state=random_state,
            )
        else:
            splitter = KFold(
                n_splits=effective_splits,
                shuffle=shuffle,
                random_state=random_state,
            )
        # Why: a per-fold pipeline keeps scaler fitting inside the training
        # split only (no validation-set leakage).
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", self._build_cv_estimator()),
            ]
        )
        scores = cross_val_score(pipeline, X, encoded_labels, cv=splitter)
        return CrossValidationResult(scores=np.asarray(scores, dtype=float))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict original-state labels for new analog observations.

        Args:
            X (pandas.DataFrame): Inference feature matrix with the same
                columns (and order) as during :meth:`fit`; see
                :attr:`feature_columns`. Must not contain NaN.

        Returns:
            numpy.ndarray: Predicted labels mapped back to the original state
            identifiers, one per row of ``X``.

        Raises:
            sklearn.exceptions.NotFittedError: If the classifier has not been
                fitted or loaded.
        """
        scaled_features = self.scaler.transform(X)
        encoded_predictions = np.asarray(self.estimator.predict(scaled_features))
        if encoded_predictions.ndim > 1:
            encoded_predictions = encoded_predictions.argmax(axis=1)
        return self.label_encoder.inverse_transform(encoded_predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities aligned with the original label space.

        Args:
            X (pandas.DataFrame): Inference feature matrix (same contract as
                :meth:`predict`).

        Returns:
            numpy.ndarray: A ``float`` array of shape ``(n_samples, K)`` whose
            columns follow ``self.label_encoder.classes_``.

        Raises:
            AttributeError: If the estimator does not implement
                ``predict_proba`` (translated message).

        Notes:
            Features are scaled before the capability check, so an unfitted
            scaler raises first.
        """
        scaled_features = self.scaler.transform(X)
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(
                self._("The selected model does not expose probability estimates.")
            )
        return np.asarray(self.estimator.predict_proba(scaled_features), dtype=float)

    def save(self, file_path: str) -> None:
        """Persist the full inference artifact required for later reuse.

        The saved payload contains every object needed to run predictions on
        unseen data without retraining: estimator, scaler, label encoder, and
        feature ordering metadata.

        Args:
            file_path (str): Destination path (conventionally ``*.joblib``).
                The parent directory must already exist.

        Notes:
            The payload is a plain ``dict`` with keys ``model``, ``scaler``,
            ``label_encoder``, ``model_type``, ``params`` and
            ``feature_columns``; :meth:`load` relies on these keys.
            Artifacts are pickles: load them only from trusted sources.
        """
        payload = {
            "model": self.estimator,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "model_type": self.model_type,
            "params": self.params,
            "feature_columns": self.feature_columns,
        }
        joblib.dump(payload, file_path)

    @classmethod
    def load(
        cls,
        file_path: str,
        translator: Optional[Callable[[str], str]] = None,
    ) -> "StateClassifier":
        """Restore a persisted classifier artifact from disk.

        Args:
            file_path (str): Serialized artifact path created with
                :meth:`save`.
            translator (Callable[[str], str] | None): Optional translation
                function for user-facing errors.

        Returns:
            StateClassifier: A ready-to-use :class:`StateClassifier`
            instance.

        Raises:
            KeyError: If the payload lacks ``model_type``, ``model``,
                ``scaler`` or ``label_encoder``.

        Notes:
            A fresh instance is built first (creating a throw-away estimator
            from ``params``), then its fitted components are replaced by the
            stored ones. ``feature_columns`` defaults to ``[]`` for older
            artifacts that did not store it.
        """
        payload = joblib.load(file_path)
        classifier = cls(
            model_type=payload["model_type"],
            params=payload.get("params"),
            translator=translator,
        )
        classifier.estimator = payload["model"]
        classifier.scaler = payload["scaler"]
        classifier.label_encoder = payload["label_encoder"]
        classifier.feature_columns = payload.get("feature_columns", [])
        return classifier
