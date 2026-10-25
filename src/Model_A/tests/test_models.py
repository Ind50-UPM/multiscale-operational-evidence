"""Unit tests for the Model_A classifier abstractions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from iabm.models import StateClassifier


def test_state_classifier_save_load_roundtrip_rf(tmp_path) -> None:
    """A persisted Random Forest classifier should load and predict identically."""
    X = pd.DataFrame(
        {
            "Vrms1": [220.0, 221.0, 230.0, 231.0],
            "Vrms2": [221.0, 222.0, 231.0, 232.0],
            "Vrms3": [222.0, 223.0, 232.0, 233.0],
            "Irms1": [5.0, 5.1, 8.0, 8.1],
            "Irms2": [5.2, 5.3, 8.2, 8.3],
            "Irms3": [5.4, 5.5, 8.4, 8.5],
            "PF1": [0.80, 0.81, 0.92, 0.93],
            "PF2": [0.81, 0.82, 0.93, 0.94],
            "PF3": [0.82, 0.83, 0.94, 0.95],
        }
    )
    y = pd.Series([1, 1, 4, 4])

    classifier = StateClassifier(model_type="rf")
    classifier.fit(X, y)
    expected = classifier.predict(X)

    artifact_path = tmp_path / "rf_model.joblib"
    classifier.save(str(artifact_path))
    restored = StateClassifier.load(str(artifact_path))

    assert restored.model_type == "rf"
    assert restored.feature_columns == list(X.columns)
    np.testing.assert_array_equal(restored.predict(X), expected)


def test_xgb_cross_validate_handles_singleton_classes() -> None:
    """XGBoost cross-validation should not fail when one class appears once."""
    X = pd.DataFrame(
        {
            "Vrms1": np.linspace(220.0, 235.0, 10),
            "Vrms2": np.linspace(221.0, 236.0, 10),
            "Vrms3": np.linspace(222.0, 237.0, 10),
            "Irms1": np.linspace(5.0, 9.0, 10),
            "Irms2": np.linspace(5.2, 9.2, 10),
            "Irms3": np.linspace(5.4, 9.4, 10),
            "PF1": np.linspace(0.80, 0.95, 10),
            "PF2": np.linspace(0.81, 0.96, 10),
            "PF3": np.linspace(0.82, 0.97, 10),
        }
    )
    y = pd.Series([1, 1, 2, 2, 4, 4, 8, 8, 16, 32])

    classifier = StateClassifier(model_type="xgb")
    result = classifier.cross_validate(X, y, splits=5)

    assert len(result.scores) == 5
    assert np.isfinite(result.scores).all()
