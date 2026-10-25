"""CLI-level smoke tests for the Model_A package."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import iabm.main as cli_main


@pytest.mark.parametrize(("algorithm", "prediction_format"), [("rf", "xlsx"), ("xgb", "csv")])
def test_cli_train_and_predict_workflow(
    algorithm: str,
    prediction_format: str,
    synthetic_datasets: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The CLI should train, save, load, and predict for both supported models."""
    model_path = tmp_path / f"{algorithm}_model.joblib"
    report_dir = tmp_path / "reports"
    prediction_dir = tmp_path / "predictions"

    monkeypatch.setattr(
        "sys.argv",
        [
            "industrial-id",
            "--mode",
            "train",
            "--algo",
            algorithm,
            "--data-ana",
            str(synthetic_datasets["analog_path"]),
            "--data-dig",
            str(synthetic_datasets["digital_path"]),
            "--model-out",
            str(model_path),
            "--pred-out",
            str(report_dir),
            "--start",
            synthetic_datasets["start"],
            "--end",
            synthetic_datasets["train_end"],
            "--cv-splits",
            "3",
        ],
    )
    cli_main.main()

    assert model_path.exists()
    report_files = list(report_dir.glob("*.xlsx"))
    assert len(report_files) == 1

    monkeypatch.setattr(
        "sys.argv",
        [
            "industrial-id",
            "--mode",
            "predict",
            "--algo",
            algorithm,
            "--data-ana",
            str(synthetic_datasets["analog_path"]),
            "--model-out",
            str(model_path),
            "--pred-out",
            str(prediction_dir),
            "--start",
            synthetic_datasets["predict_start"],
            "--end",
            synthetic_datasets["predict_end"],
            "--include-inactive",
            "--pred-format",
            prediction_format,
        ],
    )
    cli_main.main()

    prediction_files = list(prediction_dir.glob(f"*.{prediction_format}"))
    assert len(prediction_files) == 1

    if prediction_format == "csv":
        prediction_frame = pd.read_csv(prediction_files[0])
    else:
        prediction_frame = pd.read_excel(prediction_files[0])
    assert "Predicted_State" in prediction_frame.columns
    assert not prediction_frame.empty


def test_cli_evaluate_workflow(
    synthetic_datasets: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The CLI should compare multiple saved models in a single evaluation run."""
    model_paths: list[Path] = []
    report_dir = tmp_path / "reports"

    for algorithm in ("rf", "xgb"):
        model_path = tmp_path / f"{algorithm}_model.joblib"
        model_paths.append(model_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "industrial-id",
                "--mode",
                "train",
                "--algo",
                algorithm,
                "--data-ana",
                str(synthetic_datasets["analog_path"]),
                "--data-dig",
                str(synthetic_datasets["digital_path"]),
                "--model-out",
                str(model_path),
                "--pred-out",
                str(report_dir),
                "--start",
                synthetic_datasets["start"],
                "--end",
                synthetic_datasets["train_end"],
                "--cv-splits",
                "3",
            ],
        )
        cli_main.main()

    evaluation_dir = tmp_path / "evaluation"
    monkeypatch.setattr(
        "sys.argv",
        [
            "industrial-id",
            "--mode",
            "evaluate",
            "--data-ana",
            str(synthetic_datasets["analog_path"]),
            "--data-dig",
            str(synthetic_datasets["digital_path"]),
            "--pred-out",
            str(evaluation_dir),
            "--model-paths",
            *(str(model_path) for model_path in model_paths),
            "--start",
            synthetic_datasets["start"],
            "--end",
            synthetic_datasets["predict_end"],
        ],
    )
    cli_main.main()

    evaluation_files = list(evaluation_dir.glob("evaluation_*.xlsx"))
    assert len(evaluation_files) == 1

    prediction_frame = pd.read_excel(evaluation_files[0], sheet_name="predictions")
    assert "Real_Class" in prediction_frame.columns
    assert "RF_Prediction" in prediction_frame.columns
    assert "XGB_Prediction" in prediction_frame.columns
    assert not prediction_frame.empty

    roc_files = list(evaluation_dir.glob("roc_*.png"))
    assert len(roc_files) == 2
    roc_pdf_files = list(evaluation_dir.glob("roc_*.pdf"))
    assert len(roc_pdf_files) == 2

    confusion_files = list(evaluation_dir.glob("cm_*.png"))
    assert len(confusion_files) == 2
    confusion_pdf_files = list(evaluation_dir.glob("cm_*.pdf"))
    assert len(confusion_pdf_files) == 2
