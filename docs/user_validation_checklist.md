# User Validation Checklist

Project: `industrial-asset-behavioral-monitoring`  
Date: `_____________`  
Tester: `_____________`

## 1. Model_A Train Random Forest

VS Code configuration: `Model_A: Train Random Forest`

Result:
- `OK / FAIL`
- Observations: `________________________________________`

Validate:
- the execution starts without errors
- the expected time window is processed
- cross-validation metrics are printed
- the `.joblib` model artifact is saved
- the metrics Excel report is generated in `src/predictions/Model_A`

Expected artifacts:
- `src/models/Model_A/analogicas_nonans.parquet_RF.joblib`
- metrics Excel report in `src/predictions/Model_A`

## 2. Model_A Predict Random Forest

VS Code configuration: `Model_A: Predict Random Forest`

Result:
- `OK / FAIL`
- Observations: `________________________________________`

Validate:
- the RF model loads correctly
- no digital labels are required for inference
- the prediction file is generated
- the Excel columns and timestamps are readable

Expected artifact:
- `src/predictions/Model_A/predictions_analogicas_nonans.parquet_RF.xlsx`

## 3. Model_A Train XGBoost

VS Code configuration: `Model_A: Train XGBoost`

Result:
- `OK / FAIL`
- Observations: `________________________________________`

Validate:
- the workflow does not fail because of `xgboost` / `scikit-learn` integration
- cross-validation folds are printed
- the `.joblib` artifact is saved
- the metrics Excel report is generated

Expected artifacts:
- `src/models/Model_A/analogicas_nonans.parquet_XGB.joblib`
- metrics Excel report in `src/predictions/Model_A`

## 4. Model_A Predict XGBoost

VS Code configuration: `Model_A: Predict XGBoost`

Result:
- `OK / FAIL`
- Observations: `________________________________________`

Validate:
- the XGBoost model loads correctly
- predictions are generated without errors
- the final Excel file is created in the expected output path

Expected artifact:
- `src/predictions/Model_A/predictions_analogicas_nonans.parquet_XGB.xlsx`

## 5. Model_B Sequence Analysis

VS Code configuration: `Model_B: Sequence Analysis`

Result:
- `OK / FAIL`
- Observations: `________________________________________`

Validate:
- the configuration accepts Model_A output
- the state-run report is generated
- the active-sequence report is generated
- the sequence-word summary is generated

Expected artifacts:
- `src/predictions/Model_B/state_runs.xlsx`
- `src/predictions/Model_B/active_sequences.xlsx`
- `src/predictions/Model_B/sequence_words.xlsx`

## 6. Model_B Compare Against Nominal

VS Code configuration: `Model_B: Compare Against Nominal`

Result:
- `OK / FAIL`
- Observations: `________________________________________`

Validate:
- the configuration accepts both observed and nominal inputs
- `sequence_comparison.xlsx` is generated
- distance metrics and `anomaly_score` columns are present
- the configured threshold produces reasonable results

Expected artifact:
- `src/predictions/Model_B/sequence_comparison.xlsx`

## 7. Model_C Semantic Interpretation

VS Code configuration: `Model_C: Semantic Interpretation`

Result:
- `OK / FAIL`
- Observations: `________________________________________`

Validate:
- the configuration accepts `active_sequences.xlsx`
- semantic assignments are generated
- the semantic summary is generated
- mode naming is understandable

Expected artifacts:
- `src/predictions/Model_C/semantic_assignments.xlsx`
- `src/predictions/Model_C/semantic_mode_summary.xlsx`

## 8. Model_C Semantic Interpretation With Comparison

VS Code configuration: `Model_C: Semantic Interpretation With Comparison`

Result:
- `OK / FAIL`
- Observations: `________________________________________`

Validate:
- the configuration correctly integrates the Model_B comparison report
- anomaly-enriched semantic labels are produced
- the aggregated summary remains coherent

Expected artifacts:
- `src/predictions/Model_C/semantic_assignments.xlsx`
- `src/predictions/Model_C/semantic_mode_summary.xlsx`

## Acceptance Summary

- `Model_A RF workflow`: `OK / FAIL`
- `Model_A XGB workflow`: `OK / FAIL`
- `Model_B workflow`: `OK / FAIL`
- `Model_C workflow`: `OK / FAIL`
- `A -> B -> C end-to-end traceability`: `OK / FAIL`

## Issues Found

- `1. ________________________________________`
- `2. ________________________________________`
- `3. ________________________________________`

## Overall Decision

- `Accepted for user testing`
- `Accepted with minor fixes`
- `Needs correction before testing`
