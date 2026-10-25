"""Model_A public API: supervised operating-state identification (``iabm``).

``iabm`` (Industrial Asset Behavioral Monitoring) is the first stage of the
multi-scale operational-evidence framework::

    Model_A (state inference) -> Model_B (behavioural sequences)
        -> Model_C (semantic modes) -> Model_D (incident evidence/episodes)

Model_A turns low-intrusion analog electrical telemetry (per-phase RMS
voltage, RMS current, power factor and active power) into a discrete
operating-state label per timestamp. Labels are learned in a supervised way
from a digital state stream (column ``estado``) that is only needed at
training/evaluation time; inference uses analog data alone.

The package re-exports the objects that downstream code is expected to use:

* :class:`~iabm.data_processor.IndustrialDataProcessor` -- loads Parquet
  inputs, imputes missing values, filters inactive rows and transfers digital
  labels to analog timestamps.
* :class:`~iabm.data_processor.TrainingDataset` and
  :class:`~iabm.data_processor.InferenceDataset` -- immutable containers
  returned by the processor.
* :class:`~iabm.models.StateClassifier` -- estimator lifecycle (scaling, label
  encoding, cross-validation, fitting, persistence, prediction).
* :class:`~iabm.models.CrossValidationResult` -- fold-wise CV scores.

:class:`~iabm.data_processor.EvaluationDataset` is intentionally not
re-exported here; import it from :mod:`iabm.data_processor` when needed.

.. mermaid::

    flowchart LR
        A[analog Parquet] --> P[IndustrialDataProcessor]
        D[digital Parquet<br/>column estado] -.training/evaluation only.-> P
        P --> T[TrainingDataset]
        P --> I[InferenceDataset]
        T --> C[StateClassifier]
        I --> C
        C --> J[(joblib artifact)]
        C --> S[predicted states<br/>to Model_B]

Maintenance notes:
    Keep ``__all__`` in sync with the imports above; it defines the stable
    surface consumed by Model_B and by the tests. Adding objects here is a
    public-API change and should be reflected in ``README.md``. Importing the
    package imports :mod:`iabm.models`, which requires ``xgboost``.
"""

from .data_processor import InferenceDataset, IndustrialDataProcessor, TrainingDataset
from .models import CrossValidationResult, StateClassifier

__all__ = [
    "CrossValidationResult",
    "InferenceDataset",
    "IndustrialDataProcessor",
    "StateClassifier",
    "TrainingDataset",
]
