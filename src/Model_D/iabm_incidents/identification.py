"""Fault-type identification from the shape operator of fourth-order evidence.

Role in the pipeline
--------------------
The transient score :math:`T^{(4)}_w` and the kurtosis ratio tell *how much*
the transient organisation of the command-conditioned residuals changed. The
shape change :math:`\\Delta\\rho = \\rho_w-\\rho_0` of the residual density
operator (see :mod:`iabm_incidents.quantum`) tells *in which direction* it
changed. This module evaluates whether that direction identifies the type of
fault, using the labelled windows of the fault-injection benchmark produced by
``run_model_d_extensions.py --stage telemetry``.

Two identification tasks are defined on held-out evaluation windows:

* ``family``: clean, under-response, power dropouts or additive bursts, on the
  moderate scenarios listed in :data:`FAMILY_SCENARIOS`;
* ``phase``: whether an under-response affects one phase or all three.

Feature sets are compared with grouped cross-validation, where all scenarios
injected into the same calendar window share a group, so that a window never
appears in both the training and the test folds.

.. mermaid::

    flowchart LR
        W["injection_windows.parquet<br/>(scalars + Pauli signature)"] --> T["tasks: family / phase"]
        T --> F["feature sets: scalars, + Pauli, + matrix entries, Pauli only"]
        F --> C["grouped K-fold logistic regression"]
        C --> S["macro-F1 / accuracy"]
        W --> G["mean Pauli shift per scenario vs clean"]

Main public objects
-------------------
* :data:`SCALAR_FEATURES`, :data:`FAMILY_SCENARIOS`.
* :class:`IdentificationConfig`.
* :func:`identification_tasks`, :func:`matrix_entries`,
  :func:`grouped_identification`, :func:`pauli_signatures`.

Data contract
-------------
The input frame has one row per (window, scenario) with ``start_time``,
``split`` (``threshold``/``evaluation``), ``scenario`` (``clean`` or a label
``kind|m=..|rate=..|len=..s|phases=..``), the scalar statistics of
:data:`iabm_incidents.injection.STATISTICS` and the
:data:`iabm_incidents.injection.SHAPE_COLUMNS`.

Maintenance notes
-----------------
* The Pauli coefficients are a linear reparametrisation of the entries of
  :math:`\\Delta\\rho`; :func:`matrix_entries` exists to check that the gain
  comes from the operator and not from the basis.
* scikit-learn is imported lazily so that the package does not depend on it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .injection import SHAPE_COLUMNS, STATISTICS
from .quantum import PAULI

#: Scalar statistics used as the baseline feature set.
SCALAR_FEATURES = STATISTICS

#: Moderate scenarios of the ``family`` task (labels as written by the
#: telemetry stage). Strong faults are excluded because every feature set
#: separates them trivially.
FAMILY_SCENARIOS = (
    "clean",
    "under_response|m=0.05|rate=0|len=0s|phases=1",
    "under_response|m=0.05|rate=0|len=0s|phases=3",
    "dropouts|m=1|rate=2|len=1s|phases=3",
    "dropouts|m=1|rate=6|len=1s|phases=3",
    "bursts|m=12|rate=30|len=3s|phases=3",
    "bursts|m=6|rate=30|len=3s|phases=3",
)


@dataclass(frozen=True)
class IdentificationConfig:
    """Cross-validation settings of the identification study.

    Attributes:
        folds: Number of grouped folds (``>= 2``).
        regularisation: Inverse L2 strength ``C`` of the logistic regression.
        max_iter: Iteration cap of the solver.
    """

    folds: int = 5
    regularisation: float = 1.0
    max_iter: int = 3000

    def __post_init__(self) -> None:
        if self.folds < 2 or self.regularisation <= 0 or self.max_iter < 1:
            raise ValueError("Invalid identification configuration")


def _signed_log(values: np.ndarray) -> np.ndarray:
    """Return ``sign(x) log(1+|x|)``, which tames the heavy tails of the scores."""
    return np.sign(values)*np.log1p(np.abs(values))


def identification_tasks(windows: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
    """Build the ``family`` and ``phase`` tasks from the injection windows.

    Args:
        windows: Injection-benchmark frame (see the module data contract).

    Returns:
        dict: Task name to ``(rows, labels, groups)``; ``groups`` are integer
        window keys derived from ``start_time``.
    """
    evaluation = windows[windows.split.eq("evaluation")].dropna(subset=list(SCALAR_FEATURES))
    family = evaluation[evaluation.scenario.isin(FAMILY_SCENARIOS)]
    phase = evaluation[evaluation.scenario.str.startswith("under_response")]
    key = lambda frame: pd.to_datetime(frame.start_time).astype("int64").to_numpy()
    return {
        "family": (family, family.scenario.str.split("|").str[0].to_numpy(), key(family)),
        "phase": (phase, phase.scenario.str.extract(r"phases=(\d)")[0].map({"1": "one phase", "3": "three phases"}).to_numpy(), key(phase)),
    }


def matrix_entries(frame: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the upper-triangular entries of :math:`\\Delta\\rho` from its Pauli signature.

    Args:
        frame: Rows holding the :data:`~iabm_incidents.injection.SHAPE_COLUMNS`.

    Returns:
        pandas.DataFrame: 36 columns ``m_ij`` (``i <= j``) with the same index.
    """
    labels = [c[len("pauli_"):] for c in SHAPE_COLUMNS]
    basis = np.stack([np.real(np.kron(np.kron(PAULI[l[0]], PAULI[l[1]]), PAULI[l[2]])) for l in labels])
    matrices = np.einsum("nk,kij->nij", frame[list(SHAPE_COLUMNS)].fillna(0).to_numpy(float), basis)
    upper = np.triu_indices(8)
    return pd.DataFrame(matrices[:, upper[0], upper[1]], index=frame.index,
                        columns=[f"m_{i}{j}" for i, j in zip(*upper)])


def grouped_identification(rows: pd.DataFrame, labels: np.ndarray, groups: np.ndarray,
                           feature_sets: dict[str, pd.DataFrame],
                           config: IdentificationConfig = IdentificationConfig()) -> pd.DataFrame:
    """Score feature sets with grouped K-fold logistic regression.

    Args:
        rows: Task rows (only their index is used to align features).
        labels: Class label per row.
        groups: Group key per row; rows sharing a key never split across folds.
        feature_sets: Name to feature frame aligned with ``rows``.
        config: Cross-validation settings.

    Returns:
        pandas.DataFrame: One row per feature set with ``features``,
        ``n_features``, ``macro_f1`` and ``accuracy`` of the out-of-fold
        predictions.

    Notes:
        Features are signed-log transformed and standardised inside each
        training fold, so no information from the test fold leaks into the
        scaler.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    results = []
    for name, features in feature_sets.items():
        x = _signed_log(features.loc[rows.index].fillna(0).to_numpy(float))
        predicted = np.empty(len(labels), dtype=object)
        for train, test in GroupKFold(config.folds).split(x, labels, groups):
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=config.max_iter, C=config.regularisation))
            model.fit(x[train], labels[train])
            predicted[test] = model.predict(x[test])
        results.append(dict(features=name, n_features=x.shape[1],
                            macro_f1=float(f1_score(labels, predicted, average="macro")),
                            accuracy=float((predicted == labels).mean())))
    return pd.DataFrame(results)


def pauli_signatures(windows: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    """Mean Pauli shift of each injected scenario relative to clean windows.

    Args:
        windows: Injection-benchmark frame.
        top: Number of largest-magnitude terms kept per scenario.

    Returns:
        pandas.DataFrame: Rows ``scenario``, ``rank``, ``pauli``, ``shift``
        (mean coefficient minus the clean mean) and ``clean_sd`` (standard
        deviation over clean evaluation windows, to judge the size of the
        shift).
    """
    evaluation = windows[windows.split.eq("evaluation")]
    clean = evaluation[evaluation.scenario.eq("clean")][list(SHAPE_COLUMNS)]
    rows = []
    for scenario, part in evaluation[evaluation.scenario.ne("clean")].groupby("scenario"):
        shift = part[list(SHAPE_COLUMNS)].mean()-clean.mean()
        for rank, column in enumerate(shift.abs().sort_values(ascending=False).index[:top], start=1):
            rows.append(dict(scenario=scenario, rank=rank, pauli=column[len("pauli_"):],
                             shift=float(shift[column]), clean_sd=float(clean[column].std())))
    return pd.DataFrame(rows)
