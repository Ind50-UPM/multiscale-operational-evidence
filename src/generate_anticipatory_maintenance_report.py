"""Generate the anticipatory-maintenance report from retained Model_D episode outputs.

Purpose:
    Fits and validates the horizon-specific risk forecaster
    :class:`iabm_incidents.anticipation.AnticipatoryMaintenanceForecaster` on
    the window scores and detected episodes of a previously executed Model_D
    episode evaluation (the "localized fallback" variant), and exports an
    HTML/CSV report with risk coefficients, validation metrics and suggested
    maintenance actions.

Command line (from the repository root)::

    python src/generate_anticipatory_maintenance_report.py --dataset probe
    python src/generate_anticipatory_maintenance_report.py --dataset full \\
        --horizons 24 72 168 --risk-threshold 0.55 --warning-threshold 0.35

No ``PYTHONPATH`` is needed: Python places the script folder ``src/`` on
``sys.path``, which makes ``Model_D.iabm_incidents`` importable as a
namespace package. Paths are relative, so the command must be launched from
the repository root.

Inputs (``<root>`` is ``src/predictions/episode_eval_probe`` or ``src/predictions/episode_eval_full``):
    * ``<root>/Model_D_episodes_localized_fallback/window_scores.csv``: scored
      windows; must contain ``start_time`` and ``end_time`` (parsed as
      dates) plus the score/family columns expected by the forecaster.
    * ``<root>/Model_D_episodes_localized_fallback/detected_episodes.csv``:
      detected episodes; must contain ``event_start`` and ``event_end``.

Outputs (``<root>/anticipation_report/``, written by :meth:`~iabm_incidents.anticipation.AnticipatoryMaintenanceForecaster.export_report`):
    ``feature_frame.csv``, ``scored_windows.csv``, ``validation_summary.csv``,
    ``horizon_coefficients.csv``, ``maintenance_actions.csv``,
    ``anticipation_report.html`` and ``anticipation_metadata.json``.

Script dependencies:
    Requires the retained Model_D episode evaluation folders under
    ``src/predictions/episode_eval_*`` (produced by the Model_D episode
    pipeline, ``iabm_incidents``; not by a top-level script of ``src/``).
    Independent of the ablation/multiscale/extensions chain.

.. mermaid::

    flowchart LR
        W[window_scores.csv] --> F[AnticipatoryMaintenanceForecaster.run]
        E[detected_episodes.csv] --> F
        F --> R[anticipation_report/*.csv + HTML + JSON]

Maintenance notes:
    Only the ``Model_D_episodes_localized_fallback`` variant is read; the
    baseline variant is ignored. ``--horizons`` accepts zero or more integers
    (hours); passing the flag with no values yields an empty horizon tuple,
    which the forecaster is not designed for. Existing report files are
    overwritten. In the public release the large ``window_scores.csv`` of the
    ``full`` evaluation is not distributed, so ``--dataset full`` only works
    after regenerating that file locally; ``--dataset probe`` works with the
    distributed files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from Model_D.iabm_incidents import AnticipationConfig, AnticipatoryMaintenanceForecaster

#: Retained Model_D evaluation roots selectable with ``--dataset``.
DATASET_ROOTS = {
    "probe": Path("src/predictions/episode_eval_probe"),
    "full": Path("src/predictions/episode_eval_full"),
}


def parse_args() -> argparse.Namespace:
    """Parse the command-line options of the anticipation report.

    Returns:
        argparse.Namespace: Attributes ``dataset`` (``"full"`` or
        ``"probe"``, default ``"probe"``), ``horizons`` (list[int],
        forecast horizons in hours, default ``[24, 72, 168]``),
        ``risk_threshold`` (float in [0, 1], default ``0.55``) and
        ``warning_threshold`` (float in [0, 1], default ``0.35``).

    Notes:
        In the forecaster, a window whose maximum predicted risk over all
        horizons is ``>= risk_threshold`` gets high priority
        (``inspect_and_prepare_intervention``); ``>= warning_threshold``
        gets medium priority (``increase_monitoring_and_schedule_check``);
        otherwise low priority (``routine_follow_up``). Keep
        ``warning_threshold < risk_threshold``.
    """
    parser = argparse.ArgumentParser(description="Generate anticipatory maintenance report from retained Model_D outputs.")
    parser.add_argument("--dataset", choices=sorted(DATASET_ROOTS), default="probe")
    parser.add_argument("--horizons", nargs="*", type=int, default=[24, 72, 168])
    parser.add_argument("--risk-threshold", type=float, default=0.55)
    parser.add_argument("--warning-threshold", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    """Load retained episode outputs, run the forecaster and export the report.

    Raises:
        FileNotFoundError: If ``window_scores.csv`` or
            ``detected_episodes.csv`` is missing for the selected dataset.

    Notes:
        Writes to ``<root>/anticipation_report/`` (see module docstring) and
        prints the output folder.
    """
    args = parse_args()
    root = DATASET_ROOTS[args.dataset]
    window_scores = pd.read_csv(root / "Model_D_episodes_localized_fallback" / "window_scores.csv", parse_dates=["start_time", "end_time"])
    episodes = pd.read_csv(root / "Model_D_episodes_localized_fallback" / "detected_episodes.csv", parse_dates=["event_start", "event_end"])
    forecaster = AnticipatoryMaintenanceForecaster(
        AnticipationConfig(
            horizons_hours=tuple(args.horizons),
            risk_threshold=args.risk_threshold,
            warning_threshold=args.warning_threshold,
        )
    )
    result = forecaster.run(window_scores=window_scores, episodes=episodes)
    output_dir = root / "anticipation_report"
    forecaster.export_report(result, output_dir)
    print(f"Anticipation report written to: {output_dir}")


if __name__ == "__main__":
    main()
