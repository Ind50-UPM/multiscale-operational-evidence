"""Command-line entry point for Model_C semantic interpretation.

The CLI converts behavioral outputs from ``Model_B`` into semantic and
reliability-oriented labels that downstream incident processing can consume.
It supports optional anomaly-context enrichment through Model_B comparison
reports and keeps reporting concerns isolated from the interpreter itself.

The module is exposed as the ``industrial-semantics`` console script (see
``pyproject.toml``) and can also be run with ``python -m iabm_semantics.main``.

Data contract:

* ``--input``: Model_B ``active_sequences`` report (``.parquet``,
  ``.xlsx``/``.xls`` or ``.csv``) with at least a ``states`` column holding
  state words, typically serialised as tuple text such as ``"(1, 2)"``.
* ``--comparison-input`` (optional): Model_B ``sequence_comparison`` report
  with ``anomaly_score`` and ``is_anomalous`` columns, row-aligned with the
  sequences report.
* ``--rules`` (optional): JSON list of objects with keys
  ``required_components`` (list of component names), ``operating_mode`` and
  ``working_mode``.
* Outputs (``<name>.<xlsx|csv>`` in ``--output-dir``):
  ``semantic_assignments``, ``semantic_mode_summary``,
  ``incident_family_assignments`` (same content as ``semantic_assignments``)
  and ``asset_life_regime_summary``.

.. mermaid::

   flowchart TD
       A[_detect_language] --> B[setup_i18n]
       B --> C[parse_arguments]
       C --> D[SemanticModeInterpreter]
       C -->|--rules| E[load_rules]
       E --> D
       D --> F[load_active_sequences]
       C -->|--comparison-input| G[load_comparison_report]
       F --> H[interpret_sequences]
       G --> H
       H --> I[summarize_modes]
       H --> J[summarize_life_regimes]
       H & I & J --> K[_write_report]

Maintenance notes:
    The language is sniffed from ``sys.argv`` before argparse runs so that
    help strings can be translated. Console messages passed to
    ``translator(...)`` are ``.po`` msgids; changing their text breaks the
    translations. Output file names are consumed by Model_D and
    ``src/generate_model_bc_batch.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

from .semantics import SemanticModeInterpreter
from .utils import setup_i18n


def parse_arguments(translator: Callable[[str], str]) -> argparse.Namespace:
    """Build the CLI parser for semantic interpretation.

    Options:

    * ``--input`` (required): Model_B active-sequence report.
    * ``--comparison-input``: optional Model_B comparison report.
    * ``--rules``: optional JSON rules file.
    * ``--output-dir`` (required): report directory (created if missing).
    * ``--lang`` (``en`` | ``es``, default ``en``): interface language.
    * ``--output-format`` / ``--output_format`` (``xlsx`` | ``csv``, default
      ``xlsx``).

    Args:
        translator: Translation function used for localized help strings.

    Returns:
        Parsed command-line arguments for input selection, optional custom rule
        injection, and output formatting.

    Raises:
        SystemExit: Raised by :mod:`argparse` on invalid arguments or when
            ``--help`` is requested.

    Notes:
        Arguments are read from ``sys.argv``.
    """
    _ = translator
    parser = argparse.ArgumentParser(
        description=_("Semantic interpretation of industrial behavioral sequences")
    )
    parser.add_argument("--input", required=True, help=_("Path to the Model_B active-sequence report."))
    parser.add_argument("--comparison-input", help=_("Optional Model_B comparison report used to enrich semantic status."))
    parser.add_argument("--rules", help=_("Optional JSON file with semantic interpretation rules."))
    parser.add_argument("--output-dir", required=True, help=_("Directory where Model_C reports will be written."))
    parser.add_argument("--lang", default="en", choices=["es", "en"], help=_("Interface language."))
    # Note: both spellings map to ``args.output_format`` for backward compatibility.
    parser.add_argument("--output-format", "--output_format", dest="output_format", choices=["xlsx", "csv"], default="xlsx", help=_("Report export format."))
    return parser.parse_args()


def main() -> None:
    """Run semantic interpretation and persist the resulting reports.

    Returns:
        ``None``. Output artefacts are written into the selected directory.

    Raises:
        SystemExit: On invalid command-line arguments.
        ValueError: Propagated from
            :meth:`~iabm_semantics.semantics.SemanticModeInterpreter.load_active_sequences`
            for unsupported file extensions.
        KeyError: Propagated when the rules JSON or the input reports lack
            required keys/columns.

    Notes:
        The CLI emits both a row-level assignment table and multiple summary
        views because downstream layers often require different semantic slices:
        one for per-sequence incident-family alignment and another for broader
        operating-mode and life-regime analysis.
    """
    lang = _detect_language(sys.argv)
    translator = setup_i18n(lang)
    args = parse_arguments(translator)

    interpreter = SemanticModeInterpreter()
    if args.rules:
        interpreter.load_rules(args.rules)

    sequences = interpreter.load_active_sequences(args.input)
    comparison = interpreter.load_comparison_report(args.comparison_input) if args.comparison_input else None

    assignments = interpreter.interpret_sequences(sequences, comparison=comparison)
    summary = interpreter.summarize_modes(assignments)
    life_regime_summary = interpreter.summarize_life_regimes(assignments)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.output_format
    assignments_path = output_dir / f"semantic_assignments.{suffix}"
    summary_path = output_dir / f"semantic_mode_summary.{suffix}"
    family_path = output_dir / f"incident_family_assignments.{suffix}"
    life_regime_summary_path = output_dir / f"asset_life_regime_summary.{suffix}"
    _write_report(assignments, assignments_path)
    _write_report(summary, summary_path)
    # Note: intentional alias export of the same row-level table for Model_D.
    _write_report(assignments, family_path)
    _write_report(life_regime_summary, life_regime_summary_path)

    print(translator("Semantic assignments saved to: {}").format(assignments_path))
    print(translator("Semantic mode summary saved to: {}").format(summary_path))
    print(translator("Incident family assignments saved to: {}").format(family_path))
    print(translator("Asset life regime summary saved to: {}").format(life_regime_summary_path))


def _write_report(frame: pd.DataFrame, path: Path) -> None:
    """Persist one semantic report in CSV or Excel form.

    Args:
        frame: Semantic dataframe to export. The index is not written.
        path: Output path whose suffix selects the persistence backend:
            ``.csv`` (case-insensitive) uses :meth:`pandas.DataFrame.to_csv`;
            any other suffix uses :meth:`pandas.DataFrame.to_excel` (requires
            ``openpyxl`` for ``.xlsx``).

    Returns:
        ``None``. The dataframe is written to disk.
    """
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
        return
    frame.to_excel(path, index=False)


def _detect_language(argv: list[str]) -> str:
    """Infer the CLI language from the raw command line.

    Only the two-token form ``--lang <code>`` is recognised; the value is
    returned without validation (argparse validates it later).

    Args:
        argv: Raw command-line argument list (typically ``sys.argv``).

    Returns:
        Requested language code or ``"en"`` when no explicit language was
        provided (or ``--lang`` is the last token).

    Notes:
        The ``--lang=es`` spelling is accepted by argparse but is not detected
        here, so help strings stay in English in that case.
    """
    if "--lang" in argv:
        try:
            return argv[argv.index("--lang") + 1]
        except (IndexError, ValueError):
            return "en"
    return "en"


if __name__ == "__main__":
    main()
