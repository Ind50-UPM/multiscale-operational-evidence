"""Generate candidate incident registries month by month and consolidate the outputs.

This utility exists to make large-scale incident-registry generation robust on
historical industrial campaigns where full-table in-memory runs may be too slow
or too fragile. It processes each ``source_month`` independently, persists a
monthly artifact, records traceability in a manifest, and finally produces a
consolidated flat-file registry consumed by downstream analysis steps.

The registry is a *weakly labelled* list of candidate incidents derived
heuristically from the telemetry itself by
:class:`Model_D.iabm_incidents.registry_builder.CandidateIncidentRegistryBuilder`
(with the default :class:`~Model_D.iabm_incidents.registry_builder.RegistryGenerationConfig`).
It is not independent incident ground truth; downstream studies use it as a
fixed weak reference.

Command line (from the repository root)::

    python src/generate_candidate_incident_registry_batch.py --output-dir src/predictions/Model_D

    # Resume a long campaign, reusing months already written:
    python src/generate_candidate_incident_registry_batch.py --skip-existing-months

    # Re-consolidate only from the distributed monthly parquet files. No raw
    # inputs are read when every listed month already exists; only the
    # listed months enter the consolidated registry, so list all of them:
    python src/generate_candidate_incident_registry_batch.py --skip-existing-months \\
        --months $(ls src/predictions/Model_D/monthly_candidate_registry | sed 's/.parquet//')

No ``PYTHONPATH`` is required: the script inserts its own folder (``src/``)
into ``sys.path`` and imports ``Model_D.iabm_incidents`` from there.

Inputs:
    * ``data/estados_nonans.parquet`` (``--states``): columns
      ``source_month``, ``date``, ``RP1``, ``RP2``, ``RP3``, ``RP4``,
      ``estado``, ``pred_estado``.
    * ``data/secuencias.parquet`` (``--sequences``): state-run sequence
      summaries with a ``source_month`` column (all columns are loaded).
    * ``data/palabras.parquet`` (``--words``): sequence-word summaries with a
      ``source_month`` column (all columns are loaded).

    Missing input files are tolerated (treated as empty), so a month present
    in only some of the inputs still appears in the manifest.

Outputs (in ``--output-dir``, default ``src/predictions/Model_D``; ``<stem>`` defaults to ``candidate_incident_registry_refined``):
    * ``monthly_candidate_registry/<YYYYMM>.parquet`` (and ``.csv`` with
      ``--keep-monthly-csv``): one registry per non-empty month.
    * ``<stem>_manifest.csv``: one row per processed month with ``status``
      (``written``, ``reused``, ``empty`` or ``failed``), row counts and the
      error message of failed months.
    * ``<stem>.csv``: consolidated registry sorted by ``event_time`` with
      sequential ``incident_id`` values ``AUTO-000001``, ``AUTO-000002``, ...
    * ``<stem>_summary.csv``: candidate counts per ``incident_family``,
      ``secondary_family`` and ``source_type``.
    * ``<stem>_metadata.json``: requested, written, empty and failed months
      and the output paths.

Script dependencies:
    Run after ``concat_monthly_csv_to_parquet.py`` (which builds the three
    Parquet inputs). ``<stem>.csv`` is required by ``run_model_d_ablation.py``,
    ``run_model_d_multiscale.py`` and ``run_model_d_extensions.py`` (fusion
    stage) as ``src/predictions/Model_D/candidate_incident_registry_refined.csv``.

.. mermaid::

    flowchart TD
        S[data/estados_nonans.parquet] --> L{for each source_month}
        Q[data/secuencias.parquet] --> L
        W[data/palabras.parquet] --> L
        L -->|reuse| R[monthly_candidate_registry/YYYYMM.parquet]
        L -->|build| B[CandidateIncidentRegistryBuilder.build_from_frames]
        B --> R
        L --> M[manifest CSV]
        R --> C[consolidate + renumber incident_id]
        C --> REG[candidate_incident_registry_refined.csv]
        C --> SUM[summary CSV]
        M --> META[metadata JSON]
        REG --> ABL[run_model_d_ablation.py / multiscale / extensions]

Maintenance notes:
    Months are processed independently with a fresh slice of each input, so
    peak memory is bounded by the largest month. Incident identifiers are
    reassigned after consolidation; they are therefore *not* stable across
    reruns that add or remove months and must not be used as long-lived
    keys. ``build_month_registry`` relies on the builder's private helper
    ``_empty_final_registry``; keep it in sync with the builder API. Empty
    months are recomputed (not reused) on every rerun because no monthly
    artifact is written for them. Versioning: ``.gitignore`` excludes
    ``src/predictions/**/*.csv`` while ``.gitattributes`` routes ``*.csv`` and
    ``*.parquet`` through Git LFS, so a regenerated CSV that should be
    distributed has to be added explicitly (``git add -f``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Note: make the sibling packages (Model_D, ...) importable without PYTHONPATH.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Model_D.iabm_incidents.registry_builder import CandidateIncidentRegistryBuilder, RegistryGenerationConfig


#: Default state-timeline parquet (relative to the repository root).
DEFAULT_STATES = "data/estados_nonans.parquet"
#: Default state-run sequence parquet.
DEFAULT_SEQUENCES = "data/secuencias.parquet"
#: Default sequence-word parquet.
DEFAULT_WORDS = "data/palabras.parquet"
#: Default destination of monthly and consolidated registries.
DEFAULT_OUTPUT_DIR = "src/predictions/Model_D"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for batch candidate-registry generation.

    Returns:
        argparse.Namespace: Attributes ``states``, ``sequences``, ``words``
        (str, input Parquet paths; defaults :data:`DEFAULT_STATES`,
        :data:`DEFAULT_SEQUENCES`, :data:`DEFAULT_WORDS`), ``output_dir``
        (str, default :data:`DEFAULT_OUTPUT_DIR`), ``months`` (list[str] or
        ``None``; explicit ``source_month`` values such as ``"202207"``),
        ``output_stem`` (str, base name of consolidated outputs, default
        ``"candidate_incident_registry_refined"``), ``keep_monthly_csv``,
        ``skip_existing_months`` and ``fail_fast`` (bool flags).

    Notes:
        The CLI is intentionally restart-friendly. ``--skip-existing-months``
        allows long campaigns to resume from already written monthly parquet
        artifacts without recomputing earlier months. ``--fail-fast`` stops
        the month loop at the first exception, but consolidation of the
        months completed so far still runs.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build candidate incident registries month by month from parquet inputs, "
            "persist partial outputs, and consolidate a final registry and summary."
        )
    )
    parser.add_argument('--states', default=DEFAULT_STATES, help=f'States parquet path. Default: {DEFAULT_STATES}')
    parser.add_argument('--sequences', default=DEFAULT_SEQUENCES, help=f'Sequences parquet path. Default: {DEFAULT_SEQUENCES}')
    parser.add_argument('--words', default=DEFAULT_WORDS, help=f'Words parquet path. Default: {DEFAULT_WORDS}')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help=f'Output directory. Default: {DEFAULT_OUTPUT_DIR}')
    parser.add_argument('--months', nargs='+', help='Optional explicit source_month values to process.')
    parser.add_argument('--output-stem', default='candidate_incident_registry_refined', help='Base name for consolidated outputs.')
    parser.add_argument('--keep-monthly-csv', action='store_true', help='Also write one CSV per month in addition to monthly parquet files.')
    parser.add_argument('--skip-existing-months', action='store_true', help='Reuse existing monthly parquet outputs when present.')
    parser.add_argument('--fail-fast', action='store_true', help='Stop at the first month that raises an exception.')
    return parser.parse_args()


def collect_months(states_path: Path, sequences_path: Path, words_path: Path) -> list[str]:
    """Collect every ``source_month`` available across the three parquet inputs.

    Args:
        states_path (Path): Path to the state-history parquet.
        sequences_path (Path): Path to the sequence-history parquet.
        words_path (Path): Path to the sequence-word parquet.

    Returns:
        list[str]: Sorted month identifiers (string form of
        ``source_month``, e.g. ``"202207"``) detected across the available
        sources; empty when none of the files exists.

    Notes:
        Missing files are skipped. Only the ``source_month`` column is read;
        an existing file without that column makes the Parquet engine raise.
        The union of all months is used so that installation stops or partially
        missing exports can still be represented in the manifest as empty months
        rather than silently disappearing from the orchestration layer.
    """
    months: set[str] = set()
    for parquet_path in [states_path, sequences_path, words_path]:
        if not parquet_path.is_file():
            continue
        frame = pd.read_parquet(parquet_path, columns=['source_month'])
        months.update(frame['source_month'].dropna().astype(str).unique().tolist())
    return sorted(months)


def load_month_frame(parquet_path: Path, month: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Load a single month slice from a parquet file.

    Args:
        parquet_path (Path): Parquet file containing a ``source_month``
            column.
        month (str): Month identifier to filter, compared with ``==``
            against ``source_month``; it must have the same type as the
            stored column (string ``YYYYMM`` for the pipeline inputs).
        columns (list[str] | None): Optional column projection to reduce
            I/O and memory; ``None`` loads all columns.

    Returns:
        pandas.DataFrame: A month-filtered dataframe, or an empty frame
        (without columns) if the parquet is absent.

    Notes:
        The filter is pushed down to the Parquet reader, so only the row
        groups of the requested month are materialised.
    """
    if not parquet_path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(parquet_path, columns=columns, filters=[('source_month', '==', month)])


def build_month_registry(
    builder: CandidateIncidentRegistryBuilder,
    *,
    month: str,
    states_path: Path,
    sequences_path: Path,
    words_path: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the weakly labeled registry for a single month.

    Args:
        builder (CandidateIncidentRegistryBuilder): Configured builder
            implementing the heuristic incident logic.
        month (str): Month identifier being processed (``YYYYMM``).
        states_path (Path): Source of state and power telemetry; only the
            columns ``source_month``, ``date``, ``RP1``--``RP4``, ``estado``
            and ``pred_estado`` are read.
        sequences_path (Path): Source of state-run sequence summaries (all
            columns read).
        words_path (Path): Source of sequence-word summaries (all columns
            read).

    Returns:
        tuple[pandas.DataFrame, dict[str, int]]: The month registry (the
        builder's empty registry schema when all three inputs are empty) and
        the input row counts under the keys ``states_rows``,
        ``sequences_rows`` and ``words_rows``.

    Notes:
        The returned counts are written to the manifest so operational users can
        later distinguish empty installations from failed computation.
        Exceptions raised by the builder propagate to the caller, which
        records the month as ``failed``.
    """
    states = load_month_frame(
        states_path,
        month,
        columns=['source_month', 'date', 'RP1', 'RP2', 'RP3', 'RP4', 'estado', 'pred_estado'],
    )
    sequences = load_month_frame(sequences_path, month)
    words = load_month_frame(words_path, month)

    counts = {
        'states_rows': int(len(states)),
        'sequences_rows': int(len(sequences)),
        'words_rows': int(len(words)),
    }
    if states.empty and sequences.empty and words.empty:
        return builder._empty_final_registry(), counts

    registry = builder.build_from_frames(states=states, sequences=sequences, words=words)
    return registry, counts


def consolidate_registries(monthly_paths: list[Path]) -> pd.DataFrame:
    """Concatenate monthly registry parquet artifacts into one flat dataframe.

    Args:
        monthly_paths (list[Path]): Paths to monthly parquet files written
            or reused during the batch; paths that are not files are ignored.

    Returns:
        pandas.DataFrame: A consolidated registry sorted by ``event_time``
        (when that column exists) and reindexed with a fresh, sequential
        ``incident_id`` namespace (``AUTO-000001`` onwards). When no monthly
        file exists, an empty frame with the canonical 20 registry columns.

    Notes:
        Incident identifiers are reassigned after concatenation so the final
        registry is deterministic regardless of whether some months were reused,
        skipped, or regenerated, provided the same set of months is
        consolidated. The sort is not declared stable, so ties in
        ``event_time`` may be ordered arbitrarily.
    """
    frames = [pd.read_parquet(path) for path in monthly_paths if path.is_file()]
    if not frames:
        return pd.DataFrame(
            columns=[
                'incident_id', 'source_window_start', 'source_window_end', 'documented_start', 'documented_end',
                'event_time', 'event_time_precision', 'incident_family', 'family', 'secondary_family',
                'recovery_time', 'recovery_status', 'label_strength', 'source_type', 'downtime_start',
                'downtime_end', 'maintenance_time', 'affected_subsystem', 'notes', 'asset_id',
            ]
        )
    full_registry = pd.concat(frames, ignore_index=True)
    if 'event_time' in full_registry.columns:
        full_registry = full_registry.sort_values('event_time').reset_index(drop=True)
    full_registry['incident_id'] = [f"AUTO-{index + 1:06d}" for index in range(len(full_registry))]
    return full_registry


def write_summary(registry: pd.DataFrame, summary_path: Path) -> None:
    """Write the final family/source summary for the consolidated registry.

    Args:
        registry (pandas.DataFrame): Consolidated candidate incident registry;
            must contain ``incident_family``, ``secondary_family`` and
            ``source_type``.
        summary_path (Path): Destination CSV path (overwritten).

    Returns:
        None: The summary (columns ``incident_family``, ``secondary_family``,
        ``source_type``, ``candidate_count``) is persisted to disk, sorted by
        decreasing count and then by family names. Missing family values are
        kept as their own group (``dropna=False``).

    Raises:
        KeyError: If one of the grouping columns is missing.
    """
    summary = (
        registry.groupby(['incident_family', 'secondary_family', 'source_type'], dropna=False)
        .size()
        .reset_index(name='candidate_count')
        .sort_values(['candidate_count', 'incident_family', 'secondary_family'], ascending=[False, True, True])
    )
    summary.to_csv(summary_path, index=False)


def main() -> int:
    """Execute the month-wise candidate-registry orchestration.

    Returns:
        int: Process-style exit code where ``0`` means successful completion
        and ``1`` indicates that no ``source_month`` was found or at least one
        month failed.

    Notes:
        The function is intentionally verbose in its artifact writing:
        monthly parquet files, a manifest CSV, a metadata JSON file, the
        consolidated registry, and the consolidated summary are all written so
        long historical campaigns remain auditable and restartable. Failures
        of individual months are caught, recorded in the manifest and do not
        prevent consolidation of the other months. Output directories are
        created before the month list is resolved.
    """
    args = parse_args()
    states_path = Path(args.states).resolve()
    sequences_path = Path(args.sequences).resolve()
    words_path = Path(args.words).resolve()
    output_dir = Path(args.output_dir).resolve()
    monthly_dir = output_dir / 'monthly_candidate_registry'
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly_dir.mkdir(parents=True, exist_ok=True)

    months = args.months or collect_months(states_path, sequences_path, words_path)
    if not months:
        print('No source_month values found in the provided parquet files.', file=sys.stderr)
        return 1

    builder = CandidateIncidentRegistryBuilder(RegistryGenerationConfig())
    monthly_paths: list[Path] = []
    manifest_rows: list[dict[str, object]] = []
    failures = 0

    for month in months:
        monthly_parquet = monthly_dir / f'{month}.parquet'
        monthly_csv = monthly_dir / f'{month}.csv'
        # Note: reused months record no input row counts (None) because the
        # raw inputs are not read at all for them.
        if args.skip_existing_months and monthly_parquet.is_file():
            existing = pd.read_parquet(monthly_parquet)
            monthly_paths.append(monthly_parquet)
            manifest_rows.append({
                'source_month': month,
                'status': 'reused',
                'registry_rows': int(len(existing)),
                'states_rows': None,
                'sequences_rows': None,
                'words_rows': None,
                'secondary_family_non_null': int(existing['secondary_family'].notna().sum()) if 'secondary_family' in existing.columns else 0,
                'error': None,
            })
            print(f'[{month}] reused {len(existing)} rows from {monthly_parquet}')
            continue

        try:
            registry, counts = build_month_registry(
                builder,
                month=month,
                states_path=states_path,
                sequences_path=sequences_path,
                words_path=words_path,
            )
            # Why: no artifact is written for empty months, so they are never
            # 'reused' and are recomputed on every rerun.
            if registry.empty:
                manifest_rows.append({
                    'source_month': month,
                    'status': 'empty',
                    'registry_rows': 0,
                    'states_rows': counts['states_rows'],
                    'sequences_rows': counts['sequences_rows'],
                    'words_rows': counts['words_rows'],
                    'secondary_family_non_null': 0,
                    'error': None,
                })
                print(f'[{month}] empty month or installation stop; skipped final monthly artifact')
                continue

            registry.to_parquet(monthly_parquet, index=False)
            if args.keep_monthly_csv:
                registry.to_csv(monthly_csv, index=False)
            monthly_paths.append(monthly_parquet)
            manifest_rows.append({
                'source_month': month,
                'status': 'written',
                'registry_rows': int(len(registry)),
                'states_rows': counts['states_rows'],
                'sequences_rows': counts['sequences_rows'],
                'words_rows': counts['words_rows'],
                'secondary_family_non_null': int(registry['secondary_family'].notna().sum()) if 'secondary_family' in registry.columns else 0,
                'error': None,
            })
            print(
                f'[{month}] wrote {len(registry)} rows '
                f'(secondary_family non-null: {int(registry["secondary_family"].notna().sum()) if "secondary_family" in registry.columns else 0})'
            )
        except Exception as exc:  # pragma: no cover - operational CLI safeguard
            failures += 1
            manifest_rows.append({
                'source_month': month,
                'status': 'failed',
                'registry_rows': None,
                'states_rows': None,
                'sequences_rows': None,
                'words_rows': None,
                'secondary_family_non_null': None,
                'error': f'{type(exc).__name__}: {exc}',
            })
            print(f'[{month}] failed: {type(exc).__name__}: {exc}', file=sys.stderr)
            if args.fail_fast:
                break

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / f'{args.output_stem}_manifest.csv'
    manifest.to_csv(manifest_path, index=False)

    successful_paths = [path for path in monthly_paths if path.is_file()]
    full_registry = consolidate_registries(successful_paths)
    registry_path = output_dir / f'{args.output_stem}.csv'
    summary_path = output_dir / f'{args.output_stem}_summary.csv'
    full_registry.to_csv(registry_path, index=False)
    write_summary(full_registry, summary_path)

    metadata = {
        'months_requested': months,
        'months_written': manifest.loc[manifest['status'].isin(['written', 'reused']), 'source_month'].astype(str).tolist(),
        'empty_months': manifest.loc[manifest['status'] == 'empty', 'source_month'].astype(str).tolist(),
        'failed_months': manifest.loc[manifest['status'] == 'failed', 'source_month'].astype(str).tolist(),
        'failure_count': failures,
        'final_registry_rows': int(len(full_registry)),
        'outputs': {
            'registry_csv': str(registry_path),
            'summary_csv': str(summary_path),
            'manifest_csv': str(manifest_path),
            'monthly_dir': str(monthly_dir),
        },
    }
    metadata_path = output_dir / f'{args.output_stem}_metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    print(f'Manifest saved to: {manifest_path}')
    print(f'Consolidated registry saved to: {registry_path}')
    print(f'Summary saved to: {summary_path}')
    print(f'Metadata saved to: {metadata_path}')
    if failures:
        print(f'Completed with {failures} failed months.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
