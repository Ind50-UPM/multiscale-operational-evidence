"""Run Model_B and Model_C month by month from estados_nonans parquet and consolidate outputs.

Purpose:
    Batch orchestration of the two intermediate layers of the framework:

    * **Model_B** (:mod:`iabm_behavior`): the per-sample state timeline of a
      month is turned into contiguous state runs, *active sequences*
      (consecutive non-idle runs) and sequence-word summaries, and every
      active sequence is compared with a cumulative nominal reference built
      from all earlier months.
    * **Model_C** (:mod:`iabm_semantics`): every active sequence is
      interpreted into operating/working modes, a semantic status
      (``NORMAL``/``ANOMALOUS``), an incident family, a life regime and a
      recovery regime.

    Processing month by month keeps memory bounded and lets long campaigns be
    resumed with ``--skip-existing-months``.

Command line (from the repository root)::

    python src/generate_model_bc_batch.py --output-dir src/predictions

    # Resume, reusing months whose sequences and assignments already exist:
    python src/generate_model_bc_batch.py --output-dir src/predictions --skip-existing-months

    # Only some months, with a stricter anomaly threshold:
    python src/generate_model_bc_batch.py --months 202207 202208 --anomaly-threshold 1.5

No ``PYTHONPATH`` is required: the script inserts its own folder (``src/``)
into ``sys.path`` and imports ``Model_B.iabm_behavior`` and
``Model_C.iabm_semantics`` from there.

Inputs:
    ``data/estados_nonans.parquet`` (``--states``) with the columns
    ``source_month``, ``date`` and the state column selected by
    ``--state-column`` (default ``pred_estado``, the retained Model_A
    prediction). ``date`` values are parsed with mixed formats, converted to
    UTC and made timezone-naive.

Outputs (``<out>`` is ``--output-dir``, default ``src/predictions``):
    Monthly (``<out>/Model_B/monthly/`` and ``<out>/Model_C/monthly/``):

    * ``<YYYYMM>_state_runs.csv``, ``<YYYYMM>_active_sequences.csv``,
      ``<YYYYMM>_sequence_words.csv`` and, from the second processed month
      on, ``<YYYYMM>_sequence_comparison.csv`` (Model_B);
    * ``<YYYYMM>_semantic_assignments.csv``,
      ``<YYYYMM>_incident_family_assignments.csv`` (an identical copy of the
      semantic assignments) and ``<YYYYMM>_semantic_mode_summary.csv``
      (Model_C).

    Consolidated:

    * ``<out>/Model_B/active_sequences.csv`` and
      ``<out>/Model_B/sequence_comparison.csv``;
    * ``<out>/Model_C/semantic_assignments.csv``,
      ``<out>/Model_C/incident_family_assignments.csv`` and
      ``<out>/Model_C/semantic_mode_summary.csv`` (counts per
      ``operating_mode``, ``working_mode``, ``semantic_status``,
      ``incident_family``, ``life_regime``);
    * ``<out>/model_bc_batch_manifest.csv`` (one row per month with status
      ``written``, ``reused``, ``empty``, ``no_active_sequences`` or
      ``failed``) and ``<out>/model_bc_batch_metadata.json``.

Script dependencies:
    Run after ``concat_monthly_csv_to_parquet.py`` (which builds
    ``data/estados_nonans.parquet``). Its consolidated Model_B/Model_C CSVs
    are the inputs of the Model_D episode pipeline (``iabm_incidents``
    command line, ``--sequences``/``--assignments``). It is independent of
    ``generate_candidate_incident_registry_batch.py``.

.. mermaid::

    flowchart TD
        S[data/estados_nonans.parquet] --> T[load_month_timeline]
        T --> R[BehavioralSequenceAnalyzer: runs, active sequences, words]
        N[(cumulative nominal lookup of earlier months)] --> F[fast_compare_to_nominal]
        R --> F
        R --> C[SemanticModeInterpreter.interpret_sequences]
        F --> C
        C --> MC[Model_C monthly CSVs]
        R --> MB[Model_B monthly CSVs]
        R -->|append after month| N
        MB --> CONS[consolidated CSVs + manifest + metadata]
        MC --> CONS

Maintenance notes:
    *Chronological boundary:* each month is compared only against sequences
    of months processed *before* it, in the order of ``--months`` (or sorted
    ``source_month`` order), so the first processed month has no comparison
    and every sequence of it is ``NORMAL``. Every earlier sequence, including
    anomalous ones, enters the nominal reference. The comparison is a fast
    approximation implemented here (:func:`fast_compare_to_nominal`), not the
    full Model_B DTW comparison. Reused months still feed the nominal
    reference, so a resumed run reproduces the original comparisons only if
    the reused months are the same. Consolidation concatenates the monthly
    CSV files, so stale monthly files from earlier runs are included if they
    are reused.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd

# Note: make the sibling packages (Model_B, Model_C) importable without PYTHONPATH.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Model_B.iabm_behavior.sequences import ActiveSequence, BehavioralSequenceAnalyzer
from Model_C.iabm_semantics.semantics import SemanticModeInterpreter

#: Default state-timeline parquet (relative to the repository root).
DEFAULT_STATES = "data/estados_nonans.parquet"
#: Default base output directory for Model_B/Model_C artifacts.
DEFAULT_OUTPUT_DIR = "src/predictions"


def parse_args() -> argparse.Namespace:
    """Parse the command-line options of the Model_B/Model_C batch.

    Returns:
        argparse.Namespace: Attributes ``states`` (str, input parquet path,
        default :data:`DEFAULT_STATES`), ``output_dir`` (str, default
        :data:`DEFAULT_OUTPUT_DIR`), ``months`` (list[str] or ``None``,
        explicit ``source_month`` values processed in the given order),
        ``state_column`` (str, state column to analyse, default
        ``"pred_estado"``), ``anomaly_threshold`` (float, a sequence is
        anomalous when its anomaly score is ``>=`` this value, default
        ``1.0``), ``skip_existing_months`` and ``fail_fast`` (bool flags).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate Model_B and Model_C outputs source_month by source_month, "
            "persist monthly artifacts, and consolidate final reports."
        )
    )
    parser.add_argument('--states', default=DEFAULT_STATES, help=f'States parquet path. Default: {DEFAULT_STATES}')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help=f'Base output directory. Default: {DEFAULT_OUTPUT_DIR}')
    parser.add_argument('--months', nargs='+', help='Optional explicit source_month values to process.')
    parser.add_argument('--state-column', default='pred_estado', help='State column read from the states parquet. Default: pred_estado')
    parser.add_argument('--anomaly-threshold', type=float, default=1.0, help='Anomaly threshold passed to Model_B nominal comparison. Default: 1.0')
    parser.add_argument('--skip-existing-months', action='store_true', help='Reuse existing monthly outputs when present.')
    parser.add_argument('--fail-fast', action='store_true', help='Stop at the first failed month.')
    return parser.parse_args()


def collect_months(states_path: Path) -> list[str]:
    """List the ``source_month`` values present in the states parquet.

    Args:
        states_path (Path): Parquet file with a ``source_month`` column.

    Returns:
        list[str]: Sorted unique month identifiers as strings (``YYYYMM``),
        null values dropped.

    Raises:
        FileNotFoundError: If ``states_path`` does not exist.
    """
    frame = pd.read_parquet(states_path, columns=['source_month'])
    return sorted(frame['source_month'].dropna().astype(str).unique().tolist())


def load_month_timeline(states_path: Path, month: str, state_column: str) -> pd.DataFrame:
    """Load and normalise the state timeline of one month.

    Args:
        states_path (Path): States parquet with ``source_month``, ``date``
            and ``state_column``.
        month (str): ``source_month`` value to load (pushed down as a
            Parquet filter).
        state_column (str): Name of the state column, e.g. ``pred_estado``.

    Returns:
        pandas.DataFrame: Columns ``source_month``, ``Time`` (timezone-naive
        UTC timestamps) and ``Predicted_State``, sorted by ``Time``. The
        unmodified empty frame is returned when the month has no rows.

    Notes:
        Timestamps are parsed with ``format='mixed'`` because the monthly
        exports do not share a single date format.
    """
    frame = pd.read_parquet(states_path, columns=['source_month', 'date', state_column], filters=[('source_month', '==', month)])
    if frame.empty:
        return frame
    frame = frame.rename(columns={'date': 'Time', state_column: 'Predicted_State'})
    frame['Time'] = pd.to_datetime(frame['Time'], format='mixed', utc=True).dt.tz_convert(None)
    return frame.sort_values('Time')


def active_sequences_to_frame(sequences: list[ActiveSequence]) -> pd.DataFrame:
    """Convert Model_B active sequences into a flat dataframe.

    Args:
        sequences (list[ActiveSequence]): Sequences from
            :meth:`BehavioralSequenceAnalyzer.extract_active_sequences`.

    Returns:
        pandas.DataFrame: One row per sequence with ``start_time``,
        ``end_time``, ``states`` (string form of the state tuple, e.g.
        ``"(3, 5)"``), ``total_duration_seconds`` and ``run_count``. Empty
        (no columns) when ``sequences`` is empty.

    Notes:
        The string form of ``states`` is parsed back by
        :func:`frame_to_active_sequences`; keep both functions consistent.
    """
    return pd.DataFrame(
        [
            {
                'start_time': sequence.start_time,
                'end_time': sequence.end_time,
                'states': str(sequence.states),
                'total_duration_seconds': sequence.total_duration_seconds,
                'run_count': sequence.run_count,
            }
            for sequence in sequences
        ]
    )


def runs_to_frame(runs: list) -> pd.DataFrame:
    """Convert Model_B state runs into a dataframe.

    Args:
        runs (list[StateRun]): Runs from
            :meth:`BehavioralSequenceAnalyzer.extract_runs`.

    Returns:
        pandas.DataFrame: One row per run with the dataclass fields
        (``state``, ``start_time``, ``end_time``, ``sample_count``,
        ``duration_seconds``).

    Notes:
        Relies on ``run.__dict__``; it would break if ``StateRun`` were
        declared with ``slots=True``.
    """
    return pd.DataFrame([run.__dict__ for run in runs])


def consolidate_csvs(paths: list[Path]) -> pd.DataFrame:
    """Concatenate monthly CSV files into one dataframe.

    Args:
        paths (list[Path]): CSV paths in the desired output order; paths
            that are not files are ignored.

    Returns:
        pandas.DataFrame: Row-wise concatenation with a fresh index, or an
        empty dataframe when no file exists.
    """
    frames = [pd.read_csv(path) for path in paths if path.is_file()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_nominal_lookup(sequences: list[ActiveSequence]) -> dict[tuple[int, ...], tuple[int, float]]:
    """Summarise reference sequences by their state word.

    Args:
        sequences (list[ActiveSequence]): Reference (nominal) sequences,
            in practice all sequences of previously processed months.

    Returns:
        dict[tuple[int, ...], tuple[int, float]]: For every distinct state
        word, the number of occurrences and the mean
        ``total_duration_seconds`` (seconds). Empty dict for no sequences.

    Notes:
        The lookup is rebuilt from the full cumulative list after every
        month, so its cost grows linearly with the campaign length.
    """
    if not sequences:
        return {}
    rows = {}
    for sequence in sequences:
        count, total_duration = rows.get(sequence.states, (0, 0.0))
        rows[sequence.states] = (count + 1, total_duration + float(sequence.total_duration_seconds))
    return {states: (count, total_duration / max(count, 1)) for states, (count, total_duration) in rows.items()}


def fast_compare_to_nominal(
    sequences: list[ActiveSequence],
    nominal_lookup: dict[tuple[int, ...], tuple[int, float]],
    *,
    anomaly_threshold: float,
) -> pd.DataFrame:
    """Compare sequences with the nominal lookup using a fast heuristic.

    For a sequence whose state word exists in the lookup, the anomaly score
    is the absolute relative duration deviation
    ``|duration - nominal_mean| / nominal_mean``. Otherwise the nominal
    candidate is the word with the closest length (ties broken by the
    highest occurrence count), and the score is
    ``positional_mismatch_fraction + |relative duration deviation| + 1``,
    so an unseen word always scores at least ``1.0``.

    Args:
        sequences (list[ActiveSequence]): Sequences of the current month.
        nominal_lookup (dict[tuple[int, ...], tuple[int, float]]): Output
            of :func:`build_nominal_lookup`; must not be empty.
        anomaly_threshold (float): Sequences with score ``>=`` this value
            are flagged ``is_anomalous``.

    Returns:
        pandas.DataFrame: One row per input sequence, in input order, with
        ``observed_states``, ``nominal_states``, ``exact_match``,
        ``state_distance`` (number of positional mismatches, counting the
        length difference), ``dtw_distance`` (``state_distance`` divided by
        the longer length; a positional proxy, not true DTW),
        ``duration_ratio_delta`` (signed relative duration deviation; ``0``
        when the nominal mean duration is not positive), ``anomaly_score``,
        ``is_anomalous`` and ``nominal_count``.

    Raises:
        ValueError: If ``nominal_lookup`` is empty and some sequence has to
            be matched approximately (``min`` of an empty sequence).

    Notes:
        Row order must match ``sequences`` because Model_C aligns the
        comparison with the sequence table by position.
    """
    rows: list[dict[str, object]] = []
    for sequence in sequences:
        observed = tuple(sequence.states)
        if observed in nominal_lookup:
            nominal_count, nominal_duration = nominal_lookup[observed]
            duration_ratio_delta = 0.0 if nominal_duration <= 0 else float((sequence.total_duration_seconds - nominal_duration) / nominal_duration)
            anomaly_score = abs(duration_ratio_delta)
            rows.append(
                {
                    'observed_states': observed,
                    'nominal_states': observed,
                    'exact_match': True,
                    'state_distance': 0,
                    'dtw_distance': 0.0,
                    'duration_ratio_delta': duration_ratio_delta,
                    'anomaly_score': anomaly_score,
                    'is_anomalous': anomaly_score >= anomaly_threshold,
                    'nominal_count': nominal_count,
                }
            )
            continue

        # Note: nearest nominal word = closest length, then most frequent.
        candidate_states = min(
            nominal_lookup.keys(),
            key=lambda nominal: (abs(len(observed) - len(nominal)), -nominal_lookup[nominal][0]),
        )
        nominal_count, nominal_duration = nominal_lookup[candidate_states]
        overlap = sum(1 for left, right in zip(observed, candidate_states) if left == right)
        prefix_scale = max(len(observed), len(candidate_states), 1)
        state_distance = max(len(observed), len(candidate_states)) - overlap
        dtw_distance = float(state_distance / prefix_scale)
        duration_ratio_delta = 0.0 if nominal_duration <= 0 else float((sequence.total_duration_seconds - nominal_duration) / nominal_duration)
        # Why: the +1.0 offset guarantees that unseen words score at least 1.0,
        # i.e. they are anomalous under the default threshold.
        anomaly_score = float(dtw_distance + abs(duration_ratio_delta) + 1.0)
        rows.append(
            {
                'observed_states': observed,
                'nominal_states': candidate_states,
                'exact_match': False,
                'state_distance': state_distance,
                'dtw_distance': dtw_distance,
                'duration_ratio_delta': duration_ratio_delta,
                'anomaly_score': anomaly_score,
                'is_anomalous': anomaly_score >= anomaly_threshold,
                'nominal_count': nominal_count,
            }
        )
    return pd.DataFrame(rows)



def frame_to_active_sequences(frame: pd.DataFrame) -> list[ActiveSequence]:
    """Rebuild :class:`ActiveSequence` objects from a monthly sequence CSV.

    Used when a month is reused so that its sequences still feed the
    cumulative nominal reference.

    Args:
        frame (pandas.DataFrame): Output of :func:`active_sequences_to_frame`
            read back from CSV (``states`` stored as a tuple literal).

    Returns:
        list[ActiveSequence]: One sequence per row; empty for an empty frame.

    Raises:
        ValueError: If a ``states`` value is not a valid Python literal
            (raised by :func:`ast.literal_eval`).
    """
    if frame.empty:
        return []
    sequences: list[ActiveSequence] = []
    for row in frame.itertuples(index=False):
        sequences.append(
            ActiveSequence(
                start_time=pd.to_datetime(row.start_time),
                end_time=pd.to_datetime(row.end_time),
                states=tuple(int(item) for item in ast.literal_eval(row.states)),
                total_duration_seconds=float(row.total_duration_seconds),
                run_count=int(row.run_count),
            )
        )
    return sequences


def main() -> int:
    """Execute the month-wise Model_B -> Model_C orchestration.

    Returns:
        int: ``0`` on success; ``1`` when no ``source_month`` is found or at
        least one month failed.

    Notes:
        Per-month exceptions are caught, recorded in the manifest with status
        ``failed`` and, unless ``--fail-fast`` is set, processing continues.
        Consolidated CSVs are only (over)written when their content is not
        empty; the metadata JSON always lists their intended paths. Outputs
        are described in the module docstring.
    """
    args = parse_args()
    states_path = Path(args.states).resolve()
    base_output_dir = Path(args.output_dir).resolve()
    model_b_dir = base_output_dir / 'Model_B'
    model_c_dir = base_output_dir / 'Model_C'
    monthly_b_dir = model_b_dir / 'monthly'
    monthly_c_dir = model_c_dir / 'monthly'
    for directory in [model_b_dir, model_c_dir, monthly_b_dir, monthly_c_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    months = args.months or collect_months(states_path)
    if not months:
        print('No source_month values found in the states parquet.', file=sys.stderr)
        return 1

    analyzer = BehavioralSequenceAnalyzer(state_column='Predicted_State')
    interpreter = SemanticModeInterpreter()
    manifest_rows: list[dict[str, object]] = []
    sequence_paths: list[Path] = []
    assignment_paths: list[Path] = []
    family_paths: list[Path] = []
    comparison_paths: list[Path] = []
    cumulative_nominal_sequences: list[ActiveSequence] = []
    cumulative_nominal_lookup: dict[tuple[int, ...], tuple[int, float]] = {}
    failures = 0

    for month in months:
        month_runs_path = monthly_b_dir / f'{month}_state_runs.csv'
        month_sequences_path = monthly_b_dir / f'{month}_active_sequences.csv'
        month_words_path = monthly_b_dir / f'{month}_sequence_words.csv'
        month_comparison_path = monthly_b_dir / f'{month}_sequence_comparison.csv'
        month_assignments_path = monthly_c_dir / f'{month}_semantic_assignments.csv'
        month_family_path = monthly_c_dir / f'{month}_incident_family_assignments.csv'
        month_summary_path = monthly_c_dir / f'{month}_semantic_mode_summary.csv'

        # Note: reuse requires sequences and assignments; the comparison file
        # is optional because the first processed month never has one.
        if args.skip_existing_months and month_sequences_path.is_file() and month_assignments_path.is_file():
            sequence_paths.append(month_sequences_path)
            assignment_paths.append(month_assignments_path)
            family_paths.append(month_family_path)
            if month_comparison_path.is_file():
                comparison_paths.append(month_comparison_path)
            existing_sequences = pd.read_csv(month_sequences_path)
            existing_assignments = pd.read_csv(month_assignments_path)
            reused_sequences = frame_to_active_sequences(existing_sequences)
            cumulative_nominal_sequences.extend(reused_sequences)
            cumulative_nominal_lookup = build_nominal_lookup(cumulative_nominal_sequences)
            anomalous_count = 0
            if month_comparison_path.is_file():
                comparison_frame = pd.read_csv(month_comparison_path)
                if 'is_anomalous' in comparison_frame.columns:
                    anomalous_count = int(comparison_frame['is_anomalous'].fillna(False).astype(bool).sum())
            manifest_rows.append({
                'source_month': month,
                'status': 'reused',
                'timeline_rows': None,
                'sequence_rows': int(existing_sequences.shape[0]),
                'assignment_rows': int(existing_assignments.shape[0]),
                'anomalous_sequences': anomalous_count,
                'incident_family_non_null': int(existing_assignments['incident_family'].notna().sum()) if 'incident_family' in existing_assignments.columns else 0,
                'error': None,
            })
            print(f'[{month}] reused monthly Model_B/Model_C outputs')
            continue

        try:
            timeline = load_month_timeline(states_path, month, args.state_column)
            if timeline.empty:
                manifest_rows.append({
                    'source_month': month,
                    'status': 'empty',
                    'timeline_rows': 0,
                    'sequence_rows': 0,
                    'assignment_rows': 0,
                    'anomalous_sequences': 0,
                    'incident_family_non_null': 0,
                    'error': None,
                })
                print(f'[{month}] empty month or stopped installation')
                continue

            timeline = timeline.set_index('Time')
            runs = analyzer.extract_runs(timeline)
            sequences = analyzer.extract_active_sequences(timeline)
            words = analyzer.summarize_sequence_words(sequences)
            sequences_frame = active_sequences_to_frame(sequences)
            runs_frame = runs_to_frame(runs)
            if sequences_frame.empty:
                manifest_rows.append({
                    'source_month': month,
                    'status': 'no_active_sequences',
                    'timeline_rows': int(len(timeline)),
                    'sequence_rows': 0,
                    'assignment_rows': 0,
                    'anomalous_sequences': 0,
                    'incident_family_non_null': 0,
                    'error': None,
                })
                print(f'[{month}] no active sequences')
                continue

            # Invariant: the nominal reference only contains earlier months, so
            # no month is compared against itself.
            comparison_frame = pd.DataFrame()
            if cumulative_nominal_lookup:
                comparison_frame = fast_compare_to_nominal(
                    sequences,
                    cumulative_nominal_lookup,
                    anomaly_threshold=args.anomaly_threshold,
                )
                assignments = interpreter.interpret_sequences(sequences_frame, comparison=comparison_frame)
            else:
                assignments = interpreter.interpret_sequences(sequences_frame)
            summary = interpreter.summarize_modes(assignments)

            runs_frame.to_csv(month_runs_path, index=False)
            sequences_frame.to_csv(month_sequences_path, index=False)
            words.to_csv(month_words_path, index=False)
            if not comparison_frame.empty:
                comparison_frame.to_csv(month_comparison_path, index=False)
                comparison_paths.append(month_comparison_path)
            assignments.to_csv(month_assignments_path, index=False)
            assignments.to_csv(month_family_path, index=False)
            summary.to_csv(month_summary_path, index=False)

            sequence_paths.append(month_sequences_path)
            assignment_paths.append(month_assignments_path)
            family_paths.append(month_family_path)
            cumulative_nominal_sequences.extend(sequences)
            cumulative_nominal_lookup = build_nominal_lookup(cumulative_nominal_sequences)
            anomalous_count = int(comparison_frame['is_anomalous'].fillna(False).astype(bool).sum()) if not comparison_frame.empty else 0
            manifest_rows.append({
                'source_month': month,
                'status': 'written',
                'timeline_rows': int(len(timeline)),
                'sequence_rows': int(len(sequences_frame)),
                'assignment_rows': int(len(assignments)),
                'anomalous_sequences': anomalous_count,
                'incident_family_non_null': int(assignments['incident_family'].notna().sum()) if 'incident_family' in assignments.columns else 0,
                'error': None,
            })
            print(f'[{month}] wrote {len(sequences_frame)} sequences, {len(assignments)} assignments, anomalous={anomalous_count}')
        except Exception as exc:  # pragma: no cover
            failures += 1
            manifest_rows.append({
                'source_month': month,
                'status': 'failed',
                'timeline_rows': None,
                'sequence_rows': None,
                'assignment_rows': None,
                'anomalous_sequences': None,
                'incident_family_non_null': None,
                'error': f'{type(exc).__name__}: {exc}',
            })
            print(f'[{month}] failed: {type(exc).__name__}: {exc}', file=sys.stderr)
            if args.fail_fast:
                break

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(base_output_dir / 'model_bc_batch_manifest.csv', index=False)

    consolidated_sequences = consolidate_csvs(sequence_paths)
    consolidated_assignments = consolidate_csvs(assignment_paths)
    consolidated_family = consolidate_csvs(family_paths)
    consolidated_comparison = consolidate_csvs(comparison_paths)

    if not consolidated_sequences.empty:
        consolidated_sequences.to_csv(model_b_dir / 'active_sequences.csv', index=False)
    if not consolidated_comparison.empty:
        consolidated_comparison.to_csv(model_b_dir / 'sequence_comparison.csv', index=False)
    if not consolidated_assignments.empty:
        consolidated_assignments.to_csv(model_c_dir / 'semantic_assignments.csv', index=False)
        consolidated_family.to_csv(model_c_dir / 'incident_family_assignments.csv', index=False)
        summary = (
            consolidated_assignments.groupby(
                ['operating_mode', 'working_mode', 'semantic_status', 'incident_family', 'life_regime'],
                dropna=False,
            )
            .size()
            .reset_index(name='count')
            .sort_values('count', ascending=False)
        )
        summary.to_csv(model_c_dir / 'semantic_mode_summary.csv', index=False)

    metadata = {
        'months_requested': months,
        'months_completed': manifest.loc[manifest['status'].isin(['written', 'reused']), 'source_month'].astype(str).tolist(),
        'empty_like_months': manifest.loc[manifest['status'].isin(['empty', 'no_active_sequences']), 'source_month'].astype(str).tolist(),
        'failed_months': manifest.loc[manifest['status'] == 'failed', 'source_month'].astype(str).tolist(),
        'failure_count': failures,
        'anomaly_threshold': args.anomaly_threshold,
        'outputs': {
            'manifest_csv': str(base_output_dir / 'model_bc_batch_manifest.csv'),
            'active_sequences_csv': str(model_b_dir / 'active_sequences.csv'),
            'sequence_comparison_csv': str(model_b_dir / 'sequence_comparison.csv'),
            'semantic_assignments_csv': str(model_c_dir / 'semantic_assignments.csv'),
            'incident_family_assignments_csv': str(model_c_dir / 'incident_family_assignments.csv'),
            'semantic_mode_summary_csv': str(model_c_dir / 'semantic_mode_summary.csv'),
        },
    }
    (base_output_dir / 'model_bc_batch_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    print(f"Manifest saved to: {base_output_dir / 'model_bc_batch_manifest.csv'}")
    if failures:
        print(f'Completed with {failures} failed months.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
