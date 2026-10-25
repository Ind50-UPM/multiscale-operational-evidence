"""Concatenate monthly CSV datasets and export one Parquet file per CSV type.

Purpose:
    First data-preparation step of the batch pipeline. Historical campaigns
    are delivered as one sub-directory per ``source_month`` (for example
    ``data/202207/``), each containing the same set of CSV exports. This
    utility scans those monthly folders, concatenates every CSV that shares a
    file name and writes a single Parquet file per name, so the downstream
    batch runners can read one columnar file with predicate push-down on
    ``source_month`` instead of hundreds of small CSV files.

Command line (from the repository root)::

    python src/concat_monthly_csv_to_parquet.py --data-dir data --include-source-month

    # Inspect what would be processed without writing anything:
    python src/concat_monthly_csv_to_parquet.py --data-dir data --dry-run

No ``PYTHONPATH`` is required: the script only depends on pandas and a
Parquet engine (pyarrow).

Inputs:
    * ``<data-dir>/<YYYYMM>/palabras.csv``: monthly sequence-word summaries.
    * ``<data-dir>/<YYYYMM>/secuencias.csv``: monthly state-run sequence
      summaries.
    * ``<data-dir>/<YYYYMM>/estados_nonans.csv``: monthly state timeline. The
      downstream runners read the columns ``date``, ``estado``,
      ``pred_estado`` and ``RP1``--``RP4`` from it.

    Only sub-directories whose name consists solely of digits are treated as
    monthly folders. Any other CSV name can be aggregated with
    ``--filenames``. No column schema is enforced here; files are concatenated
    as read by :func:`pandas.read_csv` (column union, missing values as NaN).

Outputs:
    One ``<stem>.parquet`` per requested CSV name in ``--output-dir`` (default:
    the data directory itself), i.e. by default ``data/palabras.parquet``,
    ``data/secuencias.parquet`` and ``data/estados_nonans.parquet``.

Script dependencies:
    Must run before ``generate_model_bc_batch.py`` (reads
    ``data/estados_nonans.parquet``) and
    ``generate_candidate_incident_registry_batch.py`` (reads all three
    Parquet files). ``run_model_d_ablation.py`` also reads
    ``data/estados_nonans.parquet`` and filters it on ``source_month``.

.. mermaid::

    flowchart LR
        M["data/YYYYMM/*.csv (one folder per month)"] --> C[concat_monthly_csv_to_parquet.py]
        C --> P1[data/estados_nonans.parquet]
        C --> P2[data/secuencias.parquet]
        C --> P3[data/palabras.parquet]
        P1 --> BC[generate_model_bc_batch.py]
        P1 --> REG[generate_candidate_incident_registry_batch.py]
        P2 --> REG
        P3 --> REG
        P1 --> ABL[run_model_d_ablation.py]

Maintenance notes:
    The ``source_month`` column is only added when ``--include-source-month``
    is given. Every downstream batch runner filters on ``source_month``, so
    that flag is effectively mandatory when preparing inputs for the
    pipeline (unless the monthly CSVs already carry the column). The process
    exit code is ``1`` when any requested file name is missing in every
    monthly folder, even if other names were written successfully. Each
    concatenated table is held fully in memory before being written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


#: CSV file names aggregated by default; each becomes ``<stem>.parquet``.
DEFAULT_FILENAMES = (
    "palabras.csv",
    "secuencias.csv",
    "estados_nonans.csv",
)


def parse_args() -> argparse.Namespace:
    """Parse the command-line options of the consolidation utility.

    Returns:
        argparse.Namespace: Parsed options with the attributes ``data_dir``
        (str, base folder holding the monthly sub-directories, default
        ``"data"``), ``output_dir`` (str or ``None``; ``None`` means "same as
        ``data_dir``"), ``filenames`` (list[str], CSV names to aggregate,
        default :data:`DEFAULT_FILENAMES`), ``include_source_month`` (bool)
        and ``dry_run`` (bool).

    Notes:
        ``argparse`` exits the process with status 2 on invalid options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Scan monthly subdirectories, concatenate matching CSV files, "
            "and write one Parquet file per CSV name."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base directory containing monthly subdirectories. Default: data",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where Parquet files will be written. Default: same as --data-dir",
    )
    parser.add_argument(
        "--filenames",
        nargs="+",
        default=list(DEFAULT_FILENAMES),
        help=(
            "CSV filenames to aggregate across monthly folders. "
            f"Default: {' '.join(DEFAULT_FILENAMES)}"
        ),
    )
    parser.add_argument(
        "--include-source-month",
        action="store_true",
        help="Add a source_month column derived from the subdirectory name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be processed without writing output.",
    )
    return parser.parse_args()


def discover_month_dirs(data_dir: Path) -> list[Path]:
    """List the monthly sub-directories of a data folder.

    Args:
        data_dir (Path): Base directory to scan (not recursive).

    Returns:
        list[Path]: Sub-directories whose name is made only of digits
        (for example ``202207``), sorted lexicographically, which for
        ``YYYYMM`` names is chronological order.

    Raises:
        OSError: Propagated from :meth:`pathlib.Path.iterdir` if the
            directory cannot be listed.
    """
    return sorted(
        path
        for path in data_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    )


def collect_existing_files(month_dirs: list[Path], filename: str) -> list[Path]:
    """Return the monthly copies of one CSV file that actually exist.

    Args:
        month_dirs (list[Path]): Monthly folders, usually from
            :func:`discover_month_dirs`.
        filename (str): Bare file name to look for in each folder, e.g.
            ``"secuencias.csv"``.

    Returns:
        list[Path]: ``month_dir / filename`` for every folder where it is a
        regular file, preserving the order of ``month_dirs``. Months without
        the file are silently skipped.
    """
    return [month_dir / filename for month_dir in month_dirs if (month_dir / filename).is_file()]


def load_and_concat(csv_paths: list[Path], include_source_month: bool) -> pd.DataFrame:
    """Read several CSV files and stack them into one dataframe.

    Args:
        csv_paths (list[Path]): CSV files to read, in the order they should
            appear in the output.
        include_source_month (bool): When ``True``, insert a first column
            ``source_month`` holding the name of each file's parent folder
            (a string such as ``"202207"``).

    Returns:
        pandas.DataFrame: Row-wise concatenation with a fresh
        ``RangeIndex``; an empty dataframe when ``csv_paths`` is empty.

    Raises:
        pandas.errors.ParserError: If a CSV file is malformed.
        OSError: If a file cannot be read.

    Notes:
        All files are loaded into memory before concatenation. Column types
        are inferred independently per file, so a column may end up with
        ``object`` dtype if months disagree.
    """
    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        if include_source_month:
            frame.insert(0, "source_month", csv_path.parent.name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parquet_name_for(filename: str) -> str:
    """Map a CSV file name to its Parquet output name.

    Args:
        filename (str): CSV name such as ``"estados_nonans.csv"``.

    Returns:
        str: The same stem with a ``.parquet`` suffix, e.g.
        ``"estados_nonans.parquet"``.

    Examples:
        >>> parquet_name_for("palabras.csv")
        'palabras.parquet'
    """
    return f"{Path(filename).stem}.parquet"


def main() -> int:
    """Run the monthly CSV-to-Parquet consolidation.

    Returns:
        int: Process exit code. ``0`` when every requested file name was
        found in at least one monthly folder (and written unless
        ``--dry-run``); ``1`` when the data directory is missing, contains
        no monthly sub-directories, or at least one requested name was not
        found anywhere.

    Notes:
        Writes ``<output-dir>/<stem>.parquet`` for every requested CSV name
        (overwriting existing files) and prints the list of source files and
        the number of rows written. With ``--dry-run`` nothing is created,
        not even the output directory.
    """
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else data_dir

    if not data_dir.is_dir():
        print(f"Data directory does not exist: {data_dir}", file=sys.stderr)
        return 1

    month_dirs = discover_month_dirs(data_dir)
    if not month_dirs:
        print(f"No monthly subdirectories found in: {data_dir}", file=sys.stderr)
        return 1

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    for filename in args.filenames:
        csv_paths = collect_existing_files(month_dirs, filename)
        if not csv_paths:
            print(f"Skipped {filename}: no matching files found.")
            exit_code = 1
            continue

        print(f"{filename}: found {len(csv_paths)} files")
        for csv_path in csv_paths:
            print(f"  - {csv_path}")

        if args.dry_run:
            continue

        combined = load_and_concat(csv_paths, args.include_source_month)
        parquet_path = output_dir / parquet_name_for(filename)
        combined.to_parquet(parquet_path, index=False)
        print(f"Wrote {len(combined)} rows to {parquet_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
