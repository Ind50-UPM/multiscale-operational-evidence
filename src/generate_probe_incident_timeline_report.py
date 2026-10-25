"""Compare detected Model_D episodes with reported maintenance and work orders.

Purpose:
    Builds a visual and tabular comparison, over the focus horizon
    2024-02-01 to 2026-04-30, between

    * maintenance/incident periods listed in a Word table
      (``Tabla_Incidencias.docx``),
    * work-order dates (OT, *orden de trabajo*) listed in a second Word
      document (``OT - INCIDENCIAS LAVA RUEDAS.docx``), and
    * the episodes detected by two retained Model_D variants (``baseline``
      and ``localized_fallback``) plus the anomalous windows of the
      localized variant.

    The Word files are parsed directly (``word/document.xml`` inside the
    ``.docx`` zip archive); no python-docx dependency is required. The
    report and figure texts are deliberately in Spanish (they were produced
    for the operator); this documentation is English only.

Command line (from the repository root)::

    python src/generate_probe_incident_timeline_report.py --dataset probe
    python src/generate_probe_incident_timeline_report.py --dataset full

No ``PYTHONPATH`` is required (pandas, matplotlib with the ``Agg`` backend).

Inputs:
    * :data:`MAINTENANCE_DOC` = ``/tmp/Tabla_Incidencias.docx``: first table;
      first row is a header; column 0 holds a concatenated start/end
      timestamp range (``YYYY-MM-DD[T ]HH:MM:SS`` twice), column 2 a comment.
    * :data:`OT_DOC` = ``/tmp/OT - INCIDENCIAS LAVA RUEDAS.docx``: first
      table; first two rows are headers; column 0 a window label (carried
      forward when blank), column 1 one or more ``YYYY-MM-DD`` dates,
      column 2 the work-order description.
    * ``<root>/Model_D_episodes_baseline/detected_episodes.csv`` and
      ``<root>/Model_D_episodes_localized_fallback/detected_episodes.csv``:
      columns ``event_start``, ``event_end``, ``primary_family``.
    * ``<root>/Model_D_episodes_localized_fallback/window_scores.csv``:
      columns ``start_time``, ``end_time``, ``semantic_status``.

    ``<root>`` is ``src/predictions/episode_eval_probe`` or
    ``src/predictions/episode_eval_full`` (``--dataset``).

Outputs (``<root>/comparison_report/``):
    ``incident_timeline_comparison.png``, ``unavailability_timeline.png``,
    ``monthly_burden_comparison.png``, ``comparison_summary.csv`` (per
    variant: episode count, median duration in hours, maintenance periods
    overlapped by at least one episode, work-order dates within +/-24 h of an
    episode) and ``incident_timeline_report.html``.

Script dependencies:
    Requires the retained Model_D episode evaluations under
    ``src/predictions/episode_eval_*`` (produced with the ``iabm_incidents``
    episode pipeline). ``run_model_d_ablation.py`` only *audits* the
    presence of this script, its Word sources and its full-dataset outputs
    (``reference_source_audit.csv``); it does not run it.

.. mermaid::

    flowchart LR
        D1[Tabla_Incidencias.docx] --> P[load_maintenance_periods]
        D2[OT - INCIDENCIAS LAVA RUEDAS.docx] --> O[load_ot_points]
        B[baseline detected_episodes.csv] --> E[load_episodes]
        L[localized detected_episodes.csv] --> E
        W[localized window_scores.csv] --> WS[load_windows]
        P --> F[3 PNG figures]
        O --> F
        E --> F
        WS --> F
        P --> S[build_summary]
        O --> S
        E --> S
        S --> H[comparison_summary.csv + HTML]

Maintenance notes:
    The Word sources are read from hard-coded temporary paths under
    ``/tmp`` and are not distributed with the repository, so the script
    cannot be rerun from the public release alone; the retained
    ``comparison_report`` folders are the reference outputs. The suptitle of
    the episode timeline still mentions ``2024-02 a 2024-10`` while the focus
    horizon constants extend to 2026-04-30. ``load_maintenance_periods`` and
    ``load_ot_points`` fail if no row falls inside the horizon (sorting an
    empty frame without columns). In the public release the
    ``window_scores.csv`` of the ``full`` dataset is not distributed.
"""
from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

#: XML namespace of WordprocessingML used to query ``document.xml``.
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
#: Word table of reported maintenance/incident periods (not distributed).
MAINTENANCE_DOC = Path("/tmp/Tabla_Incidencias.docx")
#: Word table of work orders (OT) for the wheel washer (not distributed).
OT_DOC = Path("/tmp/OT - INCIDENCIAS LAVA RUEDAS.docx")
#: Dataset used when ``--dataset`` is omitted.
DEFAULT_DATASET = "probe"
#: Retained Model_D evaluation roots selectable with ``--dataset``.
DATASET_ROOTS = {
    "probe": Path("src/predictions/episode_eval_probe"),
    "full": Path("src/predictions/episode_eval_full"),
}
#: Inclusive start of the analysed horizon (naive timestamp).
FOCUS_START = pd.Timestamp("2024-02-01 00:00:00")
#: Inclusive end of the analysed horizon (naive timestamp).
FOCUS_END = pd.Timestamp("2026-04-30 23:59:59")


def _extract_docx_rows(path: Path) -> list[list[str]]:
    """Extract the text of the first table of a ``.docx`` file.

    Args:
        path (Path): Word document (Office Open XML).

    Returns:
        list[list[str]]: One list per table row, one stripped string per
        cell (all text runs of the cell concatenated).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        zipfile.BadZipFile: If the file is not a valid ``.docx`` archive.
        KeyError: If the archive has no ``word/document.xml``.
        IndexError: If the document contains no table.
    """
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    table = root.findall(".//w:tbl", NS)[0]
    rows: list[list[str]] = []
    for tr in table.findall("./w:tr", NS):
        cells: list[str] = []
        for tc in tr.findall("./w:tc", NS):
            texts = [node.text or "" for node in tc.findall(".//w:t", NS)]
            cells.append("".join(texts).strip())
        rows.append(cells)
    return rows


def _parse_concat_range(text: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Parse a start/end range written as two concatenated timestamps.

    Args:
        text (str): Cell text such as ``"2024-03-01T08:00:002024-03-01 12:00:00"``;
            ``None`` or empty is accepted.

    Returns:
        tuple[pandas.Timestamp | None, pandas.Timestamp | None]: The first
        two timestamps found (naive), or ``(None, None)`` when fewer than
        two are present.
    """
    matches = re.findall(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", text or "")
    if len(matches) < 2:
        return None, None
    return pd.Timestamp(matches[0].replace("T", " ")), pd.Timestamp(matches[1].replace("T", " "))


def _parse_dates(text: str) -> list[pd.Timestamp]:
    """Extract every ``YYYY-MM-DD`` date from a text.

    Args:
        text (str): Free text; ``None`` or empty is accepted.

    Returns:
        list[pandas.Timestamp]: Dates at midnight, in order of appearance.
    """
    return [pd.Timestamp(item) for item in re.findall(r"\d{4}-\d{2}-\d{2}", text or "")]


def load_maintenance_periods() -> pd.DataFrame:
    """Load reported maintenance periods clipped to the focus horizon.

    Returns:
        pandas.DataFrame: Columns ``start``, ``end`` (clipped to
        [:data:`FOCUS_START`, :data:`FOCUS_END`]), ``label`` (comment, or
        the Spanish placeholder ``"SIN COMENTARIOS"``) and ``source``;
        sorted by ``start``.

    Raises:
        KeyError: If no period overlaps the horizon (empty frame without a
            ``start`` column).

    Notes:
        Reads :data:`MAINTENANCE_DOC`; rows without a parsable range are
        skipped.
    """
    rows = _extract_docx_rows(MAINTENANCE_DOC)[1:]
    periods: list[dict[str, object]] = []
    for row in rows:
        start, end = _parse_concat_range(row[0])
        if start is None or end is None:
            continue
        if end < FOCUS_START or start > FOCUS_END:
            continue
        periods.append(
            {
                "start": max(start, FOCUS_START),
                "end": min(end, FOCUS_END),
                "label": row[2] or "SIN COMENTARIOS",
                "source": "Tabla incidencias",
            }
        )
    return pd.DataFrame(periods).sort_values("start").reset_index(drop=True)


def load_ot_points() -> pd.DataFrame:
    """Load work-order (OT) dates inside the focus horizon.

    Returns:
        pandas.DataFrame: Columns ``date``, ``label`` (description or the
        Spanish placeholder ``"(sin etiqueta)"``), ``window`` (last non-empty
        value of column 0) and ``is_unavailability`` (bool, see
        :func:`_is_unavailability_label`); duplicates dropped, sorted by
        ``date``.

    Raises:
        KeyError: If no work-order date falls inside the horizon.

    Notes:
        Reads :data:`OT_DOC`. A cell may list several dates; each yields one
        record.
    """
    rows = _extract_docx_rows(OT_DOC)[2:]
    records: list[dict[str, object]] = []
    current_window = ""
    for row in rows:
        if row and row[0].strip():
            current_window = row[0].strip()
        part_date = row[1].strip() if len(row) > 1 else ""
        label = row[2].strip() if len(row) > 2 else ""
        for date in _parse_dates(part_date):
            if not (FOCUS_START <= date <= FOCUS_END):
                continue
            records.append(
                {
                    "date": date,
                    "label": label or "(sin etiqueta)",
                    "window": current_window,
                    "is_unavailability": _is_unavailability_label(label),
                }
            )
    frame = pd.DataFrame(records).drop_duplicates().sort_values("date").reset_index(drop=True)
    return frame


def _is_unavailability_label(label: str) -> bool:
    """Classify a work-order description as an unavailability event.

    Args:
        label (str): Work-order description (Spanish free text).

    Returns:
        bool: ``True`` when the lower-cased text contains one of the Spanish
        keywords meaning "not working", "not operational", "pit pump not
        working", "malfunction", "overflowing".
    """
    text = (label or "").lower()
    keywords = (
        "no funciona",
        "no operativo",
        "bomba no operativa",
        "bomba de foso no funciona",
        "fallo funcionamiento",
        "desbordando",
        "rebosando",
    )
    return any(keyword in text for keyword in keywords)


def load_episodes(directory: Path) -> pd.DataFrame:
    """Load detected episodes overlapping the horizon, clipped to it.

    Args:
        directory (Path): Model_D episode output folder containing
            ``detected_episodes.csv``.

    Returns:
        pandas.DataFrame: Overlapping episodes with ``event_start`` and
        ``event_end`` clipped to the horizon and an added
        ``duration_hours`` column (clipped duration), sorted by start.

    Raises:
        FileNotFoundError: If ``detected_episodes.csv`` is missing.

    Notes:
        Timestamps are compared with naive horizon bounds, so the CSV must
        store naive (or consistently local) timestamps.
    """
    frame = pd.read_csv(directory / "detected_episodes.csv", parse_dates=["event_start", "event_end"])
    frame = frame[(frame["event_end"] >= FOCUS_START) & (frame["event_start"] <= FOCUS_END)].copy()
    frame["event_start"] = frame["event_start"].clip(lower=FOCUS_START)
    frame["event_end"] = frame["event_end"].clip(upper=FOCUS_END)
    frame["duration_hours"] = (frame["event_end"] - frame["event_start"]).dt.total_seconds() / 3600.0
    return frame.sort_values("event_start").reset_index(drop=True)


def load_windows(directory: Path) -> pd.DataFrame:
    """Load scored windows overlapping the horizon.

    Args:
        directory (Path): Model_D episode output folder containing
            ``window_scores.csv``.

    Returns:
        pandas.DataFrame: Overlapping windows (not clipped), sorted by
        ``start_time``.

    Raises:
        FileNotFoundError: If ``window_scores.csv`` is missing.
    """
    frame = pd.read_csv(directory / "window_scores.csv", parse_dates=["start_time", "end_time"])
    frame = frame[(frame["end_time"] >= FOCUS_START) & (frame["start_time"] <= FOCUS_END)].copy()
    return frame.sort_values("start_time").reset_index(drop=True)


def plot_episode_timeline(periods: pd.DataFrame, ot_points: pd.DataFrame, baseline: pd.DataFrame, localized: pd.DataFrame, output_path: Path) -> None:
    """Plot reported periods, work orders and episodes for both variants.

    One panel per variant (baseline, localized fallback) with three lanes:
    maintenance periods (y=2), detected episodes coloured by
    ``primary_family`` (``process_saturation`` in red, others in green; y=1)
    and work-order dates (y=0, triangles for unavailability).

    Args:
        periods (pandas.DataFrame): Output of :func:`load_maintenance_periods`.
        ot_points (pandas.DataFrame): Output of :func:`load_ot_points`.
        baseline (pandas.DataFrame): Baseline episodes (:func:`load_episodes`).
        localized (pandas.DataFrame): Localized-fallback episodes.
        output_path (Path): PNG destination (180 dpi).

    Notes:
        ``legend(ncols=...)`` requires Matplotlib 3.6 or newer.
    """
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True, constrained_layout=True)
    variants = [("Baseline", baseline), ("Localized fallback", localized)]
    for axis, (title, episodes) in zip(axes, variants):
        for _, row in periods.iterrows():
            axis.barh(2, row["end"] - row["start"], left=row["start"], height=0.35, color="#d9d9d9", edgecolor="#888888")
        for _, row in episodes.iterrows():
            color = "#c1666b" if row["primary_family"] == "process_saturation" else "#2d6a4f"
            axis.barh(1, row["event_end"] - row["event_start"], left=row["event_start"], height=0.4, color=color, edgecolor="#333333")
        unavailability = ot_points[ot_points["is_unavailability"]]
        other_points = ot_points[~ot_points["is_unavailability"]]
        axis.scatter(unavailability["date"], [0] * len(unavailability), marker="v", s=70, color="#e76f51", label="OT indisponibilidad")
        axis.scatter(other_points["date"], [0] * len(other_points), marker="o", s=40, color="#1d3557", label="OT mantenimiento")
        axis.set_yticks([0, 1, 2])
        axis.set_yticklabels(["OT", "Episodios", "Tabla incidencias"])
        axis.set_title(title)
        axis.grid(axis="x", linestyle="--", alpha=0.3)
    axes[0].legend(loc="upper left", ncols=3)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(axes[-1].get_xticklabels(), rotation=30, ha="right")
    fig.suptitle("Incidentes en el tiempo y mantenimiento reportado (2024-02 a 2024-10)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _merge_touching_ranges(frame: pd.DataFrame, start_col: str, end_col: str, gap_hours: float = 6.0) -> pd.DataFrame:
    """Merge intervals that overlap or are separated by a small gap.

    Args:
        frame (pandas.DataFrame): Intervals with timestamp columns.
        start_col (str): Name of the start column.
        end_col (str): Name of the end column.
        gap_hours (float): Maximum gap (hours) between the end of one merged
            interval and the start of the next for them to be fused.

    Returns:
        pandas.DataFrame: Merged intervals with only ``start_col`` and
        ``end_col``, in chronological order; empty with those columns when
        ``frame`` is empty.
    """
    if frame.empty:
        return pd.DataFrame(columns=[start_col, end_col])
    rows = frame.sort_values(start_col)[[start_col, end_col]].to_dict("records")
    merged = [rows[0].copy()]
    gap = pd.Timedelta(hours=gap_hours)
    for row in rows[1:]:
        current = merged[-1]
        if row[start_col] <= current[end_col] + gap:
            current[end_col] = max(current[end_col], row[end_col])
        else:
            merged.append(row.copy())
    return pd.DataFrame(merged)


def plot_unavailability(periods: pd.DataFrame, ot_points: pd.DataFrame, localized: pd.DataFrame, windows: pd.DataFrame, output_path: Path) -> None:
    """Plot unavailability evidence on a single timeline.

    Lanes, from bottom to top: work-order unavailability dates (y=-0.55),
    reported maintenance periods (y=0), localized-fallback episodes merged
    with a 6 h gap (y=1) and ``ANOMALOUS`` windows merged with a 2 h gap
    (y=2).

    Args:
        periods (pandas.DataFrame): Reported maintenance periods.
        ot_points (pandas.DataFrame): Work-order dates with
            ``is_unavailability``.
        localized (pandas.DataFrame): Localized-fallback episodes.
        windows (pandas.DataFrame): Localized-fallback window scores with
            ``semantic_status``.
        output_path (Path): PNG destination (180 dpi).
    """
    anomalous = windows[windows["semantic_status"] == "ANOMALOUS"].copy()
    anomalous_ranges = _merge_touching_ranges(anomalous.rename(columns={"start_time": "start", "end_time": "end"}), "start", "end", gap_hours=2.0)
    predicted = _merge_touching_ranges(localized.rename(columns={"event_start": "start", "event_end": "end"}), "start", "end", gap_hours=6.0)
    reported = periods.copy()
    reported = reported.rename(columns={"start": "start", "end": "end"})
    ot_unavailability = ot_points[ot_points["is_unavailability"]]

    fig, axis = plt.subplots(figsize=(18, 6), constrained_layout=True)
    for _, row in anomalous_ranges.iterrows():
        axis.barh(2, row["end"] - row["start"], left=row["start"], height=0.35, color="#f4a261", edgecolor="#b56576")
    for _, row in predicted.iterrows():
        axis.barh(1, row["end"] - row["start"], left=row["start"], height=0.35, color="#e76f51", edgecolor="#7f5539")
    for _, row in reported.iterrows():
        axis.barh(0, row["end"] - row["start"], left=row["start"], height=0.35, color="#adb5bd", edgecolor="#495057")
    axis.scatter(ot_unavailability["date"], [-0.55] * len(ot_unavailability), marker="v", s=70, color="#1d3557")
    axis.set_yticks([-0.55, 0, 1, 2])
    axis.set_yticklabels(["OT no servicio", "Tabla incidencias", "Episodios localized", "Ventanas anómalas"])
    axis.set_title("Indisponibilidades y periodos degradados")
    axis.grid(axis="x", linestyle="--", alpha=0.3)
    axis.xaxis.set_major_locator(mdates.MonthLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_monthly_summary(periods: pd.DataFrame, baseline: pd.DataFrame, localized: pd.DataFrame, output_path: Path) -> None:
    """Plot the monthly episode burden of both variants.

    Top panel: episodes overlapping each month (lines) and the number of
    maintenance periods overlapping the month (grey bars). Bottom panel:
    episode hours falling inside each month.

    Args:
        periods (pandas.DataFrame): Reported maintenance periods.
        baseline (pandas.DataFrame): Baseline episodes.
        localized (pandas.DataFrame): Localized-fallback episodes.
        output_path (Path): PNG destination (180 dpi).
    """
    months = pd.period_range(FOCUS_START, FOCUS_END, freq="M")

    def summarize(episodes: pd.DataFrame, label: str) -> pd.DataFrame:
        """Count overlapping episodes and active hours per calendar month.

        Args:
            episodes (pandas.DataFrame): Episodes with ``event_start`` and
                ``event_end``.
            label (str): Variant name stored in the ``variant`` column.

        Returns:
            pandas.DataFrame: ``month``, ``variant``, ``episode_count`` and
            ``active_hours`` (episode time inside the month, hours).
        """
        rows = []
        for month in months:
            start = month.start_time
            end = month.end_time
            overlap = episodes[(episodes.event_start <= end) & (episodes.event_end >= start)].copy()
            episode_count = len(overlap)
            active_hours = 0.0
            for _, row in overlap.iterrows():
                active_start = max(row["event_start"], start)
                active_end = min(row["event_end"], end)
                active_hours += max((active_end - active_start).total_seconds(), 0.0) / 3600.0
            rows.append({"month": month.to_timestamp(), "variant": label, "episode_count": episode_count, "active_hours": active_hours})
        return pd.DataFrame(rows)

    frame = pd.concat([summarize(baseline, "Baseline"), summarize(localized, "Localized fallback")], ignore_index=True)
    maintenance_rows = []
    for month in months:
        start = month.start_time
        end = month.end_time
        overlap = periods[(periods.start <= end) & (periods.end >= start)]
        maintenance_rows.append({"month": month.to_timestamp(), "maintenance_periods": len(overlap)})
    maintenance = pd.DataFrame(maintenance_rows)

    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True, constrained_layout=True)
    for variant, color in [("Baseline", "#457b9d"), ("Localized fallback", "#e76f51")]:
        subset = frame[frame["variant"] == variant]
        axes[0].plot(subset["month"], subset["episode_count"], marker="o", linewidth=2, label=variant, color=color)
        axes[1].plot(subset["month"], subset["active_hours"], marker="o", linewidth=2, label=variant, color=color)
    axes[0].bar(maintenance["month"], maintenance["maintenance_periods"], width=20, color="#ced4da", alpha=0.5, label="Periodos mantenimiento")
    axes[0].set_title("Carga mensual de episodios frente a mantenimiento")
    axes[0].set_ylabel("Episodios")
    axes[1].set_ylabel("Horas activas")
    axes[1].set_title("Horas anómalas por mes")
    for axis in axes:
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.legend(loc="upper left")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_summary(periods: pd.DataFrame, ot_points: pd.DataFrame, baseline: pd.DataFrame, localized: pd.DataFrame) -> pd.DataFrame:
    """Compute per-variant agreement counts with the maintenance sources.

    Args:
        periods (pandas.DataFrame): Reported maintenance periods.
        ot_points (pandas.DataFrame): Work-order dates.
        baseline (pandas.DataFrame): Baseline episodes.
        localized (pandas.DataFrame): Localized-fallback episodes.

    Returns:
        pandas.DataFrame: One row per variant (``baseline``,
        ``localized_fallback``) with ``episode_count``,
        ``median_duration_hours``, ``maintenance_period_hits`` (periods
        overlapped by at least one episode), ``maintenance_period_total``,
        ``ot_point_hits_pm24h`` (work-order dates with an episode within
        +/-24 hours) and ``ot_point_total``.

    Notes:
        These are coincidence counts, not precision/recall: one long episode
        can "hit" several periods.
    """
    rows = []
    for name, episodes in [("baseline", baseline), ("localized_fallback", localized)]:
        period_hits = sum(bool(((episodes.event_start <= row.end) & (episodes.event_end >= row.start)).any()) for row in periods.itertuples())
        point_hits = sum(bool(((episodes.event_start <= row.date + pd.Timedelta(hours=24)) & (episodes.event_end >= row.date - pd.Timedelta(hours=24))).any()) for row in ot_points.itertuples())
        rows.append(
            {
                "variant": name,
                "episode_count": int(len(episodes)),
                "median_duration_hours": float(episodes["duration_hours"].median()) if not episodes.empty else 0.0,
                "maintenance_period_hits": int(period_hits),
                "maintenance_period_total": int(len(periods)),
                "ot_point_hits_pm24h": int(point_hits),
                "ot_point_total": int(len(ot_points)),
            }
        )
    return pd.DataFrame(rows)


def write_html(summary: pd.DataFrame, output_path: Path) -> None:
    """Write the (Spanish-language) HTML comparison report.

    Args:
        summary (pandas.DataFrame): Output of :func:`build_summary`.
        output_path (Path): HTML destination; the three PNG figures are
            referenced by relative name and must be in the same folder.

    Notes:
        The file is written with the platform default encoding (no explicit
        ``encoding`` argument) although it declares UTF-8.
    """
    table_html = summary.to_html(index=False, float_format=lambda value: f"{value:.1f}")
    output_path.write_text(
        f"""
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Comparativa temporal de incidentes</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    img {{ max-width: 100%; border: 1px solid #ddd; margin: 12px 0 24px; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 10px; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>Comparativa temporal de incidentes e indisponibilidades</h1>
  <p>Horizonte analizado: del 1 de febrero de 2024 al 30 de abril de 2026.</p>
  {table_html}
  <h2>Línea temporal comparativa</h2>
  <img src="incident_timeline_comparison.png" alt="Línea temporal comparativa">
  <h2>Indisponibilidades</h2>
  <img src="unavailability_timeline.png" alt="Indisponibilidades">
  <h2>Resumen mensual</h2>
  <img src="monthly_burden_comparison.png" alt="Resumen mensual">
</body>
</html>
""".strip()
    )


def main() -> None:
    """Parse ``--dataset`` and generate the full comparison report.

    Raises:
        FileNotFoundError: If a Word source or an episode/window CSV is
            missing.

    Notes:
        Creates ``<root>/comparison_report/`` and writes the files listed in
        the module docstring, overwriting previous versions.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate incident timeline comparison report for probe or full outputs.")
    parser.add_argument("--dataset", choices=sorted(DATASET_ROOTS), default=DEFAULT_DATASET)
    args = parser.parse_args()

    root_dir = DATASET_ROOTS[args.dataset]
    output_dir = root_dir / "comparison_report"
    baseline_dir = root_dir / "Model_D_episodes_baseline"
    localized_dir = root_dir / "Model_D_episodes_localized_fallback"

    output_dir.mkdir(parents=True, exist_ok=True)
    periods = load_maintenance_periods()
    ot_points = load_ot_points()
    baseline = load_episodes(baseline_dir)
    localized = load_episodes(localized_dir)
    localized_windows = load_windows(localized_dir)

    plot_episode_timeline(periods, ot_points, baseline, localized, output_dir / "incident_timeline_comparison.png")
    plot_unavailability(periods, ot_points, localized, localized_windows, output_dir / "unavailability_timeline.png")
    plot_monthly_summary(periods, baseline, localized, output_dir / "monthly_burden_comparison.png")

    summary = build_summary(periods, ot_points, baseline, localized)
    summary.to_csv(output_dir / "comparison_summary.csv", index=False)
    write_html(summary, output_dir / "incident_timeline_report.html")

    print(f"Report written to: {output_dir}")


if __name__ == "__main__":
    main()
