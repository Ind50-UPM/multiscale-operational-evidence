"""Experimental evaluation utilities for Model_D.

Role in the pipeline
--------------------
This module runs Model D end-to-end in two comparable *modes* on the same
inputs and writes an auditable report:

* ``"semantic"`` -- episodes from Model_B sequences plus Model_C semantic
  assignments through :class:`~iabm_incidents.episodes.IncidentEpisodeBuilder`;
* ``"indicators"`` -- episodes from the indicator pipeline
  :class:`~iabm_incidents.pipeline.IndicatorPipeline`, optionally with the
  frozen electrical observer/reference of :mod:`iabm_incidents.electrical`.

Both modes are evaluated against the same normalized incident registry with
:class:`~iabm_incidents.evaluation.EpisodeEvaluator`.

.. mermaid::

    flowchart LR
        IN["sequences, assignments,<br/>registry, analogue, digital"] --> RS["run_semantic"]
        IN --> RI["run_indicators"]
        RS --> AR["_assemble_result"]
        RI --> AR
        AR --> ER["ExperimentResult"]
        ER --> EX["export_report<br/>CSV + HTML + JSON"]

Main public objects
-------------------
* :class:`ExperimentResult` -- frozen bundle of per-mode tables.
* :class:`ModelDExperimentRunner` -- runs modes and exports artifacts.

Output layout of :meth:`ModelDExperimentRunner.export_report`
-------------------------------------------------------------
::

    <output_dir>/comparison_summary.csv
    <output_dir>/<mode>/{episodes,window_scores,evaluation_summary,
                         registry_matches,family_summary,lead_time_summary}.csv
    <output_dir>/<mode>/event_interpretation/...   (indicator mode, optional)
    <output_dir>/experiment_report.html
    <output_dir>/experiment_metadata.json

Maintenance notes
-----------------
* The registry is normalized again inside :meth:`_assemble_result`; normalizing
  an already normalized registry is expected to be idempotent.
* The HTML report is self-contained (inline CSS and SVG, no external assets);
  all user-controlled labels are HTML-escaped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import html
import json

import pandas as pd

from .config import ModelDConfig
from .electrical import CommandResponseObserver, TransientConcentration
from .episodes import IncidentEpisodeBuilder
from .evaluation import EpisodeEvaluator
from .interpretation import EventFrequencyInterpretationResult, FrequencyInterpretationConfig
from .metrics import summarize_episode_metrics
from .pipeline import IndicatorPipeline
from .registry import IncidentRegistry


@dataclass(frozen=True)
class ExperimentResult:
    """Artifacts produced by one experiment mode.

    Attributes:
        mode: Mode name, ``"semantic"`` or ``"indicators"``.
        episodes: Episode table (see
            :class:`~iabm_incidents.episodes.IncidentEpisode`).
        window_scores: Window-score table used for segmentation.
        evaluation_summary: One-row summary from
            :meth:`~iabm_incidents.evaluation.EpisodeEvaluator.summarize`, or
            an empty frame without registry.
        registry_matches: Per-incident matches, or an empty frame without
            registry.
        family_summary: Per-family metrics from
            :func:`~iabm_incidents.metrics.summarize_episode_metrics`.
        lead_time_summary: Per-reference-family lead-time statistics (hours).
        event_interpretation: Optional frequency interpretation of electrical
            events (indicator mode only).
    """

    mode: str
    episodes: pd.DataFrame
    window_scores: pd.DataFrame
    evaluation_summary: pd.DataFrame
    registry_matches: pd.DataFrame
    family_summary: pd.DataFrame
    lead_time_summary: pd.DataFrame
    event_interpretation: EventFrequencyInterpretationResult | None = None


class ModelDExperimentRunner:
    """Run comparative experiments for semantic and indicator-driven detection.

    See Also:
        :class:`~iabm_incidents.pipeline.IndicatorPipeline`,
        :class:`~iabm_incidents.episodes.IncidentEpisodeBuilder`.
    """

    def __init__(
        self, config: ModelDConfig | None = None, *,
        command_observer: CommandResponseObserver | None = None,
        transient_model: TransientConcentration | None = None,
        interpretation_config: FrequencyInterpretationConfig | None = None,
    ) -> None:
        """Create the runner and its collaborators.

        Args:
            config: Model D configuration; ``None`` uses ``ModelDConfig()``.
            command_observer: Optional *fitted* command-response observer
                forwarded to the indicator pipeline.
            transient_model: Optional *fitted* transient-concentration model
                forwarded to the indicator pipeline.
            interpretation_config: Optional configuration for the event
                frequency interpretation forwarded to the pipeline.
        """
        self._config = config or ModelDConfig()
        self._builder = IncidentEpisodeBuilder(self._config)
        self._pipeline = IndicatorPipeline(self._config, command_observer=command_observer, transient_model=transient_model, interpretation_config=interpretation_config)
        self._evaluator = EpisodeEvaluator()
        self._registry = IncidentRegistry()

    def run_semantic(
        self,
        *,
        sequences: pd.DataFrame,
        assignments: pd.DataFrame,
        registry: pd.DataFrame | None = None,
    ) -> ExperimentResult:
        """Run the semantic route and evaluate it.

        Args:
            sequences: Model_B sequence table.
            assignments: Model_C assignment table.
            registry: Optional incident registry (raw or normalized).

        Returns:
            :class:`ExperimentResult` with ``mode="semantic"`` and no event
            interpretation.
        """
        window_scores = self._builder.build_window_scores(sequences, assignments)
        episodes = self._builder.build_episodes_from_semantic_assignments(
            sequences,
            assignments,
            window_scores=window_scores,
        )
        return self._assemble_result("semantic", episodes, window_scores, registry)

    def run_indicators(
        self,
        *,
        sequences: pd.DataFrame,
        assignments: pd.DataFrame,
        registry: pd.DataFrame | None = None,
        analogue: pd.DataFrame | None = None,
        digital: pd.DataFrame | None = None,
    ) -> ExperimentResult:
        """Run the indicator pipeline and evaluate it.

        Args:
            sequences: Model_B sequence table.
            assignments: Model_C assignment table.
            registry: Optional incident registry, also used by the pipeline to
                exclude exposure periods from nominal fitting.
            analogue: Optional analog (or three-phase electrical) telemetry.
            digital: Optional digital telemetry.

        Returns:
            :class:`ExperimentResult` with ``mode="indicators"`` and the
            pipeline's event interpretation, if any.
        """
        pipeline_result = self._pipeline.run(
            sequences=sequences,
            assignments=assignments,
            registry=registry,
            analogue=analogue,
            digital=digital,
        )
        return self._assemble_result("indicators", pipeline_result.episodes, pipeline_result.window_scores, registry, pipeline_result.event_interpretation)

    def run_comparison(
        self,
        *,
        sequences: pd.DataFrame,
        assignments: pd.DataFrame,
        registry: pd.DataFrame,
        analogue: pd.DataFrame | None = None,
        digital: pd.DataFrame | None = None,
    ) -> dict[str, ExperimentResult]:
        """Run both modes on the same inputs and normalized registry.

        Args:
            sequences: Model_B sequence table.
            assignments: Model_C assignment table.
            registry: Incident registry; normalized once with
                :meth:`~iabm_incidents.registry.IncidentRegistry.normalize`.
            analogue: Optional analog telemetry for the indicator mode.
            digital: Optional digital telemetry for the indicator mode.

        Returns:
            Mapping ``{"semantic": ..., "indicators": ...}``.
        """
        normalized_registry = self._registry.normalize(registry)
        return {
            "semantic": self.run_semantic(sequences=sequences, assignments=assignments, registry=normalized_registry),
            "indicators": self.run_indicators(
                sequences=sequences,
                assignments=assignments,
                registry=normalized_registry,
                analogue=analogue,
                digital=digital,
            ),
        }

    def export_report(self, results: dict[str, ExperimentResult], output_dir: str | Path) -> dict[str, Path]:
        """Write CSV tables, an HTML report and JSON metadata.

        Directories are created as needed and existing files are overwritten.

        Args:
            results: Mapping from mode name to :class:`ExperimentResult`, as
                returned by :meth:`run_comparison`.
            output_dir: Destination directory.

        Returns:
            Mapping from artifact key (``"comparison_summary"``,
            ``"<mode>_<table>"``, ``"<mode>_interpretation_<name>"``,
            ``"html_report"``, ``"metadata"``) to the written path.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        comparison_summary = self._comparison_summary(results)
        comparison_path = output_path / "comparison_summary.csv"
        comparison_summary.to_csv(comparison_path, index=False)
        written["comparison_summary"] = comparison_path

        for mode, result in results.items():
            mode_prefix = output_path / mode
            mode_prefix.mkdir(parents=True, exist_ok=True)
            for label, frame in {
                "episodes": result.episodes,
                "window_scores": result.window_scores,
                "evaluation_summary": result.evaluation_summary,
                "registry_matches": result.registry_matches,
                "family_summary": result.family_summary,
                "lead_time_summary": result.lead_time_summary,
            }.items():
                path = mode_prefix / f"{label}.csv"
                frame.to_csv(path, index=False)
                written[f"{mode}_{label}"] = path

            if result.event_interpretation is not None:
                for label, path in result.event_interpretation.export(mode_prefix / "event_interpretation").items():
                    written[f"{mode}_interpretation_{label}"] = path

        report_path = output_path / "experiment_report.html"
        report_path.write_text(self._build_html_report(results, comparison_summary), encoding="utf-8")
        written["html_report"] = report_path

        metadata_path = output_path / "experiment_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "modes": list(results.keys()),
                    "artifacts": {key: str(path) for key, path in written.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        written["metadata"] = metadata_path
        return written

    def _assemble_result(
        self,
        mode: str,
        episodes: pd.DataFrame,
        window_scores: pd.DataFrame,
        registry: pd.DataFrame | None,
        event_interpretation: EventFrequencyInterpretationResult | None = None,
    ) -> ExperimentResult:
        """Evaluate one mode's episodes and bundle all tables.

        Args:
            mode: Mode name stored in the result.
            episodes: Episode table of the mode.
            window_scores: Window-score table of the mode.
            registry: Optional registry; matching and summaries are skipped
                (empty frames) when it is ``None`` or empty.
            event_interpretation: Optional interpretation result to attach.

        Returns:
            The assembled :class:`ExperimentResult`.
        """
        normalized_registry = self._registry.normalize(registry) if registry is not None else pd.DataFrame()
        matches = self._evaluator.match(normalized_registry, episodes) if not normalized_registry.empty else pd.DataFrame()
        evaluation_summary = self._evaluator.summarize(normalized_registry, episodes) if not normalized_registry.empty else pd.DataFrame()
        family_summary = summarize_episode_metrics(episodes)
        lead_time_summary = self._lead_time_summary(matches)
        return ExperimentResult(mode, episodes, window_scores, evaluation_summary, matches, family_summary, lead_time_summary, event_interpretation)

    @staticmethod
    def _lead_time_summary(matches: pd.DataFrame) -> pd.DataFrame:
        """Aggregate lead times per reference family.

        Args:
            matches: Output of
                :meth:`~iabm_incidents.evaluation.EpisodeEvaluator.match`.

        Returns:
            One row per ``reference_family`` (``NaN`` kept as a group) with
            ``match_count``, ``mean_lead_time_hours``,
            ``median_lead_time_hours`` and ``positive_lead_fraction``, sorted
            by decreasing ``match_count``; an empty frame with these columns
            when there are no matches.

        Notes:
            Missing lead times are counted as 0 seconds.
        """
        if matches.empty or "lead_time_seconds" not in matches.columns:
            return pd.DataFrame(
                columns=[
                    "reference_family",
                    "match_count",
                    "mean_lead_time_hours",
                    "median_lead_time_hours",
                    "positive_lead_fraction",
                ]
            )
        frame = matches.copy()
        frame["lead_time_seconds"] = pd.to_numeric(frame["lead_time_seconds"], errors="coerce").fillna(0.0)
        rows: list[dict[str, Any]] = []
        for family, family_frame in frame.groupby("reference_family", dropna=False):
            rows.append(
                {
                    "reference_family": family,
                    "match_count": int(len(family_frame)),
                    "mean_lead_time_hours": float(family_frame["lead_time_seconds"].mean() / 3600.0),
                    "median_lead_time_hours": float(family_frame["lead_time_seconds"].median() / 3600.0),
                    "positive_lead_fraction": float(family_frame["lead_time_seconds"].gt(0).mean()),
                }
            )
        return pd.DataFrame(rows).sort_values("match_count", ascending=False)

    @staticmethod
    def _comparison_summary(results: dict[str, ExperimentResult]) -> pd.DataFrame:
        """Build one comparison row per mode.

        Args:
            results: Mapping from mode name to :class:`ExperimentResult`.

        Returns:
            Dataframe with ``mode``, ``episode_count``, ``window_count`` and,
            when available, the columns of the mode's evaluation summary.
        """
        rows: list[dict[str, Any]] = []
        for mode, result in results.items():
            summary_row = result.evaluation_summary.iloc[0].to_dict() if not result.evaluation_summary.empty else {}
            rows.append(
                {
                    "mode": mode,
                    "episode_count": int(len(result.episodes)),
                    "window_count": int(len(result.window_scores)),
                    **summary_row,
                }
            )
        return pd.DataFrame(rows)

    def _build_html_report(self, results: dict[str, ExperimentResult], comparison_summary: pd.DataFrame) -> str:
        """Render the self-contained HTML experiment report.

        Args:
            results: Mapping from mode name to :class:`ExperimentResult`.
            comparison_summary: Output of :meth:`_comparison_summary`.

        Returns:
            Complete HTML document as a string.
        """
        sections = [
            "<html><head><meta charset='utf-8'><title>Model_D Experimental Report</title>",
            "<style>body{font-family:Arial,sans-serif;margin:24px;}table{border-collapse:collapse;margin:16px 0;}th,td{border:1px solid #ccc;padding:6px 10px;}h1,h2{margin-top:24px;} .chart{margin:18px 0;padding:12px;border:1px solid #ddd;} svg text{font-size:12px;font-family:Arial,sans-serif;}</style>",
            "</head><body>",
            "<h1>Model_D Experimental Report</h1>",
            "<h2>Comparison Summary</h2>",
            comparison_summary.to_html(index=False),
            self._svg_bar_chart(comparison_summary, "mode", "episode_precision", "Episode Precision by Mode"),
            self._svg_bar_chart(comparison_summary, "mode", "family_precision", "Family Precision by Mode"),
        ]
        for mode, result in results.items():
            sections.extend(
                [
                    f"<h2>Mode: {html.escape(mode)}</h2>",
                    f"<p>Episodes: {len(result.episodes)} | Windows: {len(result.window_scores)}</p>",
                    "<h3>Evaluation Summary</h3>",
                    result.evaluation_summary.to_html(index=False) if not result.evaluation_summary.empty else "<p>No evaluation summary available.</p>",
                    "<h3>Family Summary</h3>",
                    result.family_summary.to_html(index=False) if not result.family_summary.empty else "<p>No family summary available.</p>",
                    self._svg_bar_chart(result.family_summary, "primary_family", "episode_count", f"Episode Count by Family ({mode})"),
                    self._svg_bar_chart(result.lead_time_summary, "reference_family", "mean_lead_time_hours", f"Mean Lead Time by Family ({mode})"),
                ]
            )
        sections.append("</body></html>")
        return "\n".join(sections)

    @staticmethod
    def _svg_bar_chart(frame: pd.DataFrame, category_col: str, value_col: str, title: str) -> str:
        """Render a horizontal inline-SVG bar chart.

        Args:
            frame: Source table.
            category_col: Column used as bar labels.
            value_col: Numeric column used as bar lengths (non-numeric values
                become 0).
            title: Chart title (HTML-escaped).

        Returns:
            HTML fragment; a "No data." placeholder when the frame is empty or
            a column is missing.

        Notes:
            Bar lengths are proportional to ``value / max(value)`` with the
            maximum floored at ``1e-9``; the chart is designed for nonnegative
            values (negative values produce negative SVG widths).
        """
        if frame.empty or category_col not in frame.columns or value_col not in frame.columns:
            return f"<div class='chart'><strong>{html.escape(title)}</strong><p>No data.</p></div>"
        plotting = frame[[category_col, value_col]].copy()
        plotting[value_col] = pd.to_numeric(plotting[value_col], errors="coerce").fillna(0.0)
        if plotting.empty:
            return f"<div class='chart'><strong>{html.escape(title)}</strong><p>No data.</p></div>"
        max_value = float(plotting[value_col].max())
        max_value = max(max_value, 1e-9)
        width = 700
        bar_height = 26
        gap = 12
        left_pad = 180
        top_pad = 40
        height = top_pad + len(plotting) * (bar_height + gap) + 20
        bars: list[str] = [f"<div class='chart'><strong>{html.escape(title)}</strong><svg width='{width}' height='{height}'>"]
        bars.append(f"<text x='10' y='22'>{html.escape(title)}</text>")
        for idx, row in plotting.reset_index(drop=True).iterrows():
            y = top_pad + idx * (bar_height + gap)
            label = html.escape(str(row[category_col]))
            value = float(row[value_col])
            bar_width = 0 if max_value == 0 else int((width - left_pad - 80) * (value / max_value))
            bars.append(f"<text x='10' y='{y + 18}'>{label}</text>")
            bars.append(f"<rect x='{left_pad}' y='{y}' width='{bar_width}' height='{bar_height}' fill='#3b82f6'></rect>")
            bars.append(f"<text x='{left_pad + bar_width + 8}' y='{y + 18}'>{value:.3f}</text>")
        bars.append("</svg></div>")
        return "".join(bars)
