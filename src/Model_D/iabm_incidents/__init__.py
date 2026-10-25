"""Public package surface for Model_D incident and episode analysis.

The package exports the typed configuration objects, builders, evaluators,
pipeline orchestrators, and registry helpers required to assemble longitudinal
incident-processing workflows from either semantic assignments or richer
indicator-driven inputs.

Position in the multi-scale framework:
    ``Model_A`` infers controller states from low-intrusion telemetry,
    ``Model_B`` turns them into behavioural run-length sequences and words,
    ``Model_C`` assigns semantic modes, and ``Model_D`` (this package) turns
    that evidence, optionally enriched with three-phase electrical
    command-response residuals, into window-level incident evidence,
    episodes, family assignments, recovery assessments and occurrence
    summaries. It also hosts the matched M0-M3 evidence ablation, the
    prespecified multiscale sensitivity and the methodological extensions
    (conformal fusion, certified variational fourth-order evidence,
    fault injection).

.. mermaid::

   flowchart TB
       subgraph Inputs
           CFG[config: ModelDConfig and sub-configs]
           TAX[taxonomy: families and signatures]
           REG[registry / registry_builder: incident registries]
       end
       subgraph Evidence
           EXP[exposure: observation periods]
           WIN[windows: rolling windows]
           IND[indicators / baseline / behavioral: 7 behavioural indicators]
           ELE[electrical: command residual and Tconc, VQE solver]
           SCO[scoring / detection: deviation score]
       end
       subgraph Episodes
           SEG[segmentation]
           REC[recovery]
           CLA[classification]
           EPI[episodes: IncidentEpisodeBuilder]
           FEA[features]
       end
       subgraph Analysis
           EVA[evaluation / metrics / occurrence]
           INT[interpretation: post hoc FIR/Bode]
           ANT[anticipation: risk forecasting]
       end
       subgraph Studies
           ABL[ablation: M0-M3]
           MUL[multiscale: 15 min / 1 h / 6 h / 24 h]
           EXT[conformal / quantum / injection]
       end
       PIPE[pipeline: IndicatorPipeline] --> EXP --> WIN --> IND --> SCO
       ELE --> SCO
       SCO --> SEG --> EPI
       REC --> EPI
       CLA --> EPI
       EPI --> FEA --> EVA
       EPI --> INT
       REG --> EVA
       TAX --> CLA
       CFG --> PIPE
       ABL --> MUL
       ELE --> EXT

Public API (``__all__``), grouped:
    * Configuration: :class:`ModelDConfig`, :class:`ElectricalConfig`,
      :class:`WindowBuildConfig`, :class:`ExposureConfig`,
      :class:`AblationConfig`, :class:`MultiscaleSensitivityConfig`,
      :class:`AnticipationConfig`, :class:`FrequencyInterpretationConfig`,
      :class:`RegistryGenerationConfig`.
    * Exposure, windows and indicators: :func:`derive_observation_periods`,
      :func:`build_rolling_windows`, :func:`compute_window_indicators`,
      :func:`fit_indicator_baseline`, :class:`IndicatorBaseline`,
      :class:`BehavioralWindowEncoder`.
    * Electrical evidence: :class:`CommandResponseObserver`,
      :class:`TransientConcentration`, :class:`TransientReference`,
      :class:`DominantDirectionSolver`, :class:`VQEDirectionSolver`,
      :func:`causal_delay`, :func:`fourth_scatter`.
    * Scoring and episodes: :class:`DeviationScorer`,
      :class:`WeightedDeviationScorer`, :class:`CandidateSegment`,
      :class:`IncidentEpisodeBuilder`, :class:`IndicatorPipeline`,
      :class:`IndicatorPipelineResult`.
    * Taxonomy and registries: :data:`DEFAULT_INCIDENT_TAXONOMY`,
      :data:`FAMILY_SIGNATURES`, :class:`FamilySignature`,
      :class:`IncidentRecord`, :class:`IncidentRegistry`,
      :class:`CandidateIncidentRegistryBuilder`.
    * Evaluation and summaries: :class:`EpisodeEvaluator`,
      :class:`OccurrenceModeler`, :class:`ModelDExperimentRunner`,
      :class:`ExperimentResult`.
    * Interpretation and anticipation:
      :class:`EventFrequencyResponseInterpreter`,
      :class:`EventFrequencyInterpretationResult`,
      :func:`residual_delay_taps`, :func:`residual_frequency_response`,
      :class:`AnticipatoryMaintenanceForecaster`,
      :class:`AnticipationResult`.
    * Studies: :class:`FourLevelAblationRunner`,
      :class:`FourLevelAblationResult`, :class:`AblationLevelResult`,
      :class:`MultiscaleSensitivityRunner`,
      :class:`MultiscaleSensitivityResult`.

    The extension modules :mod:`iabm_incidents.conformal`,
    :mod:`iabm_incidents.quantum` and :mod:`iabm_incidents.injection` are not
    re-exported and must be imported explicitly.

Maintenance notes:
    * Importing the package imports every listed submodule (and therefore
      pandas/numpy); keep heavy optional dependencies such as matplotlib out
      of module import time.
    * Keep ``__all__`` and the import list in sync; Sphinx autodoc of the
      package relies on ``__all__``.
"""

from .ablation import (AblationConfig, AblationLevelResult, FourLevelAblationResult, FourLevelAblationRunner)
from .behavioral import BehavioralWindowEncoder
from .anticipation import AnticipationConfig, AnticipationResult, AnticipatoryMaintenanceForecaster
from .config import ModelDConfig
from .electrical import (ElectricalConfig, CommandResponseObserver, TransientConcentration,
                         TransientReference, VQEDirectionSolver, DominantDirectionSolver,
                         causal_delay, fourth_scatter)
from .episodes import CandidateSegment, IncidentEpisodeBuilder
from .experimental import ExperimentResult, ModelDExperimentRunner
from .evaluation import EpisodeEvaluator
from .exposure import ExposureConfig, derive_observation_periods
from .indicators import IndicatorBaseline, compute_window_indicators, fit_indicator_baseline
from .interpretation import (EventFrequencyResponseInterpreter, EventFrequencyInterpretationResult,
                             FrequencyInterpretationConfig, residual_delay_taps,
                             residual_frequency_response)
from .multiscale import (MultiscaleSensitivityConfig, MultiscaleSensitivityResult, MultiscaleSensitivityRunner)
from .occurrence import OccurrenceModeler
from .pipeline import IndicatorPipeline, IndicatorPipelineResult
from .registry import IncidentRecord, IncidentRegistry
from .registry_builder import CandidateIncidentRegistryBuilder, RegistryGenerationConfig
from .scoring import DeviationScorer, WeightedDeviationScorer
from .taxonomy import DEFAULT_INCIDENT_TAXONOMY, FAMILY_SIGNATURES, FamilySignature
from .windows import WindowBuildConfig, build_rolling_windows

__all__ = [
    "MultiscaleSensitivityConfig", "MultiscaleSensitivityResult", "MultiscaleSensitivityRunner",
    "AblationConfig", "AblationLevelResult", "FourLevelAblationResult",
    "FourLevelAblationRunner", "BehavioralWindowEncoder",
    "EventFrequencyResponseInterpreter", "EventFrequencyInterpretationResult",
    "FrequencyInterpretationConfig", "residual_delay_taps", "residual_frequency_response",
    "ElectricalConfig", "CommandResponseObserver", "TransientConcentration",
    "TransientReference", "VQEDirectionSolver", "DominantDirectionSolver",
    "causal_delay", "fourth_scatter",
    "AnticipationConfig",
    "AnticipationResult",
    "AnticipatoryMaintenanceForecaster",
    "CandidateSegment",
    "compute_window_indicators",
    "DEFAULT_INCIDENT_TAXONOMY",
    "derive_observation_periods",
    "ExperimentResult",
    "DeviationScorer",
    "EpisodeEvaluator",
    "ExposureConfig",
    "FAMILY_SIGNATURES",
    "FamilySignature",
    "fit_indicator_baseline",
    "IndicatorBaseline",
    "ModelDExperimentRunner",
    "IndicatorPipeline",
    "IndicatorPipelineResult",
    "CandidateIncidentRegistryBuilder",
    "IncidentEpisodeBuilder",
    "IncidentRecord",
    "RegistryGenerationConfig",
    "IncidentRegistry",
    "ModelDConfig",
    "OccurrenceModeler",
    "build_rolling_windows",
    "WeightedDeviationScorer",
    "WindowBuildConfig",
]
