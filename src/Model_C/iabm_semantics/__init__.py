"""Public package surface for Model_C semantic interpretation.

Model_C is the third layer of the multi-scale operational-evidence pipeline
(Model_A state inference -> Model_B behavioural sequences -> **Model_C
semantic modes** -> Model_D incident evidence/episodes). It reads Model_B
active-sequence reports (optionally with the matching nominal-comparison
report) and labels every sequence with decoded components, operating/working
modes, a semantic status, an incident-family proxy, a reliability
interpretation, and life/recovery regimes.

The package exports only the typed semantic interpreter and its rule and output
structures so downstream incident-processing layers can rely on a stable
semantic contract.

Exported objects:

* :class:`~iabm_semantics.semantics.SemanticModeInterpreter` -- the
  interpretation engine.
* :class:`~iabm_semantics.semantics.SemanticRule` -- declarative
  component-set -> mode rule.
* :class:`~iabm_semantics.semantics.SemanticAssignment` -- row-level output
  record.

Examples:
    >>> from iabm_semantics import SemanticModeInterpreter
    >>> interpreter = SemanticModeInterpreter()
    >>> interpreter.component_map[32]
    'BASIN_PUMP'

Maintenance notes:
    ``DEFAULT_COMPONENT_MAP`` lives in :mod:`iabm_semantics.semantics` and is
    intentionally not re-exported; import it from there if needed. Anything
    added to ``__all__`` becomes part of the contract consumed by Model_D and
    the repository batch utilities.
"""

from .semantics import (
    SemanticAssignment,
    SemanticModeInterpreter,
    SemanticRule,
)

__all__ = [
    "SemanticAssignment",
    "SemanticModeInterpreter",
    "SemanticRule",
]
