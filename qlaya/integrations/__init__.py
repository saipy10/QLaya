"""Third-party agent and framework integrations for QLaya."""
from ._errors import QLLowConfidenceError
from .crewai import (
    CrewRouteDecision,
    QLCrewRouter,
    QLTaskGuard,
    QLTaskGuardError,
)
from .langchain import (
    QLDecision,
    QLEvaluator,
    QLGuardrail,
    QLGuardrailError,
    QLRouter,
    QLTriage,
)
from .llamaindex import (
    QLMultiSelector,
    QLQueryRouter,
    QLSingleSelector,
)

# Backward-compat aliases (deprecated, will be removed in QLaya 1.0)
QLayaRouter = QLRouter
QLayaGuardrail = QLGuardrail
QLayaGuardrailError = QLGuardrailError
QLayaTriage = QLTriage
QLayaEvaluator = QLEvaluator
QLayaDecision = QLDecision
QLayaLowConfidenceError = QLLowConfidenceError

__all__ = [
    # QL-prefixed (current)
    "QLRouter",
    "QLGuardrail",
    "QLGuardrailError",
    "QLTriage",
    "QLEvaluator",
    "QLDecision",
    "QLSingleSelector",
    "QLMultiSelector",
    "QLQueryRouter",
    "QLCrewRouter",
    "QLTaskGuard",
    "QLTaskGuardError",
    "CrewRouteDecision",
    "QLLowConfidenceError",
    # Backward-compat aliases
    "QLayaRouter",
    "QLayaGuardrail",
    "QLayaGuardrailError",
    "QLayaTriage",
    "QLayaEvaluator",
    "QLayaDecision",
    "QLayaLowConfidenceError",
]
