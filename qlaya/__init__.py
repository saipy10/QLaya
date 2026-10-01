"""QLaya: fast, non-autoregressive System 1 decision engine with calibrated
probabilities and user-selectable quantized model variants.

Quick start::

    import qlaya
    router = qlaya.Router()

    # Use a specific QLaya quantized variant from benchmark_results.json:
    router = qlaya.Router(model="QLaya-int8")       # ONNX INT8, recommended
    router = qlaya.Router(model="QLaya-6L-int8")    # 6L INT8, 38 ms edge model
    router = qlaya.Router(model="QLaya-6L-int4")    # 6L INT4, 142 MB

    # List all available QLaya model variants:
    print(qlaya.QLAYA_MODEL_IDS)
"""

from .email import clean_email_body, email_state
from .hooks import AsyncHook, BaseHook, Hook, PredictContext, PredictHook
from .lang import analyse as detect_language
from .lang import detect_script, is_english
from .presets import (
    email_questions,
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)
from .router import (
    DEFAULT_MODELS,
    QLAYA_MODELS,
    QLAYA_MODEL_IDS,
    QLAYA_PRIMARY_MODEL_IDS,
    QLAYA_ONNX_FILES,
    RouteDecision,
    Router,
    resolve_qlaya_model,
    resolve_onnx_filename,
)
from .structured import DecisionResult, decide, decide_batch

__version__ = "0.4.4"
__qlaya_version__ = __version__

# Routing, language detection and email cleaning are pure Python. The torch-backed names are
# resolved lazily so that `import qlaya` -- and therefore `from qlaya import Router` or
# `from qlaya.lang import detect_script` -- does not pay torch's import time and memory.
_LAZY_ATTRS = {
    "Agent": (".agent", "Agent"),
    "RLAgent": (".agent", "RLAgent"),
    "load": (".agent", "load"),
    "proper_reward": (".common", "proper_reward"),
    "td_lambda_targets": (".common", "td_lambda_targets"),
    "ece_score": (".common", "ece_score"),
    "answer_confidence": (".common", "answer_confidence"),
    "confidence_from_probs": (".common", "confidence_from_probs"),
    "check_min_confidence": (".confidence", "check_min_confidence"),
    "flag_low_confidence": (".confidence", "flag_low_confidence"),
    "render_options": (".common", "render_options"),
    "QTYPES": (".common", "QTYPES"),
    "QTYPE_NAMES": (".common", "QTYPE_NAMES"),
    "shortlist_choice": (".shortlist", "shortlist_choice"),
    "predict_shortlist": (".shortlist", "predict_shortlist"),
    "embed_fn_from_agent": (".shortlist", "embed_fn_from_agent"),
    "cached_embed_fn": (".shortlist", "cached_embed_fn"),
    # Integration classes — QL-prefixed (QLaya* kept as backward-compat aliases below)
    "QLRouter": (".integrations", "QLRouter"),
    "QLGuardrail": (".integrations", "QLGuardrail"),
    "QLGuardrailError": (".integrations", "QLGuardrailError"),
    "QLTriage": (".integrations", "QLTriage"),
    "QLEvaluator": (".integrations", "QLEvaluator"),
    "QLDecision": (".integrations", "QLDecision"),
    # Backward-compat aliases (deprecated, will be removed in 1.0)
    "QLayaRouter": (".integrations", "QLRouter"),
    "QLayaGuardrail": (".integrations", "QLGuardrail"),
    "QLayaGuardrailError": (".integrations", "QLGuardrailError"),
    "QLayaTriage": (".integrations", "QLTriage"),
    "QLayaEvaluator": (".integrations", "QLEvaluator"),
    "QLayaDecision": (".integrations", "QLDecision"),
}


def __getattr__(name):
    try:
        module_name, attr = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError("module %r has no attribute %r" % (__name__, name)) from None
    import importlib

    value = getattr(importlib.import_module(module_name, __name__), attr)
    globals()[name] = value      # cache: __getattr__ runs at most once per name
    return value


def __dir__():
    return sorted(list(globals()) + list(_LAZY_ATTRS))


__all__ = [
    # Core
    "Agent",
    "RLAgent",
    "load",
    "Router",
    "RouteDecision",
    "DEFAULT_MODELS",
    # QLaya model registry
    "QLAYA_MODELS",
    "QLAYA_MODEL_IDS",
    "QLAYA_PRIMARY_MODEL_IDS",
    "QLAYA_ONNX_FILES",
    "resolve_qlaya_model",
    "resolve_onnx_filename",
    # Shortlist
    "shortlist_choice",
    "predict_shortlist",
    "embed_fn_from_agent",
    "cached_embed_fn",
    # Language
    "detect_language",
    "detect_script",
    "is_english",
    # Email
    "clean_email_body",
    "email_questions",
    "email_state",
    # Presets
    "guard_questions",
    "moderation_questions",
    "router_questions",
    "triage_questions",
    # Common utilities
    "proper_reward",
    "td_lambda_targets",
    "ece_score",
    "answer_confidence",
    "confidence_from_probs",
    "check_min_confidence",
    "flag_low_confidence",
    "render_options",
    "QTYPES",
    "QTYPE_NAMES",
    # Integration classes (QL-prefixed)
    "QLRouter",
    "QLGuardrail",
    "QLGuardrailError",
    "QLTriage",
    "QLEvaluator",
    "QLDecision",
    # Backward-compat aliases
    "QLayaRouter",
    "QLayaGuardrail",
    "QLayaGuardrailError",
    "QLayaTriage",
    "QLayaEvaluator",
    "QLayaDecision",
    # Hooks
    "PredictContext",
    "PredictHook",
    "Hook",
    "BaseHook",
    "AsyncHook",
    # Structured
    "decide",
    "decide_batch",
    "DecisionResult",
    # Version
    "__version__",
    "__qlaya_version__",
]
