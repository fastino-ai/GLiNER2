"""Lazy preprocessing exports; word splitting and schemas remain torch-free."""

from __future__ import annotations

import importlib

_LAZY = {
    "build_query_layout": ("gliner2.processing.layouts", "build_query_layout"),
    "SpanAnnotation": ("gliner2.processing.layouts", "SpanAnnotation"),
    "validate_target_graph": ("gliner2.processing.layouts", "validate_target_graph"),
    "MentionTarget": ("gliner2.processing.targets", "MentionTarget"),
    "ChoiceTarget": ("gliner2.processing.targets", "ChoiceTarget"),
    "InstanceTarget": ("gliner2.processing.targets", "InstanceTarget"),
    "EdgeTarget": ("gliner2.processing.targets", "EdgeTarget"),
    "LiteralTarget": ("gliner2.processing.targets", "LiteralTarget"),
    "RecordFieldTarget": ("gliner2.processing.targets", "RecordFieldTarget"),
    "RecordTarget": ("gliner2.processing.targets", "RecordTarget"),
    "SurfaceSpan": ("gliner2.processing.surface", "SurfaceSpan"),
    "TargetGraph": ("gliner2.processing.targets", "TargetGraph"),
    "PaddedTargetBatch": ("gliner2.processing.targets", "PaddedTargetBatch"),
    "TargetCapacityError": ("gliner2.processing.targets", "TargetCapacityError"),
    "char_span_to_word_boundaries": ("gliner2.processing.targets", "char_span_to_word_boundaries"),
    "inclusive_tokens_to_boundary_pair": (
        "gliner2.processing.targets",
        "inclusive_tokens_to_boundary_pair",
    ),
    "word_boundaries_to_char_span": ("gliner2.processing.targets", "word_boundaries_to_char_span"),
    "build_boundary_targets": ("gliner2.processing.targets", "build_boundary_targets"),
    "pad_target_graphs": ("gliner2.processing.targets", "pad_target_graphs"),
    "apply_truncation_policy": ("gliner2.processing.targets", "apply_truncation_policy"),
    "normalize_surface_occurrences": (
        "gliner2.processing.targets",
        "normalize_surface_occurrences",
    ),
    "FieldCardinality": ("gliner2.processing.records", "FieldCardinality"),
    "RecordFieldSpec": ("gliner2.processing.records", "RecordFieldSpec"),
    "RecordSpec": ("gliner2.processing.records", "RecordSpec"),
    "compile_record_specs": ("gliner2.processing.records", "compile_record_specs"),
    "normalize_record_metadata": ("gliner2.processing.records", "normalize_record_metadata"),
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module 'gliner2.processing' has no attribute {name!r}")
    module, attribute = _LAZY[name]
    value = getattr(importlib.import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
