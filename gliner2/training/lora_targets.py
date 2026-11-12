"""Architecture-aware LoRA target resolution.

This module keeps target discovery separate from the legacy compatibility
shims in :mod:`gliner2.training.lora`.
"""
from __future__ import annotations

import torch.nn as nn
from peft.tuners.tuners_utils import BaseTunerLayer

ENCODER_PATTERNS = ("query", "key", "value", "dense")
# Legacy fallback for models that predate ``task_module_names()``.
TASK_MODULES = ("span_rep", "classifier", "count_embed", "count_pred")


def _task_module_names(model: nn.Module) -> tuple[str, ...]:
    """Return architecture-specific task-head names with a legacy fallback."""
    method = getattr(model, "task_module_names", None)
    if method is None:
        return TASK_MODULES
    try:
        return tuple(method())
    except Exception:  # noqa: BLE001 - support unusual third-party models
        return TASK_MODULES


def _has_module(model: nn.Module, name: str) -> bool:
    return any(
        module_name == name or module_name.startswith(f"{name}.")
        for module_name, _ in model.named_modules()
    )


def _alias_targets(model: nn.Module, alias: str) -> tuple[str, ...]:
    """Expand a high-level alias into concrete task-module prefixes."""
    task_modules = _task_module_names(model)
    if alias == "all_task_heads":
        return task_modules
    if alias == "classification_head":
        return ("classifier",) if _has_module(model, "classifier") else ()
    if alias == "extractive_head":
        if _has_module(model, "boundary_head"):
            return ("boundary_head",)
        return tuple(
            name
            for name in ("span_rep", "count_embed", "count_pred")
            if _has_module(model, name)
        )
    if alias == "relation_head":
        return ("relation_scorer",) if _has_module(model, "relation_scorer") else ()
    if alias == "record_head":
        return ("record_decoder",) if _has_module(model, "record_decoder") else ()
    return ()


def _resolve_targets(model: nn.Module, targets: list[str]) -> list[str]:
    """Map public target names to concrete ``nn.Linear`` module paths."""
    task_modules = set(_task_module_names(model)) | set(TASK_MODULES)
    head_prefixes = {
        expanded
        for target in targets
        for expanded in _alias_targets(model, target)
    }
    head_prefixes.update(target for target in targets if target in task_modules)

    selected: list[str] = []
    wrapped: list[str] = []
    for name, module in model.named_modules():
        if any(name.startswith(prefix + ".") for prefix in wrapped):
            continue
        if isinstance(module, BaseTunerLayer):
            wrapped.append(name)
            module = module.get_base_layer()
        if not isinstance(module, nn.Linear):
            continue
        local_name = name.rsplit(".", 1)[-1]
        for target in targets:
            if (
                target == "encoder"
                and name.startswith("encoder.")
                and any(pattern in local_name for pattern in ENCODER_PATTERNS)
            ):
                selected.append(name)
            elif (
                target.startswith("encoder.")
                and name.startswith("encoder.")
                and target.split(".", 1)[1] in local_name
            ):
                selected.append(name)
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in head_prefixes
        ):
            selected.append(name)
    return sorted(set(selected))


__all__ = [
    "ENCODER_PATTERNS",
    "TASK_MODULES",
    "_alias_targets",
    "_resolve_targets",
    "_task_module_names",
]
