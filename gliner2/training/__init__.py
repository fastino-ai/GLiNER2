"""GLiNER2 training and torch-free training-data utilities.

Trainer symbols are loaded on first access. This keeps ``import gliner2`` and
``import gliner2.training`` usable without torch, transformers, or PEFT.
"""

from __future__ import annotations

import importlib

from gliner2.training.data import (
    ChoiceField,
    Classification,
    DataFormat,
    DataLoader_Factory,
    DataValidationError,
    InputExample,
    Relation,
    Structure,
    TrainDataInput,
    TrainingDataset,
    create_classification_example,
    create_entity_example,
    create_relation_example,
    create_structure_example,
)

_LAZY = {
    "TrainingSession": ("gliner2.training.session", "TrainingSession"),
    "ExtractorTrainer": ("gliner2.training.trainer", "ExtractorTrainer"),
    "GLiNER2Trainer": ("gliner2.training.trainer", "GLiNER2Trainer"),
    "TrainingConfig": ("gliner2.training.trainer", "TrainingConfig"),
    "TrainingMetrics": ("gliner2.training.trainer", "TrainingMetrics"),
    "ExtractorDataset": ("gliner2.training.trainer", "ExtractorDataset"),
    "ExtractorCollator": ("gliner2.training.trainer", "ExtractorCollator"),
    "train_gliner2": ("gliner2.training.trainer", "train_gliner2"),
}

__all__ = [
    "TrainingSession",
    "ChoiceField",
    "Classification",
    "DataFormat",
    "DataLoader_Factory",
    "DataValidationError",
    "InputExample",
    "Relation",
    "Structure",
    "TrainDataInput",
    "TrainingDataset",
    "ExtractorTrainer",
    "GLiNER2Trainer",
    "TrainingConfig",
    "TrainingMetrics",
    "ExtractorDataset",
    "ExtractorCollator",
    "create_classification_example",
    "create_entity_example",
    "create_relation_example",
    "create_structure_example",
    "train_gliner2",
]


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'gliner2.training' has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
