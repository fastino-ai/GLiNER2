"""Public GLiNER2 package surface.

Schema construction, API access, and training-data utilities are intentionally
available from the torch-free base install. Model, configuration, and LoRA
objects remain lazy so importing :mod:`gliner2` never imports the ML stack.
"""

__version__ = "2.1.0"

from .api_client import (
    AuthenticationError,
    GLiNER2API,
    GLiNER2APIError,
    ServerError,
    ValidationError,
)
from .inference.schema import RegexValidator, Schema, StructureBuilder
from .inference.schema_model import (
    ClassificationInput,
    FieldInput,
    SchemaInput,
    StructureInput,
)
from .training.data import InputExample, TrainingDataset

# Concise documented alias; keep GLiNER2API for backward compatibility.
API = GLiNER2API

_LAZY = {
    "GLiNER2": ("gliner2.inference.engine", "GLiNER2"),
    "Extractor": ("gliner2.model", "Extractor"),
    "ExtractorConfig": ("gliner2.configuration", "ExtractorConfig"),
    "AutoExtractor": ("gliner2.auto", "AutoExtractor"),
    "SpanExtractor": ("gliner2.inference.engine", "SpanExtractor"),
    "BoundaryExtractor": ("gliner2.inference.engine", "BoundaryExtractor"),
    "UnknownArchitectureError": ("gliner2.auto", "UnknownArchitectureError"),
    "ArchitectureMismatchError": ("gliner2.auto", "ArchitectureMismatchError"),
    "ArchitectureRegistrationError": (
        "gliner2.auto",
        "ArchitectureRegistrationError",
    ),
    "AttributeGroup": ("gliner2.inference.schema", "AttributeGroup"),
    "LoRAConfig": ("gliner2.training.lora", "LoRAConfig"),
    "LoRAAdapterConfig": ("gliner2.training.lora", "LoRAAdapterConfig"),
    "LoRALayer": ("gliner2.training.lora", "LoRALayer"),
    "load_lora_adapter": ("gliner2.training.lora", "load_lora_adapter"),
    "save_lora_adapter": ("gliner2.training.lora", "save_lora_adapter"),
    "unload_lora_adapter": ("gliner2.training.lora", "unload_lora_adapter"),
    "has_lora_adapter": ("gliner2.training.lora", "has_lora_adapter"),
    "apply_lora_to_model": ("gliner2.training.lora", "apply_lora_to_model"),
    "merge_lora_weights": ("gliner2.training.lora", "merge_lora_weights"),
    "unmerge_lora_weights": ("gliner2.training.lora", "unmerge_lora_weights"),
}

__all__ = [
    "API",
    "AuthenticationError",
    "ClassificationInput",
    "FieldInput",
    "GLiNER2API",
    "GLiNER2APIError",
    "InputExample",
    "RegexValidator",
    "Schema",
    "SchemaInput",
    "ServerError",
    "StructureBuilder",
    "StructureInput",
    "TrainingDataset",
    "ValidationError",
    "__version__",
]


def __getattr__(name: str):
    try:
        mod_path, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'gliner2' has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(mod_path), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(list(globals()) + list(_LAZY)))
