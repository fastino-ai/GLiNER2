__version__ = "1.3.1"

from .inference.schema import Schema, StructureBuilder, RegexValidator
from .inference.schema_model import (
    SchemaInput,
    FieldInput,
    StructureInput,
    ClassificationInput,
)
from .api_client import (
    GLiNER2API,
    GLiNER2APIError,
    AuthenticationError,
    ValidationError,
    ServerError,
)

_LAZY = {
    "GLiNER2":              ("gliner2.inference.engine", "GLiNER2"),
    "Extractor":            ("gliner2.model",            "Extractor"),
    "ExtractorConfig":      ("gliner2.model",            "ExtractorConfig"),
    "LoRAConfig":           ("gliner2.training.lora",    "LoRAConfig"),
    "LoRAAdapterConfig":    ("gliner2.training.lora",    "LoRAAdapterConfig"),
    "LoRALayer":            ("gliner2.training.lora",    "LoRALayer"),
    "load_lora_adapter":    ("gliner2.training.lora",    "load_lora_adapter"),
    "save_lora_adapter":    ("gliner2.training.lora",    "save_lora_adapter"),
    "unload_lora_adapter":  ("gliner2.training.lora",    "unload_lora_adapter"),
    "has_lora_adapter":     ("gliner2.training.lora",    "has_lora_adapter"),
    "apply_lora_to_model":  ("gliner2.training.lora",    "apply_lora_to_model"),
    "merge_lora_weights":   ("gliner2.training.lora",    "merge_lora_weights"),
    "unmerge_lora_weights": ("gliner2.training.lora",    "unmerge_lora_weights"),
}


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
