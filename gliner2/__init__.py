__version__ = "1.2.4"


# Register custom model type with transformers to avoid warnings
def _register_model():
    try:
        from transformers import AutoConfig, AutoModel

        from .model import Extractor, ExtractorConfig

        AutoConfig.register("extractor", ExtractorConfig)
        AutoModel.register(ExtractorConfig, Extractor)
    except Exception:
        # If registration fails (e.g., older transformers version), silently continue
        pass


_register_model()


def __getattr__(name: str):
    if name == "GLiNER2":
        from .inference.engine import GLiNER2

        return GLiNER2
    if name == "RegexValidator":
        from .inference.engine import RegexValidator

        return RegexValidator
    if name == "Extractor":
        from .model import Extractor

        return Extractor
    if name == "ExtractorConfig":
        from .model import ExtractorConfig

        return ExtractorConfig
    if name == "GLiNER2API":
        from .api_client import GLiNER2API

        return GLiNER2API
    if name == "GLiNER2APIError":
        from .api_client import GLiNER2APIError

        return GLiNER2APIError
    if name == "AuthenticationError":
        from .api_client import AuthenticationError

        return AuthenticationError
    if name == "ValidationError":
        from .api_client import ValidationError

        return ValidationError
    if name == "ServerError":
        from .api_client import ServerError

        return ServerError
    if name == "LoRAConfig":
        from .training.lora import LoRAConfig

        return LoRAConfig
    if name == "LoRAAdapterConfig":
        from .training.lora import LoRAAdapterConfig

        return LoRAAdapterConfig
    if name == "LoRALayer":
        from .training.lora import LoRALayer

        return LoRALayer
    if name == "load_lora_adapter":
        from .training.lora import load_lora_adapter

        return load_lora_adapter
    if name == "save_lora_adapter":
        from .training.lora import save_lora_adapter

        return save_lora_adapter
    if name == "unload_lora_adapter":
        from .training.lora import unload_lora_adapter

        return unload_lora_adapter
    if name == "has_lora_adapter":
        from .training.lora import has_lora_adapter

        return has_lora_adapter
    if name == "apply_lora_to_model":
        from .training.lora import apply_lora_to_model

        return apply_lora_to_model
    if name == "merge_lora_weights":
        from .training.lora import merge_lora_weights

        return merge_lora_weights
    if name == "unmerge_lora_weights":
        from .training.lora import unmerge_lora_weights

        return unmerge_lora_weights
    raise AttributeError(f"module 'gliner2' has no attribute '{name}'")


__all__ = [
    "GLiNER2",
    "RegexValidator",
    "Extractor",
    "ExtractorConfig",
    "GLiNER2API",
    "GLiNER2APIError",
    "AuthenticationError",
    "ValidationError",
    "ServerError",
    "LoRAConfig",
    "LoRAAdapterConfig",
    "LoRALayer",
    "load_lora_adapter",
    "save_lora_adapter",
    "unload_lora_adapter",
    "has_lora_adapter",
    "apply_lora_to_model",
    "merge_lora_weights",
    "unmerge_lora_weights",
]
