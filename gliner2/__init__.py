__version__ = "1.2.1"

from .inference.engine import GLiNER2, RegexValidator
from .model import Extractor, ExtractorConfig
from .api_client import (
    GLiNER2API,
    GLiNER2APIError,
    AuthenticationError,
    ValidationError,
    ServerError,
)
from .training.lora import (
    LoRAConfig,
    LoRAAdapterConfig,
    LoRALayer,
    load_lora_adapter,
    save_lora_adapter,
    unload_lora_adapter,
    has_lora_adapter,
    apply_lora_to_model,
    merge_lora_weights,
    unmerge_lora_weights,
)