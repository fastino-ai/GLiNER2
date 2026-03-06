__version__ = "1.2.4"

from .inference.engine import GLiNER2, RegexValidator
from .model import Extractor, ExtractorConfig
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

# API client requires optional dependencies
try:
    from .api_client import (
        GLiNER2API,
        GLiNER2APIError,
        AuthenticationError,
        ValidationError,
        ServerError,
    )
    _api_available = True
except ImportError:
    _api_available = False

    # Define stubs for type hints and documentation
    class GLiNER2APIError(Exception):
        """Base exception for GLiNER2 API errors."""
        pass

    class AuthenticationError(GLiNER2APIError):
        """Raised when API key is invalid or expired."""
        pass

    class ValidationError(GLiNER2APIError):
        """Raised when request data is invalid."""
        pass

    class ServerError(GLiNER2APIError):
        """Raised when server encounters an error."""
        pass

    class GLiNER2API:
        """
        API-based GLiNER2 client that mirrors the local model interface.

        Note: This class requires the optional 'api' dependencies.
        Install with: pip install gliner2[api]
        """
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "API client requires optional dependencies. "
                "Install with: pip install gliner2[api]"
            )
