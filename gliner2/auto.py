"""Architecture-dispatching loader for GLiNER2 extractors.

``AutoExtractor`` resolves the architecture from a checkpoint's config and
returns the matching public model class. Legacy checkpoints without an
``architecture`` field load as ``"span"``.
"""

from __future__ import annotations

import json
import os
from typing import Any, ClassVar, Dict, Optional, Type

from gliner2.configuration import (
    ExtractorConfig,
    architecture_from_config,
    normalize_architecture,
)
from gliner2.models.loading import (
    HUB_LOAD_OPTIONS,
    checkpoint_file,
    split_load_kwargs,
)


class UnknownArchitectureError(ValueError):
    pass


class ArchitectureMismatchError(ValueError):
    pass


class ArchitectureRegistrationError(ValueError):
    pass


# Backward-compatible private alias; the shared loader owns the option set.
_HUB_LOAD_KEYS = HUB_LOAD_OPTIONS


def _ensure_registered() -> None:
    """Import the engine module so built-in architectures self-register."""
    if set(AutoExtractor._registry) >= {"span", "boundary"}:
        return
    # Importing the engine registers SpanExtractor and BoundaryExtractor.
    import gliner2.inference.engine  # noqa: F401


class AutoExtractor:
    """Registry-based dispatcher over extractor architectures."""

    _registry: ClassVar[Dict[str, Type]] = {}

    # -- registry -----------------------------------------------------------
    @classmethod
    def register(
        cls,
        architecture: str,
        model_class: Type,
        *,
        exist_ok: bool = False,
    ) -> None:
        name = normalize_architecture(architecture)
        if name in cls._registry and not exist_ok:
            if cls._registry[name] is model_class:
                return
            raise ArchitectureRegistrationError(
                f"Architecture {name!r} is already registered to "
                f"{cls._registry[name]!r}. Pass exist_ok=True to override."
            )
        cls._registry[name] = model_class

    @classmethod
    def _resolve_class(cls, architecture: str) -> Type:
        _ensure_registered()
        name = normalize_architecture(architecture)
        if name not in cls._registry:
            raise UnknownArchitectureError(
                f"No model class registered for architecture {name!r}. "
                f"Registered: {sorted(cls._registry)}"
            )
        return cls._registry[name]

    # -- loading ------------------------------------------------------------
    @classmethod
    def from_config(cls, config: ExtractorConfig, **kwargs: Any):
        architecture = architecture_from_config(config)
        model_class = cls._resolve_class(architecture)
        return model_class(config, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args: Any,
        architecture: Optional[str] = None,
        config: Optional[ExtractorConfig] = None,
        allow_architecture_override: bool = False,
        **kwargs: Any,
    ):
        model_kwargs, hub_kwargs = split_load_kwargs(
            kwargs, context="AutoExtractor.from_pretrained"
        )
        source = str(pretrained_model_name_or_path)
        if _is_local_adapter_dir(source, hub_kwargs.get("subfolder")):
            return _load_saved_adapter(
                cls, source, model_args, architecture, config,
                allow_architecture_override, model_kwargs, hub_kwargs,
            )

        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

        try:
            return cls._from_full_checkpoint(
                source, model_args, architecture, config,
                allow_architecture_override, model_kwargs, hub_kwargs,
            )
        except (FileNotFoundError, EntryNotFoundError, LocalEntryNotFoundError):
            if os.path.isdir(source) or _hub_adapter_config(source, hub_kwargs) is None:
                raise
            return _load_saved_adapter(
                cls, source, model_args, architecture, config,
                allow_architecture_override, model_kwargs, hub_kwargs,
            )

    @classmethod
    def _from_full_checkpoint(
        cls,
        source: str,
        model_args: Any,
        architecture: Optional[str],
        config: Optional[ExtractorConfig],
        allow_architecture_override: bool,
        model_kwargs: Dict[str, Any],
        hub_kwargs: Dict[str, Any],
    ):
        if config is None:
            config = _load_config(source, hub_kwargs)

        saved_architecture = architecture_from_config(config)
        requested_architecture = (
            normalize_architecture(architecture)
            if architecture is not None
            else saved_architecture
        )

        if requested_architecture != saved_architecture and not allow_architecture_override:
            raise ArchitectureMismatchError(
                f"Checkpoint architecture is {saved_architecture!r}, "
                f"but {requested_architecture!r} was requested. "
                "Span and boundary heads are not checkpoint-compatible, and "
                "automatic architecture conversion is not supported. Load the "
                "checkpoint with its saved architecture or initialize and train "
                "a separate boundary checkpoint."
            )

        model_class = cls._resolve_class(requested_architecture)
        return model_class.from_pretrained(
            source,
            *model_args,
            config=config,
            **model_kwargs,
            **hub_kwargs,
        )


def _local_checkpoint_dir(path: str, subfolder: Optional[str] = None) -> str:
    return os.path.join(path, subfolder) if subfolder else path


def _is_local_adapter_dir(path: str, subfolder: Optional[str] = None) -> bool:
    """True when ``path`` (with optional ``subfolder``) is a PEFT adapter directory."""
    target = _local_checkpoint_dir(path, subfolder)
    if not os.path.isdir(target):
        return False
    if not os.path.isfile(os.path.join(target, "adapter_config.json")):
        return False
    if os.path.isfile(os.path.join(target, "model.safetensors")):
        return False
    if os.path.isfile(os.path.join(target, "pytorch_model.bin")):
        return False
    return True


def _adapter_base_id(adapter_config_path: str) -> str:
    with open(adapter_config_path, encoding="utf-8") as handle:
        data = json.load(handle)
    base = data.get("base_model_name_or_path")
    if not isinstance(base, str) or not base.strip():
        raise ValueError(
            f"{adapter_config_path} has no base_model_name_or_path; "
            "the adapter cannot be loaded without the model it was trained on"
        )
    return base


def _extractor_config_beside_adapter(source: str, hub_kwargs: Dict[str, Any]):
    """Load the extractor ``config.json`` saved beside an adapter, when present."""
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    try:
        config_file = checkpoint_file(str(source), "config.json", hub_kwargs)
    except (FileNotFoundError, EntryNotFoundError, LocalEntryNotFoundError):
        return None
    return ExtractorConfig.from_pretrained(config_file)


def _hub_adapter_config(repo_id: str, hub_kwargs: Dict[str, Any]):
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    try:
        return checkpoint_file(str(repo_id), "adapter_config.json", hub_kwargs)
    except (FileNotFoundError, EntryNotFoundError, LocalEntryNotFoundError):
        return None


def _load_saved_adapter(
    cls,
    source: str,
    model_args: Any,
    architecture: Optional[str],
    config: Optional[ExtractorConfig],
    allow_architecture_override: bool,
    model_kwargs: Dict[str, Any],
    hub_kwargs: Dict[str, Any],
):
    """Load the base model referenced by ``adapter_config.json``, then attach the adapter."""
    if os.path.isdir(source):
        target = _local_checkpoint_dir(source, hub_kwargs.get("subfolder"))
        adapter_config = os.path.join(target, "adapter_config.json")
    else:
        adapter_config = _hub_adapter_config(source, hub_kwargs)
        if adapter_config is None:
            raise FileNotFoundError(f"{source} has no adapter_config.json")
    base_id = _adapter_base_id(adapter_config)
    if config is None:
        config = _extractor_config_beside_adapter(source, hub_kwargs)
    base_hub_kwargs = {k: v for k, v in hub_kwargs.items() if k != "subfolder"}
    base = cls.from_pretrained(
        base_id,
        *model_args,
        architecture=architecture,
        config=config,
        allow_architecture_override=allow_architecture_override,
        **model_kwargs,
        **base_hub_kwargs,
    )
    from peft import PeftModel

    peft_kwargs = {}
    for key in ("subfolder", "revision", "token", "cache_dir", "force_download", "local_files_only"):
        if key in hub_kwargs:
            peft_kwargs[key] = hub_kwargs[key]
    loaded = PeftModel.from_pretrained(base, source, **peft_kwargs)
    loaded.config = base.config
    return loaded


def _load_config(path, hub_kwargs: Dict[str, Any]) -> ExtractorConfig:
    """Load an ``ExtractorConfig`` from a local dir or the Hub."""
    config_file = checkpoint_file(str(path), "config.json", hub_kwargs)
    return ExtractorConfig.from_pretrained(config_file)
