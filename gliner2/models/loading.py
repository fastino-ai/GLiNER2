"""Shared checkpoint-loading helpers for extractor architectures.

The public loaders intentionally accept a small, explicit option surface.
Hub-only options are used while resolving checkpoint files and never reach
model constructors.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Mapping, MutableMapping, Tuple

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
from safetensors.torch import load_file


logger = logging.getLogger(__name__)

HUB_LOAD_OPTIONS = frozenset(
    {
        "cache_dir",
        "force_download",
        "local_files_only",
        "token",
        "revision",
        "subfolder",
        "proxies",
    }
)
MODEL_LOAD_OPTIONS = frozenset(
    {
        "attention_backend",
        "quantize",
        "compile",
        "disentangled_flash_min_padding",
        "disentangled_flash_min_work",
        "disentangled_flash_packed",
        "disentangled_flash_tuning",
        "map_location",
        "use_flashdeberta",
        "word_splitter",
    }
)
LOAD_OPTIONS = HUB_LOAD_OPTIONS | MODEL_LOAD_OPTIONS
ATTENTION_BACKENDS = frozenset(
    {"standard", "flashdeberta", "disentangled_flash"}
)


def split_load_kwargs(
    kwargs: Mapping[str, Any],
    *,
    context: str = "from_pretrained",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Validate and split model options from Hub-only options."""
    unknown = sorted(set(kwargs) - LOAD_OPTIONS)
    if unknown:
        raise TypeError(
            f"{context} does not accept {unknown}; accepted options are "
            f"{sorted(LOAD_OPTIONS)}."
        )
    model_kwargs = {key: kwargs[key] for key in MODEL_LOAD_OPTIONS if key in kwargs}
    hub_kwargs = {key: kwargs[key] for key in HUB_LOAD_OPTIONS if key in kwargs}
    return model_kwargs, hub_kwargs


def pop_attention_backend(
    model_options: MutableMapping[str, Any],
) -> Tuple[str | None, bool | None]:
    """Resolve the unified backend option and legacy FlashDeBERTa alias."""
    backend = model_options.pop("attention_backend", None)
    legacy_flash = model_options.pop("use_flashdeberta", None)

    if backend is not None:
        backend = str(backend).strip().lower()
        if backend not in ATTENTION_BACKENDS:
            raise ValueError(
                "attention_backend must be 'standard', 'flashdeberta', or "
                f"'disentangled_flash', got {backend!r}"
            )
    if legacy_flash is not None and not isinstance(legacy_flash, bool):
        raise TypeError(
            "use_flashdeberta must be a bool or None, got "
            f"{type(legacy_flash).__name__}"
        )

    if backend is None:
        if legacy_flash is None:
            return None, None
        backend = "flashdeberta" if legacy_flash else "standard"
    elif legacy_flash is not None:
        legacy_backend = "flashdeberta" if legacy_flash else "standard"
        if backend != legacy_backend:
            raise ValueError(
                "attention_backend conflicts with the legacy "
                "use_flashdeberta option"
            )

    return backend, backend == "flashdeberta"


def pop_disentangled_flash_options(
    model_options: MutableMapping[str, Any],
) -> Dict[str, Any]:
    """Map ``disentangled_flash_*`` load options to enable kwargs."""
    options = {}
    if "disentangled_flash_packed" in model_options:
        options["packed"] = model_options.pop("disentangled_flash_packed")
    if "disentangled_flash_min_padding" in model_options:
        options["packed_min_padding"] = model_options.pop(
            "disentangled_flash_min_padding"
        )
    if "disentangled_flash_min_work" in model_options:
        options["packed_min_work"] = model_options.pop("disentangled_flash_min_work")
    if "disentangled_flash_tuning" in model_options:
        options["tuning"] = model_options.pop("disentangled_flash_tuning")
    return options


def checkpoint_file(
    repo_or_dir: str,
    filename: str,
    hub_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Resolve one checkpoint file from a local directory or the Hub."""
    options = dict(hub_kwargs or {})
    subfolder = options.pop("subfolder", None)
    if os.path.isdir(str(repo_or_dir)):
        parts = [str(repo_or_dir)]
        if subfolder:
            parts.append(str(subfolder))
        parts.append(filename)
        return os.path.join(*parts)
    return hf_hub_download(
        repo_id=str(repo_or_dir),
        filename=filename,
        subfolder=subfolder,
        **options,
    )


def load_checkpoint_state_dict(
    repo_or_dir: str,
    hub_kwargs: Mapping[str, Any] | None = None,
) -> MutableMapping[str, torch.Tensor]:
    """Load safetensors first, falling back only when that file is absent."""
    try:
        safetensors_path = checkpoint_file(
            repo_or_dir, "model.safetensors", hub_kwargs
        )
    except (FileNotFoundError, EntryNotFoundError, LocalEntryNotFoundError):
        safetensors_path = None

    if safetensors_path is not None and os.path.isfile(safetensors_path):
        try:
            return load_file(safetensors_path)
        except Exception as exc:  # corrupt/incompatible data must not be hidden
            raise RuntimeError(
                f"Failed to read safetensors checkpoint {safetensors_path!r}: {exc}"
            ) from exc

    try:
        pytorch_path = checkpoint_file(
            repo_or_dir, "pytorch_model.bin", hub_kwargs
        )
    except (FileNotFoundError, EntryNotFoundError, LocalEntryNotFoundError) as exc:
        raise FileNotFoundError(
            f"Checkpoint {repo_or_dir!r} contains neither model.safetensors "
            "nor pytorch_model.bin."
        ) from exc
    if not os.path.isfile(pytorch_path):
        raise FileNotFoundError(
            f"Checkpoint weight file does not exist: {pytorch_path!r}"
        )
    try:
        return torch.load(pytorch_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read PyTorch checkpoint {pytorch_path!r}: {exc}"
        ) from exc


def reconcile_encoder_embeddings(
    model: Any,
    state_dict: MutableMapping[str, torch.Tensor],
) -> None:
    """Make a saved encoder vocabulary compatible with the tokenizer model."""
    keys = [
        key
        for key in state_dict
        if key.startswith("encoder.") and key.endswith("word_embeddings.weight")
    ]
    if len(keys) != 1:
        return

    key = keys[0]
    saved = state_dict[key]
    current = model.encoder.get_input_embeddings().weight
    if saved.shape == current.shape:
        return
    if saved.ndim != 2 or current.ndim != 2 or saved.shape[1] != current.shape[1]:
        raise RuntimeError(
            f"Cannot reconcile {key}: checkpoint shape {tuple(saved.shape)} "
            f"does not match model shape {tuple(current.shape)}."
        )

    if saved.shape[0] > current.shape[0]:
        model.encoder.resize_token_embeddings(saved.shape[0])
        logger.info(
            "Resized encoder vocabulary from %d to %d entries for checkpoint load",
            current.shape[0],
            saved.shape[0],
        )
        return

    resized = current.detach().to(device=saved.device, dtype=saved.dtype).clone()
    resized[: saved.shape[0]].copy_(saved)
    state_dict[key] = resized
    logger.info(
        "Extended checkpoint encoder vocabulary from %d to %d entries",
        saved.shape[0],
        current.shape[0],
    )


def apply_post_load_options(
    model: Any,
    *,
    map_location: Any = None,
    quantize: bool = False,
    compile_model: bool = False,
    compile_dynamic: bool | None = None,
    attention_backend: str | None = None,
    disentangled_flash_options: Mapping[str, Any] | None = None,
) -> Any:
    """Apply device, precision, attention, then compilation options."""
    if not isinstance(quantize, bool):
        raise TypeError(f"quantize must be a bool, got {type(quantize).__name__}")
    if not isinstance(compile_model, bool):
        raise TypeError(f"compile must be a bool, got {type(compile_model).__name__}")
    if attention_backend is not None and attention_backend not in ATTENTION_BACKENDS:
        raise ValueError(f"unknown attention backend {attention_backend!r}")
    if attention_backend == "disentangled_flash" and compile_model:
        raise ValueError(
            "attention_backend='disentangled_flash' and compile=True cannot "
            "be enabled together; "
            "DisentangledFlash manages its own prepared attention path."
        )
    if disentangled_flash_options and attention_backend != "disentangled_flash":
        raise ValueError(
            "disentangled_flash_packed, disentangled_flash_min_padding, "
            "disentangled_flash_min_work and disentangled_flash_tuning "
            "require attention_backend='disentangled_flash'."
        )

    if map_location is not None:
        model = model.to(map_location)
    if quantize:
        model.quantize()
    if attention_backend == "disentangled_flash":
        model.eval()
        model.enable_disentangled_flash(**(disentangled_flash_options or {}))
    if compile_model:
        if compile_dynamic is None:
            model.compile()
        else:
            model.compile(dynamic=compile_dynamic)
    return model
