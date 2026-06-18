"""PEFT-backed LoRA utilities for GLiNER2.

New code should use ``Extractor.apply_lora()`` and PEFT's own save/load
(``PeftModel.save_pretrained`` / ``PeftModel.from_pretrained``).

Every public symbol from the pre-PEFT era is preserved for backwards
compatibility.  Each legacy entry point emits ``PendingDeprecationWarning``
on *call* (not on import) so existing user CI stays silent.
"""
from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from peft import LoraConfig as PeftLoraConfig, get_peft_model, PeftModel
from peft.tuners.lora.layer import LoraLayer as _PeftLoraLayer
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core (non-deprecated) — mirrors Gliner2Internal main
# ---------------------------------------------------------------------------

ENCODER_PATTERNS = ["query", "key", "value", "dense"]
TASK_MODULES = ["span_rep", "classifier", "count_embed", "count_pred"]


def _resolve_targets(model: nn.Module, targets: list[str]) -> list[str]:
    """Map high-level target names to concrete Linear layer paths.

    Args:
        model: The model to resolve targets against.
        targets: High-level target names, e.g. ``["encoder"]``,
            ``["encoder.query"]``, or task head names like ``["classifier"]``.

    Returns:
        Sorted list of fully-qualified module paths suitable for
        passing directly to ``peft.LoraConfig(target_modules=...)``.
    """
    selected: list[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        local = name.split(".")[-1]
        for t in targets:
            if t == "encoder" and name.startswith("encoder.") and any(p in local for p in ENCODER_PATTERNS):
                selected.append(name)
            elif t.startswith("encoder.") and name.startswith("encoder.") and t.split(".", 1)[1] in local:
                selected.append(name)
            elif t in TASK_MODULES and (name == t or name.startswith(f"{t}.")):
                selected.append(name)
    return sorted(set(selected))


def _deprecation(name: str, replacement: str) -> None:
    warnings.warn(
        f"{name} is deprecated; use {replacement} instead.",
        PendingDeprecationWarning,
        stacklevel=3,
    )


def _cast_lora_dtype(model: nn.Module) -> None:
    """Cast LoRA A/B weights to match their base-layer dtype (PR #4 fix)."""
    for mod in model.modules():
        if not isinstance(mod, _PeftLoraLayer):
            continue
        base = mod.get_base_layer()
        if base is None:
            continue
        dtype = next(base.parameters()).dtype
        for key in list(mod.lora_A):
            mod.lora_A[key] = mod.lora_A[key].to(dtype=dtype)
        for key in list(mod.lora_B):
            mod.lora_B[key] = mod.lora_B[key].to(dtype=dtype)


# ---------------------------------------------------------------------------
# Preserved dataclasses (PendingDeprecationWarning on construction)
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    """Configuration for LoRA parameter-efficient fine-tuning."""
    enabled: bool = False
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["encoder"])

    def __post_init__(self) -> None:
        _deprecation("LoRAConfig", "peft.LoraConfig + Extractor.apply_lora()")
        if self.r <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {self.r}")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be > 0, got {self.alpha}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {self.dropout}")
        if self.enabled and not self.target_modules:
            raise ValueError("target_modules cannot be empty when LoRA is enabled")


@dataclass
class LoRAAdapterConfig:
    """Serializable metadata for a saved LoRA adapter."""
    adapter_type: str = "lora"
    adapter_version: str = "1.0"
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=list)
    created_at: str = ""

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        with open(path / "adapter_config.json", "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> LoRAAdapterConfig:
        path = Path(path)
        cfg_file = path / "adapter_config.json" if path.is_dir() else path
        if not cfg_file.exists():
            raise FileNotFoundError(f"Adapter config not found at {cfg_file}")
        with open(cfg_file) as f:
            data = json.load(f)
        if "peft_type" in data:
            return cls(
                lora_r=data.get("r", 8),
                lora_alpha=data.get("lora_alpha", 16.0),
                lora_dropout=data.get("lora_dropout", 0.0),
                target_modules=data.get("target_modules", []),
            )
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def is_adapter_path(cls, path: Union[str, Path]) -> bool:
        path = Path(path)
        if path.is_dir():
            return (path / "adapter_config.json").exists()
        return path.name == "adapter_config.json" and path.exists()


# isinstance-compat alias
LoRALayer = _PeftLoraLayer


# ---------------------------------------------------------------------------
# Legacy function shims (each emits PendingDeprecationWarning)
# ---------------------------------------------------------------------------

def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
) -> Tuple[nn.Module, Dict[str, LoRALayer]]:
    _deprecation("apply_lora_to_model", "Extractor.apply_lora()")
    if not config.enabled:
        return model, {}
    peft_cfg = PeftLoraConfig(
        r=config.r, lora_alpha=config.alpha, lora_dropout=config.dropout,
        target_modules=_resolve_targets(model, config.target_modules),
        bias="none",
    )
    peft_model = get_peft_model(model, peft_cfg)
    _cast_lora_dtype(peft_model)
    layers = {n: m for n, m in peft_model.named_modules() if isinstance(m, _PeftLoraLayer)}
    return peft_model, layers


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    _deprecation("get_lora_parameters", "model.parameters() with requires_grad filter")
    return [p for p in model.parameters() if p.requires_grad]


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    _deprecation("get_lora_state_dict", "peft.get_peft_model_state_dict()")
    from peft import get_peft_model_state_dict
    if isinstance(model, PeftModel):
        return dict(get_peft_model_state_dict(model))
    return {}


def merge_lora_weights(model: nn.Module) -> int:
    _deprecation("merge_lora_weights", "PeftModel.merge_adapter()")
    if isinstance(model, PeftModel):
        model.merge_adapter()
        return sum(1 for m in model.modules() if isinstance(m, _PeftLoraLayer))
    return 0


def unmerge_lora_weights(model: nn.Module) -> int:
    _deprecation("unmerge_lora_weights", "PeftModel.unmerge_adapter()")
    if isinstance(model, PeftModel):
        model.unmerge_adapter()
        return sum(1 for m in model.modules() if isinstance(m, _PeftLoraLayer))
    return 0


def count_lora_parameters(model: nn.Module) -> Tuple[int, int, float]:
    _deprecation("count_lora_parameters", "manual parameter counting")
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    pct = (lora_params / total_params * 100) if total_params > 0 else 0.0
    return lora_params, total_params, pct


def print_lora_info(model: nn.Module, config: LoRAConfig) -> None:
    _deprecation("print_lora_info", "manual logging")
    lp, tp, pct = count_lora_parameters.__wrapped__(model) if hasattr(count_lora_parameters, "__wrapped__") else _count_lora_params_raw(model)
    n_layers = sum(1 for m in model.modules() if isinstance(m, _PeftLoraLayer))
    print(f"LoRA: r={config.r}, alpha={config.alpha}, layers={n_layers}, "
          f"trainable={lp:,}/{tp:,} ({pct:.2f}%)")


def _count_lora_params_raw(model: nn.Module) -> Tuple[int, int, float]:
    lp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tp = sum(p.numel() for p in model.parameters())
    return lp, tp, (lp / tp * 100) if tp > 0 else 0.0


def remove_lora_from_model(model: nn.Module) -> nn.Module:
    _deprecation("remove_lora_from_model", "PeftModel.merge_and_unload()")
    if isinstance(model, PeftModel):
        return model.merge_and_unload()
    return model


# ---------------------------------------------------------------------------
# Dual-format save / load
# ---------------------------------------------------------------------------

def _is_peft_native_dir(path: Path) -> bool:
    if (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists():
        return True
    cfg_file = path / "adapter_config.json"
    if cfg_file.exists():
        with open(cfg_file) as f:
            data = json.load(f)
        if "peft_type" in data:
            return True
    return False


def save_lora_adapter(model: nn.Module, save_path: Union[str, Path]) -> None:
    _deprecation("save_lora_adapter", "PeftModel.save_pretrained()")
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if isinstance(model, PeftModel):
        peft_cfg = model.peft_config.get("default")
        r = peft_cfg.r if peft_cfg else 8
        alpha = peft_cfg.lora_alpha if peft_cfg else 16.0
        dropout = peft_cfg.lora_dropout if peft_cfg else 0.0
        target_mods = list(peft_cfg.target_modules) if peft_cfg and peft_cfg.target_modules else []

        top_modules = sorted({n.split(".")[0] for n in target_mods if n})

        legacy_state: Dict[str, torch.Tensor] = {}
        for name, mod in model.named_modules():
            if not isinstance(mod, _PeftLoraLayer):
                continue
            clean_name = name.replace("base_model.model.", "").replace("base_model.", "")
            for adapter_key, mat in mod.lora_A.items():
                legacy_state[f"{clean_name}.lora_A"] = mat.weight.data if hasattr(mat, "weight") else mat.data
            for adapter_key, mat in mod.lora_B.items():
                legacy_state[f"{clean_name}.lora_B"] = mat.weight.data if hasattr(mat, "weight") else mat.data

        save_file(legacy_state, str(save_path / "adapter_weights.safetensors"))

        cfg = LoRAAdapterConfig(
            lora_r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=top_modules,
        )
        cfg.save(save_path)
    else:
        raise ValueError("Model is not a PeftModel; nothing to save.")


def load_lora_adapter(
    model: nn.Module,
    adapter_path: Union[str, Path],
    auto_unload: bool = True,
) -> Dict[str, LoRALayer]:
    _deprecation("load_lora_adapter", "PeftModel.from_pretrained()")
    adapter_path = Path(adapter_path)

    if auto_unload and isinstance(model, PeftModel):
        model = model.merge_and_unload()

    if _is_peft_native_dir(adapter_path):
        peft_model = PeftModel.from_pretrained(model, str(adapter_path))
        _cast_lora_dtype(peft_model)
    else:
        adapter_cfg = LoRAAdapterConfig.load(adapter_path)
        weights = load_file(str(adapter_path / "adapter_weights.safetensors"))

        resolved = _resolve_targets(model, adapter_cfg.target_modules)
        peft_cfg = PeftLoraConfig(
            r=adapter_cfg.lora_r, lora_alpha=adapter_cfg.lora_alpha,
            lora_dropout=adapter_cfg.lora_dropout,
            target_modules=resolved, bias="none",
        )
        peft_model = get_peft_model(model, peft_cfg)

        for name, mod in peft_model.named_modules():
            if not isinstance(mod, _PeftLoraLayer):
                continue
            clean = name.replace("base_model.model.", "").replace("base_model.", "")
            a_key, b_key = f"{clean}.lora_A", f"{clean}.lora_B"
            if a_key in weights and b_key in weights:
                base = mod.get_base_layer()
                dtype = next(base.parameters()).dtype if base is not None else torch.float32
                device = next(base.parameters()).device if base is not None else torch.device("cpu")
                for adapter_name, mat in mod.lora_A.items():
                    if hasattr(mat, "weight"):
                        mat.weight.data.copy_(weights[a_key].to(device=device, dtype=dtype))
                    else:
                        mat.data.copy_(weights[a_key].to(device=device, dtype=dtype))
                for adapter_name, mat in mod.lora_B.items():
                    if hasattr(mat, "weight"):
                        mat.weight.data.copy_(weights[b_key].to(device=device, dtype=dtype))
                    else:
                        mat.data.copy_(weights[b_key].to(device=device, dtype=dtype))

        _cast_lora_dtype(peft_model)

    layers = {n: m for n, m in peft_model.named_modules() if isinstance(m, _PeftLoraLayer)}
    return layers


def unload_lora_adapter(model: nn.Module) -> int:
    _deprecation("unload_lora_adapter", "PeftModel.merge_and_unload() or PeftModel.unload()")
    if isinstance(model, PeftModel):
        count = sum(1 for m in model.modules() if isinstance(m, _PeftLoraLayer))
        model.unload()
        return count
    return 0


def has_lora_adapter(model: nn.Module) -> bool:
    _deprecation("has_lora_adapter", "isinstance(model, PeftModel)")
    return isinstance(model, PeftModel) or any(isinstance(m, _PeftLoraLayer) for m in model.modules())


def get_adapter_config(model: nn.Module) -> Optional[LoRAAdapterConfig]:
    _deprecation("get_adapter_config", "model.peft_config")
    if not isinstance(model, PeftModel):
        return None
    peft_cfg = model.peft_config.get("default")
    if peft_cfg is None:
        return None
    return LoRAAdapterConfig(
        lora_r=peft_cfg.r,
        lora_alpha=peft_cfg.lora_alpha,
        lora_dropout=peft_cfg.lora_dropout,
        target_modules=sorted(peft_cfg.target_modules) if peft_cfg.target_modules else [],
    )
